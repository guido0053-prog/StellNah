import math
import time
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
import osmnx as ox
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP_NAME = "StellNah"

# Kriterien (Luftlinie)
WALK_M = 300          # Einkauf max 300 m
CENTER_M = 1500       # zentrumsnah max 1,5 km

# Umkreis pro Stellplatz für POI-Abfrage (klein & stabil)
POI_SEARCH_M = 800

# Render-Free Schutz: nur die nächsten X Kandidaten prüfen (nach Nähe zum Zentrum)
MAX_SITES_TO_CHECK = 25

# OSMnx stabilisieren
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_timeout = 180
ox.settings.http_headers = {"User-Agent": "StellNah (private use)"}

app = FastAPI(title=f"{APP_NAME} API")

# CORS: Frontend darf Backend aufrufen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # für private Solo-App ok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    zielort: str
    suchradius_km: float


# -------------------------
# Hilfsfunktionen
# -------------------------

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def maps_search_link(q: str) -> str:
    return "https://www.google.com/maps/search/?api=1&query=" + q.replace(" ", "+")


def safe_features_from_point(point, dist, tags, retries=2, wait_s=2):
    """
    Overpass kann manchmal zicken. Wir versuchen es ein paar Mal.
    """
    last_err = None
    for i in range(retries + 1):
        try:
            return ox.features_from_point(point, dist=dist, tags=tags)
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(wait_s)
    raise last_err


def geocode_robust(query: str, retries: int = 2) -> Tuple[float, float]:
    """
    1) Erst normal über OSMnx/Nominatim (wie bisher)
    2) Wenn das wegen Netzwerkproblemen scheitert: Fallback auf Photon (Komoot)
    """
    last = None
    for _ in range(retries + 1):
        try:
            lat, lon = ox.geocode(query)
            return float(lat), float(lon)
        except Exception as e:
            last = e
            time.sleep(1.5)

    # Fallback: Photon
    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": query, "limit": 1},
            timeout=15,
            headers={"User-Agent": "StellNah (private use)"}
        )
        r.raise_for_status()
        data = r.json()
        feats = data.get("features", [])
        if feats:
            coords = feats[0]["geometry"]["coordinates"]  # [lon, lat]
            return float(coords[1]), float(coords[0])
    except Exception as e:
        last = e

    raise last


def find_center_poi(lat0, lon0, search_m=3000):
    """
    Zentrum via Rathaus/Marktplatz versuchen.
    """
    candidates = []
    for tags in [{"amenity": "townhall"}, {"amenity": "marketplace"}]:
        try:
            gdf = safe_features_from_point((lat0, lon0), dist=search_m, tags=tags, retries=1, wait_s=1)
            if not gdf.empty:
                gdf = gdf.reset_index()
                pts = gdf.geometry.centroid
                for i, r in gdf.iterrows():
                    lat, lon = float(pts.iloc[i].y), float(pts.iloc[i].x)
                    name = str(r.get("name") or "Zentrum")
                    d = haversine_m(lat0, lon0, lat, lon)
                    candidates.append((d, lat, lon, name))
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, lat, lon, name = candidates[0]
    return lat, lon, name


def normalize_sites(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df["pt"] = df.geometry.centroid
    df["lat"] = df["pt"].y
    df["lon"] = df["pt"].x
    df["name"] = df.get("name", "").astype(str).fillna("").str.strip()
    if "tourism" not in df.columns:
        df["tourism"] = ""

    def typ_row(r):
        t = str(r.get("tourism") or "")
        if t == "caravan_site":
            return "Wohnmobilstellplatz"
        if t == "camp_site":
            return "Campingplatz"
        return "Stellplatz (Parken)"

    df["typ"] = df.apply(typ_row, axis=1)
    return df


def is_motorhome_parking_row(r: pd.Series) -> bool:
    """
    Parkplatz als Stellplatz-Kandidat nur, wenn Hinweise auf Wohnmobil/Caravan existieren.
    """
    motorhome = str(r.get("motorhome") or "").lower()
    caravan = str(r.get("caravan") or "").lower()
    parking = str(r.get("parking") or "").lower()

    if motorhome == "yes" or caravan == "yes":
        return True
    if "caravan" in parking or "motorhome" in parking:
        return True
    return False


def min_dist_to_pois(slat: float, slon: float, pois_df: pd.DataFrame) -> Optional[float]:
    if pois_df is None or pois_df.empty:
        return None
    best = None
    for _, p in pois_df.iterrows():
        d = haversine_m(slat, slon, float(p["plat"]), float(p["plon"]))
        if best is None or d < best:
            best = d
    return best


# -------------------------
# Endpunkte
# -------------------------

@app.get("/health")
def health():
    return {"ok": True, "app": APP_NAME}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    zielort = req.zielort.strip()
    radius_m = int(req.suchradius_km * 1000)

    # 1) Zielort robust geocodieren
    lat0, lon0 = geocode_robust(zielort)

    # 2) Zentrum bestimmen
    cpoi = find_center_poi(lat0, lon0, search_m=3000)
    if cpoi:
        clat, clon, cname = cpoi
    else:
        clat, clon, cname = lat0, lon0, zielort

    # 3) Stellplatz-Kandidaten sammeln
    sites_list = []

    # Klassische Tags
    try:
        sites_a = safe_features_from_point(
            (lat0, lon0),
            dist=radius_m,
            tags={"tourism": ["caravan_site", "camp_site"]},
            retries=2, wait_s=2
        )
        if sites_a is not None and not sites_a.empty:
            sites_list.append(normalize_sites(sites_a))
    except Exception:
        pass

    # Parking-Heuristik
    try:
        sites_b = safe_features_from_point(
            (lat0, lon0),
            dist=radius_m,
            tags={"amenity": "parking"},
            retries=2, wait_s=2
        )
        if sites_b is not None and not sites_b.empty:
            tmp = normalize_sites(sites_b)
            tmp = tmp[tmp.apply(is_motorhome_parking_row, axis=1)]
            if not tmp.empty:
                sites_list.append(tmp)
    except Exception:
        pass

    if not sites_list:
        return {
            "app": APP_NAME,
            "zielort": zielort,
            "suchradius_km": req.suchradius_km,
            "zentrum": {"name": cname},
            "results": []
        }

    sites = pd.concat(sites_list, ignore_index=True)

    # Duplikate entfernen (Name + Koordinate)
    sites["key"] = sites["name"].fillna("") + "|" + sites["lat"].round(6).astype(str) + "|" + sites["lon"].round(6).astype(str)
    sites = sites.drop_duplicates(subset=["key"]).drop(columns=["key"])

    # Sortieren nach Nähe zum Zentrum und begrenzen (Render-Free Schutz)
    sites["dist_center_m"] = sites.apply(lambda r: haversine_m(float(r["lat"]), float(r["lon"]), float(clat), float(clon)), axis=1)
    sites = sites.sort_values("dist_center_m").head(MAX_SITES_TO_CHECK)

    results: List[Dict[str, Any]] = []

    # 4) Pro Kandidat: kleine POI-Abfrage um den Stellplatz (stabiler)
    for _, s in sites.iterrows():
        name = s["name"] if s["name"] else "(ohne Namen)"
        slat, slon = float(s["lat"]), float(s["lon"])

        d_center = haversine_m(slat, slon, float(clat), float(clon))
        if d_center > CENTER_M:
            continue

        # POIs in kleinem Umkreis
        try:
            pois = safe_features_from_point(
                (slat, slon),
                dist=POI_SEARCH_M,
                tags={"shop": ["bakery", "butcher", "supermarket", "convenience"]},
                retries=1, wait_s=1
            )
        except Exception:
            continue

        if pois is None or pois.empty:
            continue

        pois = pois.reset_index()
        pois["pt"] = pois.geometry.centroid
        pois["plat"] = pois["pt"].y
        pois["plon"] = pois["pt"].x
        pois["shop"] = pois.get("shop", "").astype(str)

        bakery_df = pois[pois["shop"] == "bakery"]
        butcher_df = pois[pois["shop"] == "butcher"]
        super_df = pois[pois["shop"] == "supermarket"]
        conv_df = pois[pois["shop"] == "convenience"]

        d_bakery = min_dist_to_pois(slat, slon, bakery_df)
        d_butcher = min_dist_to_pois(slat, slon, butcher_df)
        d_super = min_dist_to_pois(slat, slon, super_df)
        d_conv = min_dist_to_pois(slat, slon, conv_df)

        bakery_ok = (d_bakery is not None and d_bakery <= WALK_M)
        butcher_ok = (d_butcher is not None and d_butcher <= WALK_M)
        super_ok = (d_super is not None and d_super <= WALK_M)
        conv_ok = (d_conv is not None and d_conv <= WALK_M)

        einkauf_ok = bakery_ok or butcher_ok or super_ok or conv_ok
        if not einkauf_ok:
            continue

        d_min = min([d for d in [d_bakery, d_butcher, d_super, d_conv] if d is not None], default=None)

        results.append({
            "typ": str(s.get("typ") or "Stellplatz"),
            "name": name,
            "min_einkauf_m": int(d_min) if d_min is not None else None,
            "zentrum_m": int(d_center),
            "baecker": bool(bakery_ok),
            "metzger": bool(butcher_ok),
            "supermarkt": bool(super_ok),
            "nahkauf": bool(conv_ok),
            "maps_link": maps_search_link(f"{name} {zielort}"),
        })

    results.sort(key=lambda x: x["min_einkauf_m"] if x["min_einkauf_m"] is not None else 10**9)

    return {
        "app": APP_NAME,
        "zielort": zielort,
        "suchradius_km": req.suchradius_km,
        "zentrum": {"name": cname},
        "params": {
            "einkauf_luftlinie_m": WALK_M,
            "zentrum_luftlinie_m": CENTER_M,
            "poi_search_m": POI_SEARCH_M,
            "max_sites_checked": MAX_SITES_TO_CHECK
        },
        "results": results
    }
