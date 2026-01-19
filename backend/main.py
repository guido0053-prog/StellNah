import math
import time
from typing import List, Dict, Any, Optional

import pandas as pd
import osmnx as ox
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP_NAME = "StellNah"

# Kriterien (Luftlinie)
WALK_M = 300          # Einkauf max 300 m
CENTER_M = 1500       # zentrumsnah max 1,5 km

# Wie weit wir pro Stellplatz nach POIs suchen (kleine, schnelle Abfragen)
POI_SEARCH_M = 1200   # 1,2 km Umkreis pro Stellplatz

# Sicherheitslimit: wenn extrem viele Stellplatz-Kandidaten gefunden werden
MAX_SITES_TO_CHECK = 60

# OSMnx stabilisieren
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_timeout = 180
ox.settings.http_headers = {"User-Agent": "StellNah (private use)"}

app = FastAPI(title=f"{APP_NAME} API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    zielort: str
    suchradius_km: float


def haversine_m(lat1, lon1, lat2, lon2):
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
    Kleine Stabilitäts-Hilfe: wenn Overpass mal zickt, versuchen wir es nochmal.
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


def find_center_poi(lat0, lon0, search_m=3000):
    candidates = []
    for tags in [{"amenity": "townhall"}, {"amenity": "marketplace"}]:
        try:
            gdf = safe_features_from_point((lat0, lon0), dist=search_m, tags=tags, retries=1)
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
        # Parking-basierte Heuristik:
        return "Stellplatz (Parken)"

    df["typ"] = df.apply(typ_row, axis=1)
    return df


def is_motorhome_parking_row(r: pd.Series) -> bool:
    """
    Heuristik: Stellplätze, die als Parkplatz getaggt sind.
    Wir nehmen sie nur, wenn Hinweise auf Wohnmobil/Caravan existieren.
    """
    # Oft sind die Spalten als Strings vorhanden, z.B. motorhome='yes', caravan='yes'
    motorhome = str(r.get("motorhome") or "").lower()
    caravan = str(r.get("caravan") or "").lower()
    parking = str(r.get("parking") or "").lower()

    if motorhome == "yes" or caravan == "yes":
        return True
    if "caravan" in parking or "motorhome" in parking:
        return True
    return False


@app.get("/health")
def health():
    return {"ok": True, "app": APP_NAME}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    zielort = req.zielort.strip()
    radius_m = int(req.suchradius_km * 1000)

    # 1) Zielort geocodieren
    lat0, lon0 = ox.geocode(zielort)

    # 2) Zentrum bestimmen
    cpoi = find_center_poi(lat0, lon0)
    if cpoi:
        clat, clon, cname = cpoi
    else:
        clat, clon, cname = lat0, lon0, zielort

    # 3) Stellplätze/Campingplätze: klassische Tags
    sites_a = safe_features_from_point(
        (lat0, lon0),
        dist=radius_m,
        tags={"tourism": ["caravan_site", "camp_site"]},
        retries=2
    )

    # 4) Zusätzlich: Parkplatz-Tags (damit wir Biberach & Co. besser treffen)
    sites_b = safe_features_from_point(
        (lat0, lon0),
        dist=radius_m,
        tags={"amenity": "parking"},
        retries=2
    )

    sites_list = []

    if sites_a is not None and not sites_a.empty:
        sites_list.append(normalize_sites(sites_a))

    if sites_b is not None and not sites_b.empty:
        tmp = normalize_sites(sites_b)
        # nur die Parkplätze, die nach Wohnmobil aussehen
        tmp = tmp[tmp.apply(is_motorhome_parking_row, axis=1)]
        if not tmp.empty:
            sites_list.append(tmp)

    if not sites_list:
        return {
            "app": APP_NAME,
            "zielort": zielort,
            "suchradius_km": req.suchradius_km,
            "zentrum": {"name": cname},
            "results": []
        }

    sites = pd.concat(sites_list, ignore_index=True)

    # Duplikate entfernen (gleiche Koordinate + gleicher Name)
    sites["key"] = sites["name"].fillna("") + "|" + sites["lat"].round(6).astype(str) + "|" + sites["lon"].round(6).astype(str)
    sites = sites.drop_duplicates(subset=["key"]).drop(columns=["key"])

    # Sicherheitslimit, damit Render Free nicht stirbt
    # Wir sortieren nach Nähe zum Zentrum (praktisch sinnvoll)
    sites["dist_center_m"] = sites.apply(lambda r: haversine_m(float(r["lat"]), float(r["lon"]), float(clat), float(clon)), axis=1)
    sites = sites.sort_values("dist_center_m").head(MAX_SITES_TO_CHECK)

    results: List[Dict[str, Any]] = []

    # 5) Pro Stellplatz kleine POI-Abfrage (stabiler als riesige POI-Abfrage)
    for _, s in sites.iterrows():
        name = s["name"] if s["name"] else "(ohne Namen)"
        slat, slon = float(s["lat"]), float(s["lon"])

        # zentrumsnah (Luftlinie)
        d_center = haversine_m(slat, slon, float(clat), float(clon))
        if d_center > CENTER_M:
            continue

        # kleine POI-Abfrage nur um den Stellplatz
        try:
            pois = safe_features_from_point(
                (slat, slon),
                dist=POI_SEARCH_M,
                tags={"shop": ["bakery", "butcher", "supermarket", "convenience"]},
                retries=1
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

        # min-Distanzen je Kategorie
        def min_dist(shop_value: str) -> Optional[float]:
            sub = pois[pois["shop"] == shop_value]
            if sub.empty:
                return None
            best = None
            for _, p in sub.iterrows():
                d = haversine_m(slat, slon, float(p["plat"]), float(p["plon"]))
                if best is None or d < best:
                    best = d
            return best

        d_bakery = min_dist("bakery")
        d_butcher = min_dist("butcher")
        d_super = min_dist("supermarket")
        d_conv = min_dist("convenience")

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
