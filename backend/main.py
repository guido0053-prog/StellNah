import math
from typing import List, Dict, Any

import osmnx as ox
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# StellNah – Backend (stabil, Luftlinie)
# =========================

APP_NAME = "StellNah"

# Kriterien (Luftlinie)
WALK_M = 300          # Einkauf max. 300 m
CENTER_M = 1500       # zentrumsnah max. 1,5 km

# OSMnx stabilisieren (wichtig für Render)
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_timeout = 180
ox.settings.http_headers = {"User-Agent": "StellNah (private use)"}

app = FastAPI(title=f"{APP_NAME} API")

# CORS erlauben (Frontend darf Backend aufrufen)
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


# -------------------------
# Hilfsfunktionen
# -------------------------

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


def find_center_poi(lat0, lon0, search_m=3000):
    """
    Versucht ein Stadtzentrum zu finden:
    - Rathaus
    - Marktplatz
    Fallback: Geocode-Punkt
    """
    candidates = []
    for tags in [{"amenity": "townhall"}, {"amenity": "marketplace"}]:
        try:
            gdf = ox.features_from_point((lat0, lon0), dist=search_m, tags=tags)
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


# -------------------------
# Healthcheck
# -------------------------

@app.get("/health")
def health():
    return {"ok": True, "app": APP_NAME}


# -------------------------
# Analyse (Hauptfunktion)
# -------------------------

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

    # 3) Stellplätze & Campingplätze
    sites = ox.features_from_point(
        (lat0, lon0),
        dist=radius_m,
        tags={"tourism": ["caravan_site", "camp_site"]}
    )

    if sites.empty:
        return {
            "app": APP_NAME,
            "zielort": zielort,
            "suchradius_km": req.suchradius_km,
            "zentrum": {"name": cname},
            "results": []
        }

    sites = sites.reset_index()
    sites["pt"] = sites.geometry.centroid
    sites["lat"] = sites["pt"].y
    sites["lon"] = sites["pt"].x
    sites["name"] = sites.get("name", "").astype(str).fillna("").str.strip()

    def typ(v):
        if v == "caravan_site":
            return "Wohnmobilstellplatz"
        if v == "camp_site":
            return "Campingplatz"
        return "Unbekannt"

    sites["typ"] = sites["tourism"].apply(typ)

    # 4) Einkauf-POIs
    pois = ox.features_from_point(
        (lat0, lon0),
        dist=radius_m,
        tags={"shop": ["bakery", "butcher", "supermarket"]}
    )

    if pois.empty:
        return {
            "app": APP_NAME,
            "zielort": zielort,
            "suchradius_km": req.suchradius_km,
            "zentrum": {"name": cname},
            "results": []
        }

    pois = pois.reset_index()
    pois["pt"] = pois.geometry.centroid
    pois["lat"] = pois["pt"].y
    pois["lon"] = pois["pt"].x
    pois["shop"] = pois.get("shop", "").astype(str)

    bakery = pois[pois["shop"] == "bakery"][["lat", "lon"]].to_dict("records")
    butcher = pois[pois["shop"] == "butcher"][["lat", "lon"]].to_dict("records")
    market  = pois[pois["shop"] == "supermarket"][["lat", "lon"]].to_dict("records")

    def min_dist(slat, slon, lst):
        best = None
        for p in lst:
            d = haversine_m(slat, slon, float(p["lat"]), float(p["lon"]))
            if best is None or d < best:
                best = d
        return best

    results: List[Dict[str, Any]] = []

    for _, r in sites.iterrows():
        name = r["name"] if r["name"] else "(ohne Namen)"
        slat, slon = float(r["lat"]), float(r["lon"])

        d_center = haversine_m(slat, slon, clat, clon)
        if d_center > CENTER_M:
            continue

        d_bakery = min_dist(slat, slon, bakery)
        d_butcher = min_dist(slat, slon, butcher)
        d_market = min_dist(slat, slon, market)

        ok = False
        d_min = None

        for d in [d_bakery, d_butcher, d_market]:
            if d is not None and d <= WALK_M:
                ok = True
                d_min = d if d_min is None else min(d_min, d)

        if not ok:
            continue

        q = f"{name} {zielort}"
        results.append({
            "typ": r["typ"],
            "name": name,
            "min_einkauf_m": int(d_min),
            "zentrum_m": int(d_center),
            "baecker": d_bakery is not None and d_bakery <= WALK_M,
            "metzger": d_butcher is not None and d_butcher <= WALK_M,
            "supermarkt": d_market is not None and d_market <= WALK_M,
            "maps_link": maps_search_link(q),
        })

    results.sort(key=lambda x: x["min_einkauf_m"])

    return {
        "app": APP_NAME,
        "zielort": zielort,
        "suchradius_km": req.suchradius_km,
        "zentrum": {"name": cname},
        "params": {
            "einkauf_luftlinie_m": WALK_M,
            "zentrum_luftlinie_m": CENTER_M
        },
        "results": results,
    }
