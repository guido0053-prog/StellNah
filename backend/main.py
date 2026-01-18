import math
from typing import List, Dict, Any

import osmnx as ox
import networkx as nx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# StellNah – POI Agent API (OpenStreetMap-basiert)
ox.settings.use_cache = True
ox.settings.log_console = False

APP_NAME = "StellNah"
WALK_M = 350          # ca. 0,5 km (fußläufig)
CENTER_WALK_M = 800   # zentrumsnah (~10 Min zu Fuß)

app = FastAPI(title=f"{APP_NAME} API")

# Damit die Webseite (Frontend) das Backend aufrufen darf (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # für private Solo-App ok; später ggf. enger machen
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

def find_center_poi(lat0, lon0, search_m=3000):
    """
    Versucht, ein plausibles Stadtzentrum zu finden:
    1) Rathaus (amenity=townhall)
    2) Marktplatz (amenity=marketplace)
    Falls nichts gefunden wird: Rückgabe None -> Geocode-Punkt als Zentrum.
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
                    name = str(r.get("name") or "").strip() or "Zentrum"
                    d = haversine_m(lat0, lon0, lat, lon)
                    candidates.append((d, lat, lon, name))
        except Exception:
            pass

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, lat, lon, name = candidates[0]
    return lat, lon, name

@app.get("/health")
def health():
    return {"ok": True, "app": APP_NAME}

@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    zielort = req.zielort.strip()
    radius_m = int(req.suchradius_km * 1000)

    # 1) Zielort in Koordinaten umwandeln
    lat0, lon0 = ox.geocode(zielort)

    # 2) Fußwegenetz im Radius laden
    G = ox.graph_from_point((lat0, lon0), dist=radius_m, network_type="walk", simplify=True)
    G = ox.utils_graph.get_largest_component(G, strongly=False)

    # 3) Zentrum bestimmen (Rathaus/Marktplatz) – fallback Geocode-Punkt
    cpoi = find_center_poi(lat0, lon0, search_m=3000)
    if cpoi:
        clat, clon, cname = cpoi
    else:
        clat, clon, cname = float(lat0), float(lon0), zielort

    # 4) Fußweg-Distanzen vom Zentrum zu allen Knoten berechnen
    center_node = ox.distance.nearest_nodes(G, clon, clat)
    dist_from_center = nx.single_source_dijkstra_path_length(G, source=center_node, weight="length")

    # 5) Stellplätze & Campingplätze (OSM)
    sites = ox.features_from_point((lat0, lon0), dist=radius_m, tags={"tourism": ["caravan_site", "camp_site"]})
    if sites.empty:
        return {"zielort": zielort, "suchradius_km": req.suchradius_km, "zentrum": {"name": cname}, "results": []}

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

    # 6) POIs: Bäcker/Metzger/Supermarkt
    pois = ox.features_from_point((lat0, lon0), dist=radius_m, tags={"shop": ["bakery", "butcher", "supermarket"]})
    pois = pois.reset_index()
    if pois.empty:
        return {"zielort": zielort, "suchradius_km": req.suchradius_km, "zentrum": {"name": cname}, "results": []}

    pois["pt"] = pois.geometry.centroid
    pois["lat"] = pois["pt"].y
    pois["lon"] = pois["pt"].x
    pois["poi_type"] = pois.get("shop", "").astype(str)

    # 7) POI-Typen auf Netzwerkknoten mappen
    poi_nodes = {}
    for poi_type in ["bakery", "butcher", "supermarket"]:
        sub = pois[pois["poi_type"] == poi_type]
        if sub.empty:
            poi_nodes[poi_type] = set()
            continue
        nodes = ox.distance.nearest_nodes(G, sub["lon"].values, sub["lat"].values)
        poi_nodes[poi_type] = set(int(n) for n in nodes) if not isinstance(nodes, int) else {int(nodes)}

    # 8) Distanz zu jedem POI-Typ als Multi-Source-Dijkstra (effizient)
    dist_by_type = {}
    for poi_type, sources in poi_nodes.items():
        dist_by_type[poi_type] = nx.multi_source_dijkstra_path_length(G, sources=sources, weight="length") if sources else {}

    # 9) Stellplätze auswerten und filtern
    results: List[Dict[str, Any]] = []
    for _, r in sites.iterrows():
        name = r["name"] if r["name"] else "(ohne Namen)"
        lat, lon = float(r["lat"]), float(r["lon"])
        node = ox.distance.nearest_nodes(G, lon, lat)

        d_center = dist_from_center.get(node)
        zentrumsnah = (d_center is not None and d_center <= CENTER_WALK_M)

        d_bakery = dist_by_type["bakery"].get(node)
        d_butcher = dist_by_type["butcher"].get(node)
        d_super  = dist_by_type["supermarket"].get(node)

        bakery_ok = (d_bakery is not None and d_bakery <= WALK_M)
        butcher_ok = (d_butcher is not None and d_butcher <= WALK_M)
        super_ok  = (d_super is not None and d_super <= WALK_M)

        einkauf_ok = bakery_ok or butcher_ok or super_ok

        # Deine Kriterien:
        if not (zentrumsnah and einkauf_ok):
            continue

        min_dist = min([d for d in [d_bakery, d_butcher, d_super] if d is not None], default=None)
        q = f"{name} {zielort}"

        results.append({
            "typ": r["typ"],
            "name": name,
            "min_fussweg_m": int(min_dist) if min_dist is not None else None,
            "zentrum_m": int(d_center) if d_center is not None else None,
            "baecker": bool(bakery_ok),
            "metzger": bool(butcher_ok),
            "supermarkt": bool(super_ok),
            "maps_link": maps_search_link(q),
        })

    # Sortierung: kleinste Einkaufsdistanz zuerst
    results.sort(key=lambda x: x["min_fussweg_m"] if x["min_fussweg_m"] is not None else 10**9)

    return {
        "app": APP_NAME,
        "zielort": zielort,
        "suchradius_km": req.suchradius_km,
        "zentrum": {"name": cname},
        "params": {"walk_m": WALK_M, "center_walk_m": CENTER_WALK_M},
        "results": results,
    }
