import json
import os
from typing import Optional, Tuple

import requests
from shapely.geometry import Point, shape

# Official HAD district boundary dataset (data.gov.hk)
# If this URL ever changes, you can replace it with a local file path.
HAD_DISTRICT_JSON_URL = (
    "https://data.gov.hk/en-data/dataset/hk-had-json1-hong-kong-administrative-boundaries"
)

# We store a local cached copy so you don't hit the network every run
CACHE_FILE = "hk_district_boundaries.geojson"


def _download_geojson(target_path: str) -> None:
    """
    Download district boundary data.
    data.gov.hk sometimes serves via redirect; we fetch the dataset landing page is not GeoJSON.
    So: you should manually download the JSON resource once and save as CACHE_FILE if this fails.
    """
    raise RuntimeError(
        "Auto-download is disabled because data.gov.hk may redirect to a resource file.\n"
        "Please download the district boundary JSON resource from data.gov.hk and save it as:\n"
        f"  {target_path}\n"
        "Then rerun."
    )


def load_district_boundaries(path: str = CACHE_FILE):
    """
    Load a GeoJSON/JSON file containing district polygons.
    Expected: FeatureCollection with polygon/multipolygon geometries and district names in properties.
    """
    if not os.path.exists(path):
        _download_geojson(path)

    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    features = gj.get("features") or gj.get("Features") or []
    if not features:
        raise ValueError(
            f"No features found in {path}. "
            "Make sure it is a GeoJSON FeatureCollection of district boundaries."
        )

    # Build list of (district_name, shapely_geometry)
    out = []
    for feat in features:
        props = feat.get("properties", {})
        # Try common property keys
        name = (
            props.get("District")
            or props.get("DISTRICT")
            or props.get("ENAME")
            or props.get("Name")
            or props.get("name")
        )
        geom = feat.get("geometry")
        if not name or not geom:
            continue
        out.append((name, shape(geom)))

    if not out:
        raise ValueError(
            f"Loaded features but could not parse district names/geometries from {path}."
        )

    return out


class DistrictLookup:
    def __init__(self, geojson_path: str = CACHE_FILE):
        self._district_geoms = load_district_boundaries(geojson_path)

    def district_of(self, lat: float, lon: float) -> str:
        p = Point(lon, lat)  # shapely uses (x,y) = (lon,lat)
        for name, geom in self._district_geoms:
            # contains() may fail on boundary; covers() includes boundary points
            if geom.covers(p):
                return str(name)
        return "Unknown"


_lookup_singleton: Optional[DistrictLookup] = None


def get_district(lat: float, lon: float, geojson_path: str = CACHE_FILE) -> str:
    global _lookup_singleton
    if _lookup_singleton is None:
        _lookup_singleton = DistrictLookup(geojson_path)
    return _lookup_singleton.district_of(lat, lon)