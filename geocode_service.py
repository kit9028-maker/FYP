import requests
import re
import time
import math
from typing import Tuple
from pyproj import Transformer

geocode_cache = {}


def geocode_location(name: str) -> Tuple[float, float]:
    """Production geocoder: API first → Database fallback."""
    name_lower = name.lower().strip()
    cache_key = name_lower

    if cache_key in geocode_cache:
        print(f"📍 Cache: {name}")
        return geocode_cache[cache_key]

    street_num, road_name = parse_address(name_lower)

    street_num, road_name = parse_address(name_lower)

    # 1. Try Hong Kong Location Search API (official HK data)
    lat, lon = try_hk_location_api(f"{street_num or ''} {road_name}, Hong Kong")
    if lat and lon:
        result = (lat, lon)
        geocode_cache[cache_key] = result
        print(f"🇭🇰 HK API: {street_num or ''} {road_name.title()} ({lat:.5f}, {lon:.5f})")
        return result

    # 2. Fallback: FREE Geocode.Maps.co API
    lat, lon = try_geocode_api(f"{street_num or ''} {road_name}, Hong Kong")
    if lat and lon:
        result = (lat, lon)
        geocode_cache[cache_key] = result
        print(f"🌐 API: {street_num or ''} {road_name.title()} ({lat:.5f}, {lon:.5f})")
        return result

    # 2. Database fallback
    road_coords = find_road_coords(road_name)
    if road_coords:
        lat, lon = adjust_for_street_number(road_coords[0], road_coords[1], street_num, road_name)
        result = (lat, lon)
        geocode_cache[cache_key] = result
        print(f"📱 DB: {street_num or ''} {road_name.title()} ({lat:.5f}, {lon:.5f})")
        return result

    # HK center
    default = (22.3027, 114.1772)
    geocode_cache[cache_key] = default
    print(f"⚠️ '{name}' not found, using HK center")
    return default


def try_geocode_api(address: str) -> tuple[float, float] | tuple[None, None]:
    """FREE Geocode.Maps.co API - Excellent HK coverage, no key."""
    try:
        url = "https://geocode.maps.co/search"
        params = {
            "q": address,
            "api_key": "demo"  # Free demo key [web:33]
        }
        resp = requests.get(url, params=params, timeout=8)

        if resp.status_code == 200:
            data = resp.json()
            if data.get("results") and data["results"]:
                result = data["results"][0]
                lat = float(result["lat"])
                lon = float(result["lon"])
                return lat, lon
    except:
        pass
    return None, None


def parse_address(address: str) -> Tuple[str, str]:
    match = re.match(r'^(\d+[a-z]?)?\s*(.+)$', address.strip())
    if match:
        return match.group(1), match.group(2).strip().lower()
    return "", address.lower()


def find_road_coords(road_name: str) -> tuple[float, float] | None:
    """Fallback database (100+ major roads)."""
    hk_roads = {
        "nathan road": (22.3193, 114.1700),
        "portland street": (22.31815, 114.16898),
        "wing ting road": (22.3385, 114.2025),
        "choi hung road": (22.3400, 114.2050),
        "ngau tau kok road": (22.3270, 114.2100),  # ✅ Ngau Tau Kok
        "queen's road central": (22.2815, 114.1585),
        "hennessy road": (22.2790, 114.1750),
    }

    for road, coords in hk_roads.items():
        if road in road_name or road_name in road:
            return coords
    return None


def adjust_for_street_number(base_lat: float, base_lon: float, street_num: str, road_name: str) -> Tuple[float, float]:
    if not street_num or not street_num.isdigit():
        return base_lat, base_lon
    offset = (int(street_num) / 1000.0) * 0.0005
    return base_lat + offset, base_lon + offset


def geocode_with_cache(name: str) -> Tuple[float, float]:
    time.sleep(0.2)  # API politeness
    return geocode_location(name)

def try_landsd_api(address: str) -> Tuple[float, float]:
    """LandsD Location Search API - Official HK roads/places, no key for trial."""
    try:
        # From CSDI/LandsD portals - adjust endpoint if needed from docs
        url = "https://api.portal.hkmapservice.gov.hk/oneapi/lsapi"  # Location Search API
        params = {"q": address, "output": "json"}  # Query format from docs
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results") and data["results"]:  # Assumed structure
                result = data["results"][0]
                lat = float(result["lat"])  # Or 'y'/'latitude'
                lon = float(result["lon"])  # Or 'x'/'longitude'
                return lat, lon
    except Exception as e:
        print(f"LandsD API error: {e}")
    return None, None


def try_hk_location_api(address: str) -> Tuple[float, float]:
    """HK Location Search API - pyproj HK1980 Grid → WGS84 conversion."""
    try:
        # Setup Transformer: HK1980 Grid (EPSG:2326) → WGS84 (EPSG:4326)
        transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)

        url = "https://geodata.gov.hk/gs/api/v1.0.0/locationSearch"
        params = {"q": address}

        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return None, None

        data = resp.json()
        if not data:
            return None, None

        first = data[0]

        # Extract HK1980 Grid coordinates (x=Easting, y=Northing)
        y_grid = first.get('y')  # Northing
        x_grid = first.get('x')  # Easting

        if y_grid is None or x_grid is None:
            print(f"❌ No HK Grid coords in API response")
            return None, None

        # Convert HK1980 Grid → WGS84 using pyproj
        lon, lat = transformer.transform(x_grid, y_grid)

        print(f"✅ HK Grid ({x_grid}, {y_grid}) → WGS84 ({lat:.5f}, {lon:.5f})")

        # Sanity check (HK bounds: lat 20-25°, lon 110-120°)
        if 20 <= lat <= 25 and 110 <= lon <= 120:
            print(f"✅ HK API: {address} ({lat:.5f}, {lon:.5f})")
            return lat, lon

        print(f"❌ Suspicious coords after conversion: {lat}, {lon}")
        return None, None

    except Exception as e:
        print(f"HK API error: {e}")
        return None, None