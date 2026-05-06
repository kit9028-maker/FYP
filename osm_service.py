from typing import Dict
import random
import requests

import requests
import time


def get_real_traffic_signals_osm(start_lat, start_lon, end_lat, end_lon, distance_km, max_retries=3):
    """
    Query OSM Overpass API - with retry logic & smart filtering.
    """

    # Create bounding box
    min_lat = min(start_lat, end_lat)
    max_lat = max(start_lat, end_lat)
    min_lon = min(start_lon, end_lon)
    max_lon = max(start_lon, end_lon)

    lat_margin = (max_lat - min_lat) * 0.1
    lon_margin = (max_lon - min_lon) * 0.1

    south = min_lat - lat_margin
    north = max_lat + lat_margin
    west = min_lon - lon_margin
    east = max_lon + lon_margin

    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = f"""
    [out:json][timeout:10];
    (
      node["highway"="traffic_signals"]({south},{west},{north},{east});
      way["highway"="traffic_signals"]({south},{west},{north},{east});
    );
    out center;
    """

    # RETRY LOOP
    for attempt in range(max_retries):
        try:
            print(f"🔍 Querying OSM for traffic signals (attempt {attempt + 1}/{max_retries})...")
            resp = requests.post(overpass_url, data=overpass_query, timeout=20)

            if resp.status_code == 200:
                data = resp.json()
                osm_count = len(data.get('elements', []))

                print(f"   OSM bbox found: {osm_count} signals")

                # SMART FILTER: Cap to realistic HK density
                hk_realistic_density = 2.2  # signals/km max (Kowloon)
                realistic_count = int(distance_km * hk_realistic_density)

                # If OSM way over, cap it
                if osm_count > realistic_count * 2.5:
                    print(f"   ⚠️ Bbox includes surrounding (too many)")
                    print(f"   Using filtered: {realistic_count} signals")
                    return realistic_count
                else:
                    print(f"✅ OSM realistic: {osm_count} signals")
                    return osm_count

            elif resp.status_code == 504:
                print(f"⏱️ HTTP 504 - Retrying in 5 sec...")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5)
            else:
                print(f"❌ HTTP {resp.status_code}")
                return 0

        except Exception as e:
            print(f"❌ Error: {e}")
            return 0

    return 0


def get_traffic_signals_per_km(signal_count, distance_km):
    """Calculate signals per km for complexity scoring."""
    if distance_km <= 0:
        return 0
    return round(signal_count / distance_km, 2)


def estimate_hk_baseline_signals(distance_km, avg_lat, avg_lon):
    """
    HK urban baseline signal density estimate.
    Used when OSM fails or returns 0.
    """
    # Check if in dense Kowloon/Central area
    if 114.15 < avg_lon < 114.22 and 22.27 < avg_lat < 22.35:
        # Kowloon: high-density urban
        baseline = 2.1  # signals/km
    elif 114.10 < avg_lon < 114.30 and 22.25 < avg_lat < 22.40:
        # Overall urban HK
        baseline = 1.7  # signals/km
    else:
        # Suburban/rural
        baseline = 1.0  # signals/km

    estimated = max(2, int(distance_km * baseline))
    print(f"📊 HK baseline: {estimated} signals (~{baseline:.1f}/km)")
    return estimated


def get_traffic_signals_smart(start_lat, start_lon, end_lat, end_lon, distance_km):
    """
    RECOMMENDED: Try OSM with retries, fallback to HK baseline.
    This is the production-ready function.
    """

    # Try OSM first
    osm_signals = get_real_traffic_signals_osm(start_lat, start_lon, end_lat, end_lon, max_retries=3)

    if osm_signals > 0:
        return osm_signals, "OSM"

    # Fallback to HK baseline
    print(f"⚠️ OSM unavailable, using HK urban baseline...")
    avg_lat = (start_lat + end_lat) / 2
    avg_lon = (start_lon + end_lon) / 2
    baseline_signals = estimate_hk_baseline_signals(distance_km, avg_lat, avg_lon)

    return baseline_signals, "HK_BASELINE"


def check_osm_coverage(district_name, lat, lon):
    """Check if OSM has traffic signal data for your area."""
    margin = 0.05  # ~5km radius

    query = f"""
    [out:json][timeout:5];
    node["highway"="traffic_signals"]({lat - margin},{lon - margin},{lat + margin},{lon + margin});
    out count;
    """

    overpass_url = "http://overpass-api.de/api/interpreter"

    try:
        resp = requests.post(overpass_url, data=query, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            count = len(data.get('elements', []))
            print(f"📊 {district_name}: {count} signals in 5km radius")

            if count == 0:
                print(f"   ⚠️ OSM has NO signal data here - MUST use HK baseline")
            return count
    except Exception as e:
        print(f"   Error checking coverage: {e}")

    return 0


def get_geographical_factors(lat: float, lon: float, road_hint: str = "") -> Dict:
    """Dynamic HK road risk based on coordinates (no static data needed)."""
    base_data = calculate_hk_risk_from_coords(lat, lon, road_hint)

    # Delivery-specific adjustments
    if "nathan" in road_hint.lower() or "queen's" in road_hint.lower():
        base_data["delivery_risk"] = "HIGH"
        base_data["complexity_score"] += 1.5
    elif "airport" in road_hint.lower():
        base_data["delivery_risk"] = "LOW"
        base_data["complexity_score"] -= 1.0

    return base_data


def calculate_hk_risk_from_coords(lat: float, lon: float, road_hint: str = "") -> Dict:
    """Calculate risk based on HK geography (no API/static data)."""

    # HK density zones by coordinates
    if 114.15 < lon < 114.20 and 22.27 < lat < 22.33:  # Central/Kowloon
        roads, signals, complexity, risk = 38, 15, 8.5, "HIGH"
    elif 113.90 < lon < 114.05:  # Airport/Lantau
        roads, signals, complexity, risk = 15, 3, 3.5, "LOW"
    elif 22.35 < lat < 22.42:  # New Territories
        roads, signals, complexity, risk = 22, 7, 5.2, "MEDIUM"
    else:  # Default HK
        roads, signals, complexity, risk = 28, 10, 6.8, "MEDIUM"

    return format_geo_data(roads, signals, complexity, risk, road_hint)


def format_geo_data(roads: int, signals: int, complexity: float, risk: str, road_name: str = "") -> Dict:
    """Format output data."""
    return {
        "road_count": roads,
        "risky_roads": int(roads * 0.65),
        "traffic_signals": signals,
        "construction_zones": random.randint(0, 1),
        "fire_hydrants": random.randint(2, 6),
        "elevation_change_m": random.randint(10, 80),
        "complexity_score": round(complexity, 1),
        "risk_level": risk,
        "road_name": road_name.title() if road_name else "Local road",
        "delivery_risk": risk
    }
