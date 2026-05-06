# Main_program.py (XGBoost mimic version - FIXED)
# -----------------------------------------------------------
# - Trains XGBoost to mimic your rule-based "teacher" outputs:
#   risk_score (0-10) and est_time (minutes)
# - risk_level derived from predicted risk_score with same thresholds2
# - Avoids Overpass/OSM rate-limit loops by:
#   * Disabling OSM during training (DISABLE_OSM = True)
#   * Using cached + backoff Overpass calls at runtime
#   * Falling back to estimated signals if OSM unavailable
#
# Requirements:
#   pip install xgboost scikit-learn joblib numpy requests pytz
#
# Your local modules required:
#   geocode_service.py (geocode_with_cache)
#   weather_service.py (get_current_weather, describe_weather)
#   osm_service.py (get_real_traffic_signals_osm)  # optional but recommended
# ------------------------------------------------------------

import json
import os
import re
import sys
import time
import textwrap
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

import joblib
import numpy as np
import pytz
import requests
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

try:
    import googlemaps
except Exception:
    googlemaps = None

from geocode_service import geocode_with_cache
from weather_service import describe_weather, get_current_weather
try:
    from district_lookup import get_district as get_polygon_district
except Exception:
    get_polygon_district = None

# Optional fallback for local testing. Leave blank if you prefer env vars or Streamlit secrets only.
LOCAL_GOOGLE_MAPS_API_KEY = "AIzaSyBCbcyOiTXyGoKfdMOxe_lCF2bdxzB8z5E"


def get_google_maps_api_key() -> str:
    """Read Google Maps API key from Streamlit secrets, env vars, then local fallback."""
    try:
        import streamlit as st  # Optional at runtime

        secret_key = str(st.secrets.get("GOOGLE_MAPS_API_KEY", "")).strip()
        if secret_key:
            return secret_key
    except Exception:
        pass

    env_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if env_key:
        return env_key

    return LOCAL_GOOGLE_MAPS_API_KEY.strip()


def mask_api_key(key: str) -> str:
    if not key:
        return "(missing)"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


if getattr(sys, "frozen", False):
    BUNDLE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    RUNTIME_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    RUNTIME_DIR = BUNDLE_DIR


def bundled_file_path(filename: str) -> str:
    return os.path.join(BUNDLE_DIR, filename)


def runtime_file_path(filename: str) -> str:
    return os.path.join(RUNTIME_DIR, filename)


def prefer_runtime_file(filename: str) -> str:
    runtime_path = runtime_file_path(filename)
    if os.path.exists(runtime_path):
        return runtime_path
    return bundled_file_path(filename)


GOOGLE_MAPS_API_KEY = get_google_maps_api_key()
GOOGLE_GEOCODE_CACHE = runtime_file_path("google_geocode_cache.json")

# If you don't have osm_service.get_real_traffic_signals_osm working,
# keep this import but you may get runtime fallback automatically.
try:
    from osm_service import get_real_traffic_signals_osm
except Exception:
    get_real_traffic_signals_osm = None

# ---------------- Global switches ---------------- #

DISABLE_OSM = False  # True only during model training
USE_GOOGLE_SIGNAL_PROXY = True
USE_OSM_SIGNAL_FALLBACK = False
SHOW_DEBUG = False
PANEL_COL_WIDTH = 48
PANEL_GAP_WIDTH = 2
SECTION_WIDTH = (PANEL_COL_WIDTH * 2) + PANEL_GAP_WIDTH
SECTION_DIVIDER = "-" * SECTION_WIDTH
SECTION_CONTENT_INDENT = 10
XGB_WEIGHT = 0.65
BASELINE_WEIGHT = 0.35
MODEL_VERSION = "v2.1"
FEATURE_SET_VERSION = "risk_features_v2.1"
ANALYSIS_LOG_FILE = "route_analysis_log.jsonl"
ANALYSIS_LOG_FILE = runtime_file_path("route_analysis_log.jsonl")
RISK_LEVEL_THRESHOLDS_TEXT = "LOW <= 4.0, MEDIUM 4.0-7.0, HIGH > 7.0"
DEFAULT_TRAINING_SAMPLES = 3000

# Cache file for Overpass bbox results
OSM_CACHE_FILE = runtime_file_path("osm_signal_cache.json")

# Model files
MODEL_SCORE_PATH = prefer_runtime_file("xgb_risk_score.json")
MODEL_TIME_PATH = prefer_runtime_file("xgb_est_time.json")
META_PATH = prefer_runtime_file("xgb_feature_meta.pkl")


# ---------------- Cache helpers ---------------- #

def _load_osm_cache() -> dict:
    try:
        with open(OSM_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_osm_cache(cache: dict) -> None:
    try:
        with open(OSM_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


# ---------------- Time factor ---------------- #

def get_time_factor(hour: int) -> float:
    """Hong Kong driving-time factor with narrower rush-hour windows."""
    if 7 <= hour <= 9:
        return 1.35
    if 17 <= hour <= 19:
        return 1.35
    if 22 <= hour or hour <= 6:
        return 0.85
    return 1.0


def get_time_band(hour: int) -> str:
    if 7 <= hour <= 9:
        return "AM_PEAK"
    if 17 <= hour <= 19:
        return "PM_PEAK"
    if 22 <= hour or hour <= 6:
        return "NIGHT"
    return "OFF_PEAK"


def normalize_travel_mode(user_input: str) -> str:
    value = (user_input or "").strip().lower()
    if value in {"1", "walk", "walking"}:
        return "walking"
    if value in {"2", "drive", "driving", "car"}:
        return "driving"
    if value in {"3", "compare", "both", "compare both"}:
        return "compare"
    return ""


HK_ROADWORKS_JSON_URL = "https://resource.data.one.gov.hk/td/roadworks-location/get_all_the_roadworks.json"


def _extract_numeric(value) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _extract_roadwork_points(payload) -> list[dict]:
    if isinstance(payload, dict):
        if isinstance(payload.get("features"), list):
            iterable = payload["features"]
        elif isinstance(payload.get("results"), list):
            iterable = payload["results"]
        elif isinstance(payload.get("data"), list):
            iterable = payload["data"]
        else:
            iterable = []
    elif isinstance(payload, list):
        iterable = payload
    else:
        iterable = []

    out = []
    for item in iterable:
        lat = lon = None
        props = item if isinstance(item, dict) else {}

        if isinstance(item, dict) and isinstance(item.get("geometry"), dict):
            props = item.get("properties", {}) or {}
            coords = item["geometry"].get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lon = _extract_numeric(coords[0])
                lat = _extract_numeric(coords[1])

        if lat is None or lon is None:
            lat = _extract_numeric(props.get("latitude") or props.get("lat") or props.get("y") or props.get("Y"))
            lon = _extract_numeric(props.get("longitude") or props.get("lon") or props.get("lng") or props.get("x") or props.get("X"))

        if lat is None or lon is None:
            continue

        out.append({
            "lat": float(lat),
            "lon": float(lon),
            "status": str(props.get("status") or props.get("works_status") or props.get("phase") or "Unknown"),
            "location": str(
                props.get("location")
                or props.get("road_name")
                or props.get("location_desc_en")
                or props.get("description")
                or props.get("works_location")
                or "Unknown location"
            ),
            "affected_lane": str(
                props.get("affected_lane")
                or props.get("lane")
                or props.get("traffic_impact")
                or props.get("lane_closure")
                or "Unknown"
            ),
            "start_time": str(props.get("start_time") or props.get("startdate") or props.get("start_date") or ""),
            "end_time": str(props.get("end_time") or props.get("enddate") or props.get("end_date") or ""),
        })

    return out


def _parse_roadwork_datetime(value: str):
    value = (value or "").strip()
    if not value:
        return None

    normalized = value.replace("T", " ").replace("Z", "+00:00")
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]

    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            return hkt_tz.localize(dt)
        return dt.astimezone(hkt_tz)
    except Exception:
        pass

    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return hkt_tz.localize(dt)
        except Exception:
            continue
    return None


def _status_indicates_active(status: str) -> bool:
    status_text = (status or "").strip().lower()
    active_keywords = ["active", "in progress", "ongoing", "work in progress", "commenced", "open"]
    inactive_keywords = ["completed", "finished", "cancelled", "closed", "ended", "suspended"]

    if any(word in status_text for word in inactive_keywords):
        return False
    if any(word in status_text for word in active_keywords):
        return True
    return False


def _is_roadwork_active(work: dict, ref_time=None) -> bool:
    ref_time = ref_time or datetime.now(hkt_tz)
    start_dt = _parse_roadwork_datetime(work.get("start_time", ""))
    end_dt = _parse_roadwork_datetime(work.get("end_time", ""))
    status = work.get("status", "")

    if start_dt and end_dt:
        return start_dt <= ref_time <= end_dt
    if start_dt and not end_dt:
        return ref_time >= start_dt and not any(
            word in str(status).lower() for word in ["completed", "finished", "cancelled", "closed", "ended"]
        )
    if end_dt and not start_dt:
        return ref_time <= end_dt and _status_indicates_active(status)

    return _status_indicates_active(status)


def build_construction_detail(distance_km: float, density_multiplier: float, avg_lat: float, avg_lon: float,
                              start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> dict:
    nearby_roadworks = get_nearby_roadworks_for_route(start_lat, start_lon, end_lat, end_lon)
    if nearby_roadworks:
        return {
            "estimated_work_zones": len(nearby_roadworks),
            "route_density_factor": round(density_multiplier, 2),
            "estimated_affected_distance_km": round(min(distance_km, 0.25 * len(nearby_roadworks)), 2),
            "area_type": "matched route corridor",
            "source": "Hong Kong Transport Department roadworks data matched near Google route",
            "confidence": "HIGH",
            "avg_route_coords": (round(avg_lat, 4), round(avg_lon, 4)),
            "matched_roadworks": nearby_roadworks,
        }

    estimated_work_zones = max(0, int(distance_km * 0.15 * density_multiplier))

    if density_multiplier >= 1.25:
        area_type = "dense urban corridor"
        confidence = "MEDIUM"
    elif density_multiplier <= 0.75:
        area_type = "suburban / lower-density corridor"
        confidence = "LOW"
    else:
        area_type = "mixed-density corridor"
        confidence = "LOW"

    return {
        "estimated_work_zones": estimated_work_zones,
        "route_density_factor": round(density_multiplier, 2),
        "estimated_affected_distance_km": round(distance_km * 0.12 * max(1.0, density_multiplier), 2),
        "area_type": area_type,
        "source": "Heuristic estimate based on route distance and urban density",
        "confidence": confidence,
        "avg_route_coords": (round(avg_lat, 4), round(avg_lon, 4)),
        "matched_roadworks": [],
    }


def build_construction_detail_heuristic(distance_km: float, density_multiplier: float, avg_lat: float, avg_lon: float) -> dict:
    estimated_work_zones = max(0, int(distance_km * 0.15 * density_multiplier))

    if density_multiplier >= 1.25:
        area_type = "dense urban corridor"
        confidence = "MEDIUM"
    elif density_multiplier <= 0.75:
        area_type = "suburban / lower-density corridor"
        confidence = "LOW"
    else:
        area_type = "mixed-density corridor"
        confidence = "LOW"

    return {
        "estimated_work_zones": estimated_work_zones,
        "route_density_factor": round(density_multiplier, 2),
        "estimated_affected_distance_km": round(distance_km * 0.12 * max(1.0, density_multiplier), 2),
        "area_type": area_type,
        "source": "Heuristic estimate based on route distance and urban density",
        "confidence": confidence,
        "avg_route_coords": (round(avg_lat, 4), round(avg_lon, 4)),
        "matched_roadworks": [],
    }


hkt_tz = pytz.timezone("Asia/Hong_Kong")
now = datetime.now(hkt_tz)
time_factor = get_time_factor(now.hour)


# ---------------- UI printing ---------------- #

def print_weather_report(lat: float, lon: float, location_name: str) -> None:
    weather = get_current_weather(lat, lon)
    if not weather:
        return

    temp = weather["temperature"]
    wind = weather["windspeed"]
    desc = describe_weather(weather["weathercode"])

    iso_time = weather["time"]
    hkt_time = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
    hkt_time = pytz.utc.localize(hkt_time).astimezone(pytz.timezone("Asia/Hong_Kong"))
    time_hk = hkt_time.strftime("%y/%m/%d %H:%M HKT")

    print(f"{location_name}")
    print(f" Temperature              : {temp:.1f} C")
    print(f" Condition                : {desc}")
    print(f" Wind Speed               : {wind:.1f} km/h")
    print(f" Updated At               : {time_hk}")


def build_weather_display(lat: float, lon: float, title: str) -> dict:
    weather = get_current_weather(lat, lon)
    if not weather:
        return {
            "title": title,
            "temperature": "Not Available",
            "condition": "Not Available",
            "wind_speed": "Not Available",
            "updated_at": "Not Available",
        }

    temp = weather["temperature"]
    wind = weather["windspeed"]
    desc = describe_weather(weather["weathercode"])
    iso_time = weather["time"]
    hkt_time = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
    hkt_time = pytz.utc.localize(hkt_time).astimezone(pytz.timezone("Asia/Hong_Kong"))
    time_hk = hkt_time.strftime("%y/%m/%d %H:%M HKT")

    return {
        "title": title,
        "temperature": f"{temp:.1f} C",
        "condition": desc,
        "wind_speed": f"{wind:.1f} km/h",
        "updated_at": time_hk,
    }


def print_weather_summary(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> None:
    origin = build_weather_display(start_lat, start_lon, "Route Origin")
    destination = build_weather_display(end_lat, end_lon, "Route Destination")

    left_width = 38
    right_width = 38

    def _pair(left_label: str, left_value: str, right_label: str, right_value: str) -> None:
        left_text = f"{left_label:<14}: {left_value}"
        right_text = f"{right_label:<14}: {right_value}"
        print(f"{left_text:<{left_width}}  {right_text:<{right_width}}")

    print(f"{origin['title']:<{left_width}}  {destination['title']:<{right_width}}")
    print(f"{'-' * left_width}  {'-' * right_width}")
    _pair("Temperature", origin["temperature"], "Temperature", destination["temperature"])
    _pair("Condition", origin["condition"], "Condition", destination["condition"])
    _pair("Wind Speed", origin["wind_speed"], "Wind Speed", destination["wind_speed"])
    _pair("Updated At", origin["updated_at"], "Updated At", destination["updated_at"])

def get_weather_data_quality(start_weather, end_weather) -> tuple[str, str]:
    if start_weather and end_weather:
        return "Open-Meteo current weather", "HIGH"
    if start_weather or end_weather:
        return "Open-Meteo current weather", "MEDIUM"
    return "Weather fallback / unavailable", "LOW"


def build_routing_confidence(start_meta: dict, dest_meta: dict, route_result: dict | None = None) -> dict:
    def _endpoint_score(meta: dict, resolved_place_id: str) -> int:
        provider = str(meta.get("provider", "")).strip()
        location_type = str(meta.get("location_type", "")).upper()
        score = float(meta.get("score", 0.0) or 0.0)
        points = 0

        if resolved_place_id:
            points += 2
        if provider == "GooglePlaces":
            points += 2
        elif provider == "GoogleGeocoding":
            points += 1

        if location_type in {"ROOFTOP", "PLACES_TEXTSEARCH"}:
            points += 2
        elif location_type == "RANGE_INTERPOLATED":
            points += 1
        elif location_type in {"GEOMETRIC_CENTER", "APPROXIMATE"}:
            points -= 1

        if score >= 15:
            points += 1
        elif score < 5:
            points -= 1

        if provider == "Fallback":
            points = min(points, 1)
        return points

    origin_place_id = ((route_result or {}).get("origin_place_id") or start_meta.get("place_id", ""))
    destination_place_id = ((route_result or {}).get("destination_place_id") or dest_meta.get("place_id", ""))
    origin_score = _endpoint_score(start_meta, origin_place_id)
    destination_score = _endpoint_score(dest_meta, destination_place_id)
    total_score = origin_score + destination_score

    if total_score >= 7:
        level = "HIGH"
    elif total_score >= 4:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level": level,
        "source": (route_result or {}).get("provider", "Google Directions"),
        "origin_resolution": start_meta.get("provider", "Unknown"),
        "destination_resolution": dest_meta.get("provider", "Unknown"),
        "poi_priority_applied": "Yes" if (
            start_meta.get("provider") == "GooglePlaces"
            or dest_meta.get("provider") == "GooglePlaces"
            or start_meta.get("source_type") == "places"
            or dest_meta.get("source_type") == "places"
        ) else "No",
    }


def append_route_analysis_log(record: dict) -> None:
    try:
        with open(ANALYSIS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def print_compare_summary(driving_result: dict, walking_result: dict) -> None:
    headers = [
        ("Mode", 10),
        ("Distance", 10),
        ("Time", 9),
        ("Risk Score", 12),
        ("Complexity", 12),
        ("Confidence", 12),
        ("Recommendation", 28),
    ]

    def _row(values: list[str]) -> str:
        return " ".join(f"{value:<{width}}" for value, (_, width) in zip(values, headers))

    divider = "-" * sum(width for _, width in headers)
    print_section_header("Compare Summary")
    print(_row([name for name, _ in headers]))
    print(divider)
    print(_row([
        "Driving",
        f"{driving_result['distance_km']:.1f} km",
        f"{driving_result['route_time_min']} min",
        f"{driving_result['risk_score']:.2f}",
        f"{driving_result['road_complexity']:.2f}",
        format_display_value(driving_result["routing_confidence"]),
        driving_result["recommendation"][:28],
    ]))
    print(_row([
        "Walking",
        f"{walking_result['distance_km']:.1f} km",
        f"{walking_result['route_time_min']} min",
        f"{walking_result['risk_score']:.2f}",
        f"{walking_result['road_complexity']:.2f}",
        format_display_value(walking_result["routing_confidence"]),
        walking_result["recommendation"][:28],
    ]))
    compare_ai = build_compare_ai_recommendation(driving_result, walking_result)
    print_section_header("AI Recommendation")
    print_report_line("Preferred Mode", compare_ai["preferred_mode"])
    print()
    print_report_line("Reason", compare_ai["reason"])


def print_report_line(label: str, value, label_width: int = 28, value_width: int = 72, indent: int = SECTION_CONTENT_INDENT) -> None:
    prefix = f"{label:<{label_width}}: "
    indent_text = " " * indent
    max_value_width = max(20, len(SECTION_DIVIDER) - indent - len(prefix))
    effective_width = min(value_width, max_value_width)
    wrapped = textwrap.wrap(str(value), width=effective_width) or [""]
    print(f"{indent_text}{prefix}{wrapped[0]}")
    for continuation in wrapped[1:]:
        print(f"{indent_text}{' ' * len(prefix)}{continuation}")


def print_section_header(title: str, leading_newline: bool = True) -> None:
    prefix = "\n" if leading_newline else ""
    print(f"{prefix}{SECTION_DIVIDER}")
    print(str(title).center(len(SECTION_DIVIDER)))
    print(SECTION_DIVIDER)


def print_centered_line(text: str, width: int | None = None) -> None:
    width = min(width or len(SECTION_DIVIDER), len(SECTION_DIVIDER))
    wrapped = textwrap.wrap(str(text), width=max(20, width - 4)) or [""]
    for line in wrapped:
        print(line.center(width))


def print_centered_section_header(title: str, leading_newline: bool = True, width: int | None = None) -> None:
    width = width or len(SECTION_DIVIDER)
    prefix = "\n" if leading_newline else ""
    print(f"{prefix}{SECTION_DIVIDER}")
    print(str(title).center(width))
    print(SECTION_DIVIDER)


def print_executive_summary_header(title: str, leading_newline: bool = True, width: int | None = None) -> None:
    width = width or len(SECTION_DIVIDER)
    strong_divider = "=" * len(SECTION_DIVIDER)
    prefix = "\n" if leading_newline else ""
    print(f"{prefix}{strong_divider}")
    print(str(title).upper().center(width))
    print(str("Key Route Decision Summary").center(width))
    print(strong_divider)


def print_two_column_panel(left_title: str, left_rows: list[tuple[str, str]], right_title: str, right_rows: list[tuple[str, str]],
                           col_width: int = PANEL_COL_WIDTH, label_width: int = 18, gap_width: int = PANEL_GAP_WIDTH) -> None:
    value_width = max(12, col_width - label_width - 2)

    def _format_rows(rows: list[tuple[str, str]]) -> list[str]:
        lines = []
        for label, value in rows:
            wrapped = textwrap.wrap(str(value), width=value_width) or [""]
            lines.append(f"{label:<{label_width}}: {wrapped[0]}")
            for continuation in wrapped[1:]:
                lines.append(f"{'':<{label_width}}  {continuation}")
        return lines

    left_lines = [left_title, "-" * col_width] + _format_rows(left_rows)
    right_lines = [right_title, "-" * col_width] + _format_rows(right_rows)
    max_lines = max(len(left_lines), len(right_lines))
    left_lines.extend([""] * (max_lines - len(left_lines)))
    right_lines.extend([""] * (max_lines - len(right_lines)))

    for left, right in zip(left_lines, right_lines):
        print(f"{left:<{col_width}}{' ' * gap_width}{right:<{col_width}}")


def format_display_value(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "Not Available"

    replacements = {
        "Not available": "Not Available",
        "LOW": "Low",
        "MEDIUM": "Medium",
        "HIGH": "High",
        "XGBoost": "XGBoost Model",
        "TeacherRules": "Teacher Rules",
        "Unknown": "Not Available",
    }
    return replacements.get(text, text)


def format_risk_threshold_text() -> str:
    return "Low <= 4.0, Medium 4.0-7.0, High > 7.0"


def _load_cache(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _in_hk_bounds(lat: float, lon: float) -> bool:
    # rough HK bbox
    return 22.15 <= lat <= 22.60 and 113.80 <= lon <= 114.45


def _normalize_address_text(text: str) -> str:
    text = (text or "").lower().strip()
    replacements = {
        ".": " ",
        ",": " ",
        "-": " ",
        " rd ": " road ",
        " rd.": " road",
        " st ": " street ",
        " st.": " street",
        " ave ": " avenue ",
        " kowloon hong kong": "kowloon hong kong",
    }
    text = f" {text} "
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_hk_address(query: str) -> dict:
    normalized = _normalize_address_text(query)
    number_match = re.search(r"\b(\d+[a-z]?)\b", normalized)
    street_number = number_match.group(1) if number_match else ""

    directional = ""
    for token in ("west", "east", "north", "south"):
        if re.search(rf"\b{token}\b", normalized):
            directional = token
            break

    district_hint = ""
    for token in ("kowloon", "new territories", "hong kong island", "hong kong"):
        if token in normalized:
            district_hint = token
            break

    return {
        "normalized": normalized,
        "street_number": street_number,
        "directional": directional,
        "district_hint": district_hint,
    }

def _google_geocode_candidates(query: str, limit: int = 5) -> list:
    """
    Returns list of candidates:
      [{lat, lon, formatted_address, location_type, types, place_id}, ...]
    Uses Geocoding API (simple). You can swap to Places if you prefer.
    """
    if not GOOGLE_MAPS_API_KEY:
        return []

    parsed = _parse_hk_address(query)
    q = query.strip()
    if "hong kong" not in q.lower():
        if parsed["district_hint"] and parsed["district_hint"] != "hong kong":
            q = f"{q}, {parsed['district_hint'].title()}, Hong Kong"
        else:
            q = q + ", Hong Kong"

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": q,
        "key": GOOGLE_MAPS_API_KEY,
        "region": "hk",
        "components": "country:HK",
        "bounds": "22.15,113.80|22.60,114.45",
    }
    data = requests.get(url, params=params, timeout=15).json()

    out = []
    for r in (data.get("results") or [])[:limit]:
        loc = r.get("geometry", {}).get("location") or {}
        if "lat" not in loc or "lng" not in loc:
            continue
        out.append({
            "lat": float(loc["lat"]),
            "lon": float(loc["lng"]),
            "formatted_address": r.get("formatted_address", ""),
            "location_type": r.get("geometry", {}).get("location_type", "UNKNOWN"),
            "types": r.get("types", []),
            "place_id": r.get("place_id", ""),
            "address_components": r.get("address_components", []),
        })

    return out


def _google_places_candidates(query: str, limit: int = 5) -> list:
    if not GOOGLE_MAPS_API_KEY or googlemaps is None:
        return []

    parsed = _parse_hk_address(query)
    q = query.strip()
    if "hong kong" not in q.lower():
        if parsed["district_hint"] and parsed["district_hint"] != "hong kong":
            q = f"{q}, {parsed['district_hint'].title()}, Hong Kong"
        else:
            q = q + ", Hong Kong"

    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        data = gmaps.places(query=q, region="hk", language="en")
    except Exception:
        return []

    out = []
    for r in (data.get("results") or [])[:limit]:
        loc = r.get("geometry", {}).get("location") or {}
        if "lat" not in loc or "lng" not in loc:
            continue

        formatted_address = r.get("formatted_address") or r.get("name") or ""
        if r.get("formatted_address") and r.get("name"):
            formatted_address = f"{r.get('name')} - {r.get('formatted_address')}"

        out.append({
            "lat": float(loc["lat"]),
            "lon": float(loc["lng"]),
            "formatted_address": formatted_address,
            "location_type": "PLACES_TEXTSEARCH",
            "types": r.get("types", []),
            "place_id": r.get("place_id", ""),
            "address_components": [],
            "name": r.get("name", ""),
            "source_type": "places",
        })

    return out


def _merge_google_candidates(*candidate_lists: list[dict]) -> list[dict]:
    merged = []
    seen_place_ids = set()
    seen_latlon = set()

    for candidates in candidate_lists:
        for candidate in candidates:
            place_id = str(candidate.get("place_id") or "").strip()
            if place_id:
                if place_id in seen_place_ids:
                    continue
                seen_place_ids.add(place_id)
            else:
                latlon_key = (round(float(candidate.get("lat", 0.0)), 6), round(float(candidate.get("lon", 0.0)), 6))
                if latlon_key in seen_latlon:
                    continue
                seen_latlon.add(latlon_key)

            merged.append(candidate)

    return merged


def _is_road_like_candidate(candidate: dict) -> bool:
    location_type = str(candidate.get("location_type", "")).upper()
    types = set(candidate.get("types", []))
    return (
        location_type in {"GEOMETRIC_CENTER", "APPROXIMATE", "RANGE_INTERPOLATED"}
        or "route" in types
        or "street_address" in types
    ) and candidate.get("source_type") != "places"


def _is_poi_candidate(candidate: dict) -> bool:
    types = set(candidate.get("types", []))
    poi_types = {"parking", "point_of_interest", "establishment", "premise", "subpremise"}
    return candidate.get("source_type") == "places" and bool(types.intersection(poi_types))


def _query_prefers_poi(user_query: str) -> bool:
    normalized = _normalize_address_text(user_query)
    poi_keywords = [
        "car park",
        "parking",
        "carpark",
        "car park entrance",
        "parking entrance",
        "\u505c\u8eca\u5834",
        "\u505c\u8f66\u573a",
        "\u5165\u53e3",
        "\u5927\u5ec8",
        "\u5927\u53a6",
        "\u5546\u5834",
        "\u5546\u573a",
        "\u82b1\u5712",
        "\u82b1\u56ed",
        "\u95a3",
        "\u9601",
        "\u5c45",
        "building",
        "estate",
        "tower",
        "mall",
        "plaza",
    ]
    return any(keyword in normalized for keyword in poi_keywords)

def _score_candidate(user_query: str, c: dict) -> float:
    """
    Simple 'AI-like' scoring using Google quality signals + sanity checks.
    Higher is better.
    """
    score = 0.0
    parsed = _parse_hk_address(user_query)
    normalized_query = parsed["normalized"]
    formatted_address = _normalize_address_text(c.get("formatted_address", ""))
    candidate_name = _normalize_address_text(c.get("name", ""))
    components = c.get("address_components", [])
    component_long_names = " ".join(_normalize_address_text(comp.get("long_name", "")) for comp in components)
    component_short_names = " ".join(_normalize_address_text(comp.get("short_name", "")) for comp in components)
    combined_address = f"{candidate_name} {formatted_address} {component_long_names} {component_short_names}".strip()

    # Location type preference (Google's precision hint)
    lt = (c.get("location_type") or "").upper()
    if lt == "ROOFTOP":
        score += 5.0
    elif lt == "RANGE_INTERPOLATED":
        score += 3.0
    elif lt == "GEOMETRIC_CENTER":
        score += 1.0
    elif lt == "APPROXIMATE":
        score += 0.5
    elif lt == "PLACES_TEXTSEARCH":
        score += 2.5

    # Must be in HK-ish bounds
    if _in_hk_bounds(c["lat"], c["lon"]):
        score += 2.0
    else:
        score -= 10.0

    # Prefer results whose formatted address mentions Hong Kong
    if "hong kong" in combined_address:
        score += 1.0

    # Stronger lexical scoring for HK addresses
    query_tokens = [t for t in normalized_query.split() if len(t) >= 3]
    matched_tokens = sum(1 for token in query_tokens if token in combined_address)
    score += min(3.0, matched_tokens * 0.45)

    if parsed["street_number"]:
        if re.search(rf"\b{re.escape(parsed['street_number'])}\b", combined_address):
            score += 3.0
        else:
            score -= 2.5

    if parsed["directional"]:
        if re.search(rf"\b{re.escape(parsed['directional'])}\b", combined_address):
            score += 2.0
        else:
            score -= 2.0

    if parsed["district_hint"] and parsed["district_hint"] != "hong kong":
        if parsed["district_hint"] in combined_address:
            score += 1.5
        else:
            score -= 0.5

    # Prefer street-address style results over broad area/place matches.
    types = set(c.get("types", []))
    if "street_address" in types:
        score += 3.0
    elif "premise" in types or "subpremise" in types:
        score += 2.0
    elif "route" in types:
        score += 1.0
    elif "plus_code" in types:
        score -= 1.5

    if c.get("source_type") == "places":
        score += 1.5

    poi_types = {"parking", "point_of_interest", "establishment", "premise", "subpremise"}
    poi_matches = types.intersection(poi_types)
    if poi_matches:
        score += min(2.0, 0.6 * len(poi_matches))
        if _query_prefers_poi(user_query):
            score += 2.0

    return score

def geocode_confirmed(query: str, interactive: bool = True):
    """
    Returns (lat, lon, meta_dict) or (None, None, meta_dict).
    meta_dict includes provider + chosen candidate details.
    """
    # 1) Try Google candidates
    prefer_poi = _query_prefers_poi(query)
    places_cands = _google_places_candidates(query, limit=5)
    geocode_cands = [] if prefer_poi and places_cands else _google_geocode_candidates(query, limit=5)
    cands = _merge_google_candidates(geocode_cands, places_cands)
    if cands:
        ranked = sorted(cands, key=lambda c: _score_candidate(query, c), reverse=True)
        best = ranked[0]
        best_score = _score_candidate(query, best)
        parsed = _parse_hk_address(query)
        auto_accept_threshold = 8.0 if parsed["street_number"] else 6.5
        poi_ranked = [c for c in ranked if _is_poi_candidate(c)]
        should_offer_poi_first = (
            len(poi_ranked) > 0
            and (
                _is_road_like_candidate(best)
                or _query_prefers_poi(query)
            )
        )

        # If high confidence, auto-accept
        if best_score >= auto_accept_threshold and not should_offer_poi_first:
            meta = dict(best)
            meta["provider"] = "GooglePlaces" if best.get("source_type") == "places" else "GoogleGeocoding"
            meta["confirm_mode"] = "auto"
            meta["score"] = best_score
            return best["lat"], best["lon"], meta

        # Otherwise, ask user to choose (delivery-safe)
        if interactive:
            if should_offer_poi_first:
                print("\nNOTICE: A more specific POI / entrance candidate was found for this address.")
                print("Please choose the destination that best matches your actual routing target:")
                poi_options = poi_ranked[:5]
                for i, c in enumerate(poi_options, start=1):
                    s = _score_candidate(query, c)
                    print(f"{i}) {c.get('formatted_address', '')}")
                    print(
                        f"   - source={c.get('source_type', 'places')} "
                        f"location_type={c.get('location_type')} "
                        f"score={s:.1f} latlon=({c['lat']:.5f},{c['lon']:.5f})"
                    )

                print("0) Use the normal road-address match instead")
                ans = input("Select 0-5 (or press Enter to use #1): ").strip()
                if ans == "0":
                    pass
                else:
                    idx = 1
                    if ans.isdigit():
                        idx = max(1, min(len(poi_options), int(ans)))
                    chosen = poi_options[idx - 1]
                    meta = dict(chosen)
                    meta["provider"] = "GooglePlaces"
                    meta["confirm_mode"] = "poi_choice"
                    meta["score"] = _score_candidate(query, chosen)
                    return chosen["lat"], chosen["lon"], meta

            print("\nWARNING: Location is ambiguous. Please confirm the correct address:")
            for i, c in enumerate(ranked[:5], start=1):
                s = _score_candidate(query, c)
                print(f"{i}) {c.get('formatted_address','')}")
                print(
                    f"   - source={c.get('source_type', 'geocoding')} "
                    f"location_type={c.get('location_type')} "
                    f"score={s:.1f} latlon=({c['lat']:.5f},{c['lon']:.5f})"
                )

            ans = input("Select 1-5 (or press Enter to use #1): ").strip()
            idx = 1
            if ans.isdigit():
                idx = max(1, min(5, int(ans)))
            chosen = ranked[idx - 1]
            meta = dict(chosen)
            meta["provider"] = "GooglePlaces" if chosen.get("source_type") == "places" else "GoogleGeocoding"
            meta["confirm_mode"] = "user_choice"
            meta["score"] = _score_candidate(query, chosen)
            return chosen["lat"], chosen["lon"], meta

        # Non-interactive: return best but mark low precision
        meta = dict(best)
        meta["provider"] = "GooglePlaces" if best.get("source_type") == "places" else "GoogleGeocoding"
        meta["confirm_mode"] = "low_confidence_auto"
        meta["score"] = best_score
        return best["lat"], best["lon"], meta

    # 2) Fallback to your existing geocode_with_cache (legacy)
    coords = geocode_with_cache(query)
    if coords:
        lat, lon = coords
        return lat, lon, {"provider": "LegacyGeocoder", "confirm_mode": "fallback"}

    return None, None, {"provider": "None", "confirm_mode": "fail"}

# ---------------- HK district ---------------- #

def get_google_district_from_meta(meta: dict | None) -> str | None:
    if not meta:
        return None

    components = meta.get("address_components") or []
    priority_types = [
        "administrative_area_level_2",
        "sublocality_level_1",
        "sublocality",
        "locality",
        "neighborhood",
    ]

    district_aliases = {
        "kowloon tsai": "Kowloon City",
        "kowloon tong": "Kowloon City",
        "ho man tin": "Kowloon City",
        "hung hom": "Kowloon City",
        "ma tau wai": "Kowloon City",
        "ma tau kok": "Kowloon City",
        "to kwa wan": "Kowloon City",
        "kowloon city": "Kowloon City",
        "yau tsim mong": "Yau Tsim Mong",
        "mong kok": "Yau Tsim Mong",
        "tsim sha tsui": "Yau Tsim Mong",
        "jordan": "Yau Tsim Mong",
        "sham shui po": "Sham Shui Po",
        "cheung sha wan": "Sham Shui Po",
        "lai chi kok": "Sham Shui Po",
        "mei foo": "Sham Shui Po",
        "wong tai sin": "Wong Tai Sin",
        "diamond hill": "Wong Tai Sin",
        "lok fu": "Wong Tai Sin",
        "san po kong": "Wong Tai Sin",
        "kwun tong": "Kwun Tong",
        "lam tin": "Kwun Tong",
        "yau tong": "Kwun Tong",
        "ngau tau kok": "Kwun Tong",
        "central and western district": "Central & Western",
        "central & western": "Central & Western",
        "wan chai": "Wan Chai",
        "causeway bay": "Wan Chai",
        "eastern district": "Eastern",
        "eastern": "Eastern",
        "quarry bay": "Eastern",
        "north point": "Eastern",
        "chai wan": "Eastern",
        "southern district": "Southern",
        "southern": "Southern",
        "islands district": "Islands",
        "islands": "Islands",
        "kwai tsing": "Kwai Tsing",
        "kwai chung": "Kwai Tsing",
        "tsing yi": "Kwai Tsing",
        "north district": "North",
        "north": "North",
        "sai kung": "Sai Kung",
        "sha tin": "Sha Tin",
        "tai po": "Tai Po",
        "tsuen wan": "Tsuen Wan",
        "tuen mun": "Tuen Mun",
        "yuen long": "Yuen Long",
    }

    for target_type in priority_types:
        for component in components:
            comp_types = component.get("types", [])
            if target_type not in comp_types:
                continue
            candidate = str(component.get("long_name") or component.get("short_name") or "").strip()
            if not candidate:
                continue
            mapped = district_aliases.get(candidate.lower())
            if mapped:
                return mapped

    formatted_address = str(meta.get("formatted_address", "")).lower()
    for key, mapped in district_aliases.items():
        if key in formatted_address:
            return mapped
    return None


def get_district_from_formatted_address(meta: dict | None) -> str | None:
    if not meta:
        return None

    formatted_address = str(meta.get("formatted_address", "")).strip()
    if not formatted_address:
        return None

    district_aliases = {
        "kowloon tsai": "Kowloon City",
        "kowloon tong": "Kowloon City",
        "ho man tin": "Kowloon City",
        "hung hom": "Kowloon City",
        "ma tau wai": "Kowloon City",
        "ma tau kok": "Kowloon City",
        "to kwa wan": "Kowloon City",
        "kowloon city": "Kowloon City",
        "mong kok": "Yau Tsim Mong",
        "tsim sha tsui": "Yau Tsim Mong",
        "jordan": "Yau Tsim Mong",
        "yau ma tei": "Yau Tsim Mong",
        "cheung sha wan": "Sham Shui Po",
        "lai chi kok": "Sham Shui Po",
        "mei foo": "Sham Shui Po",
        "sham shui po": "Sham Shui Po",
        "diamond hill": "Wong Tai Sin",
        "lok fu": "Wong Tai Sin",
        "san po kong": "Wong Tai Sin",
        "wong tai sin": "Wong Tai Sin",
        "kwun tong": "Kwun Tong",
        "lam tin": "Kwun Tong",
        "yau tong": "Kwun Tong",
        "ngau tau kok": "Kwun Tong",
        "central and western": "Central & Western",
        "central": "Central & Western",
        "sheung wan": "Central & Western",
        "wan chai": "Wan Chai",
        "causeway bay": "Wan Chai",
        "north point": "Eastern",
        "quarry bay": "Eastern",
        "chai wan": "Eastern",
        "eastern": "Eastern",
        "aberdeen": "Southern",
        "repulse bay": "Southern",
        "southern": "Southern",
        "tsing yi": "Kwai Tsing",
        "kwai chung": "Kwai Tsing",
        "kwai tsing": "Kwai Tsing",
        "sha tin": "Sha Tin",
        "tai wai": "Sha Tin",
        "fo tan": "Sha Tin",
        "tai po": "Tai Po",
        "tsuen wan": "Tsuen Wan",
        "tuen mun": "Tuen Mun",
        "yuen long": "Yuen Long",
        "sai kung": "Sai Kung",
        "islands": "Islands",
    }

    normalized = formatted_address.lower()
    address_parts = [part.strip().lower() for part in formatted_address.split(",") if part.strip()]

    for part in reversed(address_parts):
        for key, mapped in district_aliases.items():
            if key in part:
                return mapped

    for key, mapped in district_aliases.items():
        if key in normalized:
            return mapped

    return None


def get_hk_district_with_source(lat: float, lon: float, meta: dict | None = None) -> tuple[str, str]:
    google_district = get_google_district_from_meta(meta)
    if google_district:
        return google_district, "Google Components"

    formatted_address_district = get_district_from_formatted_address(meta)
    if formatted_address_district:
        return formatted_address_district, "Formatted Address Mapping"

    if get_polygon_district is not None:
        try:
            polygon_district = str(get_polygon_district(lat, lon)).strip()
            if polygon_district and polygon_district.lower() != "unknown":
                return polygon_district, "Polygon Lookup"
        except Exception:
            pass

    # Kowloon
    if 22.27 <= lat <= 22.38 and 114.15 <= lon <= 114.23:
        if 114.16 <= lon <= 114.185 and 22.29 <= lat <= 22.325:
            return "Yau Tsim Mong", "Coordinate Fallback"
        if 114.185 <= lon <= 114.205 and 22.33 <= lat <= 22.37:
            return "Sham Shui Po", "Coordinate Fallback"
        if 114.205 <= lon <= 114.225 and 22.315 <= lat <= 22.345:
            return "Kowloon City", "Coordinate Fallback"
        if 22.335 <= lat <= 22.36 and 114.20 <= lon <= 114.225:
            return "Wong Tai Sin", "Coordinate Fallback"
        if 114.20 <= lon <= 114.23:
            return "Kwun Tong", "Coordinate Fallback"

    # HK Island
    if 22.22 <= lat <= 22.32 and 114.10 <= lon <= 114.22:
        if 114.15 <= lon <= 114.18:
            return "Central & Western", "Coordinate Fallback"
        if lon <= 114.20 and lat <= 22.28:
            return "Wan Chai", "Coordinate Fallback"
        if 114.20 <= lon <= 114.22:
            return "Eastern", "Coordinate Fallback"
        return "Southern", "Coordinate Fallback"

    # New Territories
    if lat >= 22.35:
        if lon <= 114.05:
            if lat <= 22.42:
                return "Tuen Mun", "Coordinate Fallback"
            return "Yuen Long", "Coordinate Fallback"
        if 114.05 < lon <= 114.15:
            return "Kwai Tsing", "Coordinate Fallback"
        if 114.15 <= lon <= 114.25 and lat <= 22.42:
            return "Sha Tin", "Coordinate Fallback"
        if lon >= 114.25:
            return "Sai Kung", "Coordinate Fallback"
        return "Tai Po / North", "Coordinate Fallback"

    # Islands
    if lon <= 114.05 and lat <= 22.35:
        return "Islands", "Coordinate Fallback"

    return "Not Available", "Not Available"


def get_hk_district(lat: float, lon: float, meta: dict | None = None) -> str:
    district, _source = get_hk_district_with_source(lat, lon, meta)
    return district


# ---------------- Geometry helpers ---------------- #

def calculate_straight_distance(lat1, lon1, lat2, lon2) -> float:
    """Straight-line (haversine) distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return round(R * c, 1)


def calculate_straight_distance_precise(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _point_to_route_distance_km(lat: float, lon: float, start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> float:
    mean_lat = radians((start_lat + end_lat + lat) / 3.0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(mean_lat)

    sx, sy = start_lon * km_per_deg_lon, start_lat * km_per_deg_lat
    ex, ey = end_lon * km_per_deg_lon, end_lat * km_per_deg_lat
    px, py = lon * km_per_deg_lon, lat * km_per_deg_lat

    dx, dy = ex - sx, ey - sy
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return sqrt((px - sx) ** 2 + (py - sy) ** 2)

    t = ((px - sx) * dx + (py - sy) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = sx + t * dx
    proj_y = sy + t * dy
    return sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def get_nearby_roadworks_for_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> list[dict]:
    try:
        resp = requests.get(HK_ROADWORKS_JSON_URL, timeout=20)
        resp.raise_for_status()
        roadworks = _extract_roadwork_points(resp.json())
    except Exception:
        return []

    route_length_km = max(0.1, calculate_straight_distance_precise(start_lat, start_lon, end_lat, end_lon))
    corridor_km = 0.18 if route_length_km <= 2.0 else 0.25
    nearby = []

    active_roadworks = [work for work in roadworks if _is_roadwork_active(work)]

    for work in active_roadworks:
        dist_to_route = _point_to_route_distance_km(
            work["lat"], work["lon"], start_lat, start_lon, end_lat, end_lon
        )
        if dist_to_route <= corridor_km:
            item = dict(work)
            item["distance_to_route_km"] = round(dist_to_route, 3)
            nearby.append(item)

    nearby.sort(key=lambda x: x["distance_to_route_km"])
    return nearby[:5]


# ---------------- Overpass bbox query (safe) ---------------- #

def get_osm_traffic_signals_bbox(north, south, east, west):
    """Overpass bbox query with cache + backoff. Returns int or None."""
    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
    ]

    # Cache key (rounded)
    key = f"{south:.4f},{west:.4f},{north:.4f},{east:.4f}"
    cache = _load_osm_cache()
    if key in cache:
        return int(cache[key])

    # Tighten bbox slightly
    margin_lat = (north - south) * 0.05
    margin_lon = (east - west) * 0.05
    tight_north = north + margin_lat
    tight_south = south - margin_lat
    tight_east = east + margin_lon  
    tight_west = west - margin_lon

    query = f"""
    [out:json][timeout:20];
      (
        node["highway"="traffic_signals"]({tight_south},{tight_west},{tight_north},{tight_east});
        way["highway"="traffic_signals"]({tight_south},{tight_west},{tight_north},{tight_east});
      );
    out center;
    """

    for attempt in range(1, 4):
        url = overpass_urls[(attempt - 1) % len(overpass_urls)]
        try:
            resp = requests.post(url, data=query, timeout=30)

            if resp.status_code in (429, 504):
                wait_s = 5 * (2 ** (attempt - 1))
                print(f"WARNING: Overpass HTTP {resp.status_code}. Waiting {wait_s}s (attempt {attempt}/3)")
                time.sleep(wait_s)
                continue

            if resp.status_code != 200:
                print(f"WARNING: Overpass HTTP {resp.status_code} (attempt {attempt}/3)")
                time.sleep(2 * attempt)
                continue

            data = resp.json()
            elements = data.get("elements", [])

            # Filter: within 250m of bbox midpoint
            route_signals = 0
            mid_lat = (south + north) / 2
            mid_lon = (west + east) / 2

            for elem in elements:
                if "lat" in elem and "lon" in elem:
                    elem_lat, elem_lon = elem["lat"], elem["lon"]
                elif "center" in elem:
                    elem_lat = elem["center"].get("lat")
                    elem_lon = elem["center"].get("lon")
                else:
                    continue

                if elem_lat is None or elem_lon is None:
                    continue

                dist_to_center = calculate_straight_distance(mid_lat, mid_lon, elem_lat, elem_lon)
                if dist_to_center < 0.25:
                    route_signals += 1

            cache[key] = int(route_signals)
            _save_osm_cache(cache)

            print(f"OSM bbox signals: {len(elements)}, route-relevant (<250m): {route_signals}")
            return int(route_signals)

        except Exception as e:
            print(f"WARNING: Overpass error: {e} (attempt {attempt}/3)")
            time.sleep(2 * attempt)

    return None


# ---------------- Signals fallback logic ---------------- #

def estimate_signals(distance_km: float, avg_lat: float, avg_lon: float) -> int:
    # Calibrated HK density heuristic for fallback use.
    if 114.15 < avg_lon < 114.21 and 22.27 < avg_lat < 22.34:
        per_km = 6.5
    elif 113.90 < avg_lon < 114.05:
        per_km = 2.0
    else:
        per_km = 4.5
    return int(max(1, round(per_km * max(0.5, distance_km))))


def estimate_signals_from_route_steps(distance_km: float, step_count: int, avg_lat: float, avg_lon: float) -> int:
    """Use Google route steps as an intersection proxy when OSM is unavailable."""
    baseline = estimate_signals(distance_km, avg_lat, avg_lon)
    if step_count <= 0:
        return baseline

    step_multiplier = 1.7 if (114.15 < avg_lon < 114.21 and 22.27 < avg_lat < 22.34) else 1.3
    step_based = max(1, int(round(step_count * step_multiplier)))
    return max(baseline, step_based)


def estimate_signals_from_google_route(distance_km: float, step_count: int, turn_count: int,
                                       avg_lat: float, avg_lon: float) -> int:
    """
    Google Directions does not provide an exact traffic-light count.
    This uses route steps + turn maneuvers as a conservative Google-based intersection proxy.
    """
    baseline = estimate_signals(distance_km, avg_lat, avg_lon)
    if step_count <= 0 and turn_count <= 0:
        return max(1, min(max(1, int(round(distance_km * 2.5))), baseline))

    # Google steps/turns describe route structure, not literal signal poles.
    # Keep this estimate deliberately conservative to avoid overstating physical traffic lights.
    step_proxy = max(1, int(round(step_count * 0.22)))
    turn_proxy = max(0, int(round(turn_count * 0.30)))

    blended = int(round((baseline * 0.20) + (step_proxy * 0.45) + (turn_proxy * 0.35)))
    max_reasonable = max(1, int(round(distance_km * 2.5)))
    return max(1, min(max_reasonable, blended))


def _sample_path_for_elevation(start_lat: float, start_lon: float, end_lat: float, end_lon: float, steps: list) -> list[tuple[float, float]]:
    sampled = [(start_lat, start_lon)]
    for step in steps:
        start_loc = step.get("start_location", {})
        end_loc = step.get("end_location", {})
        s_lat = _extract_numeric(start_loc.get("lat"))
        s_lon = _extract_numeric(start_loc.get("lng"))
        e_lat = _extract_numeric(end_loc.get("lat"))
        e_lon = _extract_numeric(end_loc.get("lng"))

        if s_lat is not None and s_lon is not None:
            sampled.append((float(s_lat), float(s_lon)))
        if e_lat is not None and e_lon is not None:
            sampled.append((float(e_lat), float(e_lon)))

    sampled.append((end_lat, end_lon))

    deduped = []
    for lat, lon in sampled:
        if not deduped or abs(deduped[-1][0] - lat) > 1e-6 or abs(deduped[-1][1] - lon) > 1e-6:
            deduped.append((lat, lon))
    return deduped


def get_google_elevation_gain(path_points: list[tuple[float, float]]) -> tuple[int | None, str]:
    api_key = get_google_maps_api_key()
    if not api_key or len(path_points) < 2:
        return None, "Elevation fallback"

    try:
        samples = min(64, max(8, len(path_points) * 3))
        path_param = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in path_points[:80])
        url = "https://maps.googleapis.com/maps/api/elevation/json"
        params = {
            "path": path_param,
            "samples": samples,
            "key": api_key,
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            return None, f"Elevation fallback ({data.get('status', 'UNKNOWN')})"

        results = data.get("results", [])
        if len(results) < 2:
            return None, "Elevation fallback"

        gain = 0.0
        for prev, curr in zip(results, results[1:]):
            delta = float(curr.get("elevation", 0.0)) - float(prev.get("elevation", 0.0))
            if delta > 0:
                gain += delta
        return int(round(gain)), "Google Elevation API"
    except Exception:
        return None, "Elevation fallback"


def get_signals_safe(distance_km: float, start_lat: float, start_lon: float, end_lat: float, end_lon: float,
                     step_count: int = 0, google_signal_estimate: int | None = None) -> int:
    avg_lat = (start_lat + end_lat) / 2
    avg_lon = (start_lon + end_lon) / 2

    # Training: never call OSM
    if DISABLE_OSM:
        if USE_GOOGLE_SIGNAL_PROXY and google_signal_estimate is not None:
            return int(google_signal_estimate)
        if USE_GOOGLE_SIGNAL_PROXY and step_count > 0:
            return estimate_signals_from_google_route(distance_km, step_count, step_count, avg_lat, avg_lon)
        return estimate_signals_from_route_steps(distance_km, step_count, avg_lat, avg_lon)

    if not USE_OSM_SIGNAL_FALLBACK:
        if USE_GOOGLE_SIGNAL_PROXY and google_signal_estimate is not None:
            return int(google_signal_estimate)
        if USE_GOOGLE_SIGNAL_PROXY and step_count > 0:
            return estimate_signals_from_google_route(distance_km, step_count, step_count, avg_lat, avg_lon)
        return estimate_signals(distance_km, avg_lat, avg_lon)

    # Runtime: prefer OSM-based counts when available.
    if get_real_traffic_signals_osm is not None:
        try:
            s = get_real_traffic_signals_osm(start_lat, start_lon, end_lat, end_lon, distance_km)
            if isinstance(s, (int, float)) and s > 0:
                return int(s)
        except Exception:
            pass

    # Runtime: bbox method
    try:
        lat_margin = abs(end_lat - start_lat) * 0.2
        lon_margin = abs(end_lon - start_lon) * 0.2
        north = max(start_lat, end_lat) + lat_margin
        south = min(start_lat, end_lat) - lat_margin
        east = max(start_lon, end_lon) + lon_margin
        west = min(start_lon, end_lon) - lon_margin

        s2 = get_osm_traffic_signals_bbox(north, south, east, west)
        if isinstance(s2, (int, float)) and s2 > 0:
            return int(s2)
    except Exception:
        pass

    if USE_GOOGLE_SIGNAL_PROXY and google_signal_estimate is not None:
        return int(google_signal_estimate)

    if USE_GOOGLE_SIGNAL_PROXY and step_count > 0:
        return estimate_signals_from_google_route(distance_km, step_count, step_count, avg_lat, avg_lon)

    # Final fallback
    return estimate_signals_from_route_steps(distance_km, step_count, avg_lat, avg_lon)


# ---------------- Weather + geo risk (kept) ---------------- #

def calculate_route_risk(w_start, w_dest, g_start, g_dest):
    w_start_desc = describe_weather(w_start["weathercode"]).lower()
    w_dest_desc = describe_weather(w_dest["weathercode"]).lower()

    def weather_risk(desc: str) -> float:
        if any(word in desc for word in ["heavy rain", "thunderstorm", "heavy snow"]):
            return 3.0
        if any(word in desc for word in ["moderate rain", "fog", "dense drizzle"]):
            return 1.5
        if any(word in desc for word in ["light drizzle", "slight rain", "rain showers"]):
            return 0.5
        return 0.0

    weather_score = weather_risk(w_start_desc) + weather_risk(w_dest_desc)
    geo_start = g_start.get("complexity_score", 5.0)
    geo_dest = g_dest.get("complexity_score", 5.0)
    geo_score = (geo_start + geo_dest) / 2

    total_score = min(weather_score + geo_score, 10)

    if total_score >= 8.0:
        level = "HIGH"
        reason = "Heavy rain/thunderstorm + busy roads" if weather_score >= 3 else "Very busy roads/construction"
    elif total_score >= 5.5:
        level = "MEDIUM"
        reason = f"Light rain ({weather_score:.1f}pts) + busy roads" if weather_score > 0 else "Moderately busy roads"
    else:
        level = "LOW"
        reason = "Clear weather, no precipitation"

    return {"score": total_score, "level": level, "reason": reason}


# ---------------- Routing distance ---------------- #

def get_google_route_metrics(start_lat, start_lon, end_lat, end_lon) -> dict | None:
    """Google Directions route metrics with live traffic when a key is available."""
    if not GOOGLE_MAPS_API_KEY:
        return None

    try:
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": f"{start_lat},{start_lon}",
            "destination": f"{end_lat},{end_lon}",
            "mode": "driving",
            "departure_time": "now",
            "traffic_model": "best_guess",
            "region": "hk",
            "key": GOOGLE_MAPS_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK" or not data.get("routes"):
            print(f"WARNING: Google Directions unavailable: {data.get('status', 'UNKNOWN')}")
            return None

        route = data["routes"][0]
        leg = route["legs"][0]
        distance_km = float(leg["distance"]["value"]) / 1000.0
        duration_sec = leg.get("duration_in_traffic", leg["duration"])["value"]
        duration_min = max(3, int(round(duration_sec / 60.0)))
        steps = leg.get("steps", [])

        print(f"Google Directions: {distance_km:.1f}km, {duration_min} min")
        return {
            "distance_km": round(distance_km, 1),
            "duration_min": duration_min,
            "step_count": len(steps),
            "provider": "Google Directions",
            "summary": route.get("summary", ""),
        }
    except Exception as e:
        print(f"Google Directions error: {type(e).__name__}: {e}")
        return None


def get_google_route_metrics_client(start_lat, start_lon, end_lat, end_lon,
                                    travel_mode: str = "driving",
                                    origin_meta: dict | None = None,
                                    destination_meta: dict | None = None,
                                    include_elevation: bool = True) -> dict | None:
    """Google Directions via the official Python client."""
    api_key = get_google_maps_api_key()
    key_source = "LOCAL_GOOGLE_MAPS_API_KEY" if LOCAL_GOOGLE_MAPS_API_KEY.strip() else "GOOGLE_MAPS_API_KEY"

    if not api_key:
        print("Google Directions unavailable: GOOGLE_MAPS_API_KEY is missing")
        return None

    if googlemaps is None:
        print("Google Directions unavailable: install package with 'pip install googlemaps'")
        return None

    try:
        print(f"Using Google API key from {key_source}: {mask_api_key(api_key)}")
        gmaps = googlemaps.Client(key=api_key)
        origin_place_id = (origin_meta or {}).get("place_id", "")
        destination_place_id = (destination_meta or {}).get("place_id", "")

        origin_value = f"place_id:{origin_place_id}" if origin_place_id else (start_lat, start_lon)
        destination_value = f"place_id:{destination_place_id}" if destination_place_id else (end_lat, end_lon)

        request_kwargs = {
            "origin": origin_value,
            "destination": destination_value,
            "mode": travel_mode,
            "region": "hk",
        }
        if travel_mode == "driving":
            request_kwargs["departure_time"] = datetime.now()
            request_kwargs["traffic_model"] = "best_guess"

        routes = gmaps.directions(
            **request_kwargs
        )

        if not routes:
            print("Google Directions unavailable: no routes returned")
            return None

        route = routes[0]
        leg = route["legs"][0]
        distance_km = float(leg["distance"]["value"]) / 1000.0
        duration_sec = leg.get("duration_in_traffic", leg["duration"])["value"]
        duration_min = max(3, int(round(duration_sec / 60.0)))
        steps = leg.get("steps", [])
        elevation_gain = None
        elevation_source = "Estimated"
        if include_elevation:
            path_points = _sample_path_for_elevation(start_lat, start_lon, end_lat, end_lon, steps)
            elevation_gain, elevation_source = get_google_elevation_gain(path_points)
        maneuvers = [str(step.get("maneuver", "")).lower() for step in steps]
        turn_maneuvers = [
            m for m in maneuvers
            if any(token in m for token in ["turn", "roundabout", "ramp", "merge", "fork", "uturn"])
        ]
        avg_lat = (start_lat + end_lat) / 2
        avg_lon = (start_lon + end_lon) / 2
        signal_estimate = None
        signal_source = "Not applicable"
        if travel_mode == "driving":
            signal_estimate = estimate_signals_from_google_route(
                distance_km=distance_km,
                step_count=len(steps),
                turn_count=len(turn_maneuvers),
                avg_lat=avg_lat,
                avg_lon=avg_lon,
            )
            signal_source = "Google route intersection proxy"

        print(f"Google {travel_mode.title()} Directions: {distance_km:.1f}km, {duration_min} min")
        return {
            "distance_km": round(distance_km, 1),
            "duration_min": duration_min,
            "step_count": len(steps),
            "turn_count": len(turn_maneuvers),
            "signal_estimate": signal_estimate,
            "signal_source": signal_source,
            "provider": f"Google Directions ({travel_mode})",
            "summary": route.get("summary", ""),
            "travel_mode": travel_mode,
            "step_based_signal_ready": len(steps) > 0,
            "elevation_gain": elevation_gain,
            "elevation_source": elevation_source,
            "origin_place_id": origin_place_id,
            "destination_place_id": destination_place_id,
            "resolved_start_address": leg.get("start_address", (origin_meta or {}).get("formatted_address", "")),
            "resolved_end_address": leg.get("end_address", (destination_meta or {}).get("formatted_address", "")),
        }
    except Exception as e:
        print(f"Google Directions error: {type(e).__name__}: {e}")
        return None


def calculate_route_distance(start_lat, start_lon, end_lat, end_lon) -> float:
    """Google Directions first, GraphHopper fallback, straight-line as last resort."""
    google_route = get_google_route_metrics_client(
        start_lat, start_lon, end_lat, end_lon,
        origin_meta=None,
        destination_meta=None,
    )
    if google_route:
        return google_route["distance_km"]

    try:
        url = (
            f"https://graphhopper.com/api/1/route?"
            f"point={start_lat},{start_lon}&point={end_lat},{end_lon}&"
            f"vehicle=car&calc_points=false&key=demo"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("paths"):
                distance_m = data["paths"][0]["distance"]
                dist_km = distance_m / 1000.0
                print(f"GraphHopper: {dist_km:.1f}km")
                return round(dist_km, 1)
    except Exception as e:
        print(f"GraphHopper error: {e}")

    straight_km = calculate_straight_distance(start_lat, start_lon, end_lat, end_lon)
    driving_estimate = straight_km * 2.5
    print(f"Straight-line: {straight_km:.1f}km -> Estimated driving: {driving_estimate:.1f}km (fallback)")
    return round(driving_estimate, 1)


# ---------------- Teacher (rule-based) ---------------- #

def calculate_route_risk_factors_teacher(distance_km: float, start_lat, start_lon, end_lat, end_lon,
                                         step_count: int = 0, google_duration_min: int | None = None,
                                         google_signal_estimate: int | None = None,
                                         signal_source: str = "Estimated",
                                         google_elevation_gain: int | None = None,
                                         elevation_source: str = "Estimated",
                                         weather_score_input: float = 0.0,
                                         time_factor_override: float | None = None,
                                         travel_mode: str = "driving") -> dict:
    signals = get_signals_safe(
        distance_km, start_lat, start_lon, end_lat, end_lon,
        step_count=step_count,
        google_signal_estimate=google_signal_estimate,
    )

    avg_lat = (start_lat + end_lat) / 2
    avg_lon = (start_lon + end_lon) / 2
    density_multiplier = 1.0

    if 114.15 < avg_lon < 114.20 and 22.27 < avg_lat < 22.33:
        density_multiplier = 1.3
    elif 113.90 < avg_lon < 114.05:
        density_multiplier = 0.7

    construction_detail = build_construction_detail(
        distance_km, density_multiplier, avg_lat, avg_lon,
        start_lat, start_lon, end_lat, end_lon
    )
    construction = int(construction_detail["estimated_work_zones"])
    elevation = int(distance_km * 8 * density_multiplier)
    if google_elevation_gain is not None:
        elevation = int(google_elevation_gain)

    effective_time_factor = float(time_factor_override) if time_factor_override is not None else float(time_factor)

    base_complexity = 2.8 if travel_mode == "driving" else 2.1
    distance_factor = min(2.2, distance_km * 0.22)
    signals_per_km = signals / max(1.0, distance_km)
    short_route_smoothing = min(1.0, distance_km / 3.0)
    signal_factor = min(2.4, signals_per_km * 0.65 * (0.55 + 0.45 * short_route_smoothing))
    density_bonus = (density_multiplier - 1.0) * 1.0
    time_bonus = (effective_time_factor - 1.0) * 2.0 if effective_time_factor >= 1.0 else (effective_time_factor - 1.0) * 1.0

    road_complexity = min(10.0, base_complexity + distance_factor + signal_factor + density_bonus + time_bonus)
    weather_component = weather_score_to_component(weather_score_input)
    construction_component = min(1.6, construction * 0.8)
    elevation_component = min(1.2, elevation / 60.0)
    mode_adjustment = -0.45 if travel_mode == "walking" else 0.0

    baseline_risk_score = min(
        10.0,
        max(
            0.0,
            (road_complexity * 0.62)
            + weather_component
            + construction_component
            + elevation_component
            + mode_adjustment
        ),
    )
    baseline_risk_score = calibrate_risk_score(
        baseline_risk_score,
        distance_km=distance_km,
        weather_score=weather_score_input,
        travel_mode=travel_mode,
    )
    risk_score = baseline_risk_score
    risk_level = "HIGH" if risk_score > 7 else "MEDIUM" if risk_score > 4 else "LOW"

    base_speed_kmh = 32.0
    if risk_level == "HIGH":
        base_speed_kmh = 26.0
    elif risk_level == "MEDIUM":
        base_speed_kmh = 29.0

    base_time_min = (distance_km / base_speed_kmh) * 60
    signal_delay_min = signals * 0.2
    construction_delay_min = construction * 0.5
    elevation_delay_min = elevation / 60.0

    est_time = max(3, int(round(base_time_min + signal_delay_min + construction_delay_min + elevation_delay_min)))
    if google_duration_min is not None:
        est_time = max(3, int(google_duration_min))

    return {
        "distance_km": round(distance_km, 1),
        "signals": int(signals),
        "signals_per_km": round(signals_per_km, 2),
        "construction": int(construction),
        "construction_detail": construction_detail,
        "elevation": int(elevation),
        "elevation_source": elevation_source,
        "road_complexity": round(road_complexity, 2),
        "road_breakdown": {
            "base": round(base_complexity, 2),
            "distance": round(distance_factor, 2),
            "signals_km": round(signal_factor, 2),
            "density": round(density_bonus, 2),
            "time": round(time_bonus, 2),
        },
        "risk_components": {
            "complexity_component": round(road_complexity * 0.62, 2),
            "weather_component": round(weather_component, 2),
            "construction_component": round(construction_component, 2),
            "elevation_component": round(elevation_component, 2),
            "mode_adjustment": round(mode_adjustment, 2),
        },
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "est_time": int(est_time),
        "density_multiplier": float(density_multiplier),
        "time_factor": float(effective_time_factor),
        "weather_score_input": float(weather_score_input),
        "travel_mode": travel_mode,
        "signal_source": signal_source,
    }


# ---------------- XGBoost features + training ---------------- #

def features_from_teacher_output(distance_km: float, teacher: dict) -> np.ndarray:
    feats = [
        float(distance_km),
        float(teacher["signals"]),
        float(teacher["signals_per_km"]),
        float(teacher["construction"]),
        float(teacher["elevation"]),
        float(teacher["density_multiplier"]),
        float(teacher["time_factor"]),
        float(teacher["road_breakdown"]["base"]),
        float(teacher["road_breakdown"]["distance"]),
        float(teacher["road_breakdown"]["signals_km"]),
        float(teacher["road_breakdown"]["density"]),
        float(teacher["road_breakdown"]["time"]),
        float(teacher["weather_score_input"]),
        float(teacher["risk_components"]["weather_component"]),
        float(teacher["risk_components"]["construction_component"]),
        float(teacher["risk_components"]["elevation_component"]),
        1.0 if teacher.get("travel_mode") == "driving" else 0.0,
    ]
    return np.array(feats, dtype=np.float32)


def train_xgb_models_synthetic(n_samples: int = DEFAULT_TRAINING_SAMPLES, seed: int = 42):
    rng = np.random.default_rng(seed)

    # HK-ish bbox
    lat_min, lat_max = 22.20, 22.55
    lon_min, lon_max = 113.85, 114.35

    X = []
    y_score = []
    y_time = []

    for _ in range(n_samples):
        start_lat = float(rng.uniform(lat_min, lat_max))
        start_lon = float(rng.uniform(lon_min, lon_max))
        end_lat = float(rng.uniform(lat_min, lat_max))
        end_lon = float(rng.uniform(lon_min, lon_max))
        sampled_hour = int(rng.integers(0, 24))
        sampled_time_factor = get_time_factor(sampled_hour)
        sampled_weather_score = float(rng.uniform(0.0, 8.5))
        sampled_mode = "driving" if rng.uniform() > 0.2 else "walking"

        straight_km = calculate_straight_distance(start_lat, start_lon, end_lat, end_lon)
        distance_km = max(1.0, float(straight_km * rng.uniform(1.8, 2.8)))

        teacher = calculate_route_risk_factors_teacher(
            distance_km, start_lat, start_lon, end_lat, end_lon,
            weather_score_input=sampled_weather_score,
            time_factor_override=sampled_time_factor,
            travel_mode=sampled_mode,
        )
        X.append(features_from_teacher_output(distance_km, teacher))
        y_score.append(float(teacher["risk_score"]))
        y_time.append(float(teacher["est_time"]))

    X = np.vstack(X)
    y_score = np.array(y_score, dtype=np.float32)
    y_time = np.array(y_time, dtype=np.float32)

    X_train, X_test, ys_train, ys_test, yt_train, yt_test = train_test_split(
        X, y_score, y_time, test_size=0.2, random_state=seed
    )

    model_score = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    model_score.fit(X_train, ys_train)

    model_time = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed + 1,
        n_jobs=-1,
    )
    model_time.fit(X_train, yt_train)

    pred_s = model_score.predict(X_test)
    pred_t = model_time.predict(X_test)
    mae_s = mean_absolute_error(ys_test, pred_s)
    mae_t = mean_absolute_error(yt_test, pred_t)

    print("\n=== XGBoost Training Report (synthetic mimic) ===")
    print(f"Risk_score MAE: {mae_s:.3f}")
    print(f"Est_time   MAE: {mae_t:.3f} min")

    model_score.save_model(MODEL_SCORE_PATH)
    model_time.save_model(MODEL_TIME_PATH)
    joblib.dump(
        {
            "feature_count": int(X.shape[1]),
            "feature_set_version": FEATURE_SET_VERSION,
            "model_version": MODEL_VERSION,
            "training_source": "Synthetic teacher-labelled route samples",
            "risk_score_mae": float(mae_s),
            "est_time_mae": float(mae_t),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "total_samples": int(X.shape[0]),
        },
        META_PATH,
    )

    return model_score, model_time


def load_xgb_models():
    if not (os.path.exists(MODEL_SCORE_PATH) and os.path.exists(MODEL_TIME_PATH) and os.path.exists(META_PATH)):
        return None, None, None

    meta = joblib.load(META_PATH)
    meta.setdefault("feature_set_version", "legacy")
    meta.setdefault("model_version", "legacy")
    model_score = XGBRegressor()
    model_time = XGBRegressor()
    model_score.load_model(MODEL_SCORE_PATH)
    model_time.load_model(MODEL_TIME_PATH)
    return model_score, model_time, meta


def risk_level_from_score(score: float) -> str:
    return "HIGH" if score > 7 else "MEDIUM" if score > 4 else "LOW"


def weather_score_to_component(weather_score: float) -> float:
    return min(1.8, max(0.0, weather_score) * 0.18)


def calibrate_risk_score(score: float, distance_km: float, weather_score: float, travel_mode: str) -> float:
    calibrated = float(score)
    if distance_km <= 2.0:
        calibrated -= 0.25
    if travel_mode == "walking":
        calibrated -= 0.35
    if weather_score <= 5.5:
        calibrated -= 0.10
    return float(np.clip(calibrated, 0.0, 10.0))


def get_signal_display_config(travel_mode: str) -> tuple[str, str]:
    if travel_mode == "walking":
        return "Crossings / Intersections", "Route-step crossing proxy"
    return "Signals / Intersections", "Google route intersection proxy"


def build_overall_recommendation(route_risk: dict, routing_confidence: dict, travel_mode: str) -> dict:
    risk_level = route_risk.get("risk_level", "MEDIUM")
    confidence_level = routing_confidence.get("level", "MEDIUM")
    complexity = float(route_risk.get("road_complexity", 5.0))
    weather_component = float(route_risk.get("risk_components", {}).get("weather_component", 0.0))

    if risk_level == "HIGH":
        action = "Reconsider route or delay trip"
    elif risk_level == "MEDIUM":
        action = "Proceed with caution"
    else:
        action = "Proceed"

    reasons = []
    if complexity >= 7.0:
        reasons.append("dense urban route complexity")
    if weather_component >= 0.8:
        reasons.append("weather impact")
    if confidence_level == "LOW":
        reasons.append("low routing confidence")
    if travel_mode == "walking":
        reasons.append("pedestrian crossing exposure")
    elif route_risk.get("signals", 0) >= 10:
        reasons.append("high intersection count")

    if not reasons:
        reasons.append("stable route conditions")

    return {
        "action": action,
        "reason": ", ".join(reasons),
    }


def build_ai_risk_explanation(route_risk: dict, routing_confidence: dict, weather_risk: dict,
                              overall_recommendation: dict, travel_mode: str,
                              distance_km: float, route_duration_min: int | None) -> dict:
    risk_level = format_display_value(route_risk.get("risk_level", "MEDIUM"))
    confidence = format_display_value(routing_confidence.get("level", "MEDIUM"))
    complexity = float(route_risk.get("road_complexity", 0.0))
    signals = int(route_risk.get("signals", 0))
    weather_reason = str(weather_risk.get("reason", "Stable weather conditions"))
    action = overall_recommendation.get("action", "Proceed with caution")

    dominant_factors = []
    if complexity >= 7.0:
        dominant_factors.append("dense route complexity")
    if signals >= 10:
        dominant_factors.append("a high number of intersections")
    if "rain" in weather_reason.lower() or "storm" in weather_reason.lower() or "fog" in weather_reason.lower():
        dominant_factors.append("adverse weather conditions")
    if confidence == "Low":
        dominant_factors.append("lower routing confidence")
    if travel_mode == "walking":
        dominant_factors.append("pedestrian crossing exposure")

    if not dominant_factors:
        dominant_factors.append("generally stable route conditions")

    primary_reason = ", ".join(dominant_factors[:3])
    duration_text = f"{route_duration_min} minutes" if route_duration_min is not None else "the estimated travel duration"
    mode_text = "driving" if travel_mode == "driving" else "walking"

    summary = (
        f"The analysed {mode_text} route is classified as {risk_level} risk within the current prototype framework. "
        f"This assessment is primarily influenced by {primary_reason}. "
        f"The evaluated journey length is {distance_km:.1f} km, with an expected travel duration of approximately {duration_text}."
    )
    caution = (
        f"From an operational perspective, the recommended action is to {action.lower()}. "
        f"The routing confidence for this case is {confidence}, while the weather layer indicates {weather_reason.lower()}."
    )
    return {
        "summary": summary,
        "caution": caution,
    }


def build_compare_ai_recommendation(driving_result: dict, walking_result: dict) -> dict:
    driving_score = float(driving_result.get("risk_score", 10.0) or 10.0)
    walking_score = float(walking_result.get("risk_score", 10.0) or 10.0)
    driving_time = float(driving_result.get("route_time_min", 999) or 999)
    walking_time = float(walking_result.get("route_time_min", 999) or 999)

    if driving_score + 0.6 < walking_score:
        preferred_mode = "Driving"
        reason = "Driving offers a clearly lower risk score while keeping travel time short."
    elif walking_score + 0.6 < driving_score:
        preferred_mode = "Walking"
        reason = "Walking provides a meaningfully safer route despite the longer journey time."
    elif driving_time + 4 < walking_time:
        preferred_mode = "Driving"
        reason = "Risk scores are close, but driving is materially faster under the current conditions."
    else:
        preferred_mode = "Walking"
        reason = "Risk scores are similar, and walking avoids extra vehicle exposure."

    return {
        "preferred_mode": preferred_mode,
        "reason": reason,
    }


def calculate_route_risk_factors_xgb(distance_km: float, start_lat, start_lon, end_lat, end_lon,
                                     model_score: XGBRegressor, model_time: XGBRegressor, meta: dict,
                                     step_count: int = 0, google_duration_min: int | None = None,
                                     google_signal_estimate: int | None = None,
                                     signal_source: str = "Estimated",
                                     google_elevation_gain: int | None = None,
                                     elevation_source: str = "Estimated",
                                     weather_score_input: float = 0.0,
                                     time_factor_override: float | None = None,
                                     travel_mode: str = "driving") -> dict:
    teacher = calculate_route_risk_factors_teacher(
        distance_km, start_lat, start_lon, end_lat, end_lon,
        step_count=step_count,
        google_duration_min=google_duration_min,
        google_signal_estimate=google_signal_estimate,
        signal_source=signal_source,
        google_elevation_gain=google_elevation_gain,
        elevation_source=elevation_source,
        weather_score_input=weather_score_input,
        time_factor_override=time_factor_override,
        travel_mode=travel_mode,
    )
    feats = features_from_teacher_output(distance_km, teacher).reshape(1, -1)

    if meta and feats.shape[1] != meta.get("feature_count"):
        raise ValueError(f"Feature mismatch: got {feats.shape[1]}, expected {meta.get('feature_count')}")

    raw_pred_score = float(model_score.predict(feats)[0])
    pred_time = float(model_time.predict(feats)[0])

    raw_pred_score = float(np.clip(raw_pred_score, 0.0, 10.0))
    baseline_score = float(teacher["risk_score"])
    blended_score = float(np.clip((raw_pred_score * XGB_WEIGHT) + (baseline_score * BASELINE_WEIGHT), 0.0, 10.0))
    blended_score = calibrate_risk_score(
        blended_score,
        distance_km=distance_km,
        weather_score=weather_score_input,
        travel_mode=travel_mode,
    )

    pred_time = int(max(3, round(pred_time)))
    if google_duration_min is not None:
        pred_time = max(3, int(google_duration_min))

    out = dict(teacher)
    out["risk_score"] = round(blended_score, 2)
    out["risk_level"] = risk_level_from_score(blended_score)
    out["est_time"] = pred_time
    out["model_used"] = "XGBoost"
    out["xgb_raw_score"] = round(raw_pred_score, 2)
    out["baseline_risk_score"] = round(baseline_score, 2)
    out["risk_formula"] = f"final = {XGB_WEIGHT:.2f} * XGBoost + {BASELINE_WEIGHT:.2f} * baseline"
    out["risk_formula_weights"] = {"xgboost": XGB_WEIGHT, "baseline": BASELINE_WEIGHT}
    return out


def calculate_route_risk_factors_fast(distance_km: float, start_lat, start_lon, end_lat, end_lon,
                                      step_count: int = 0, google_duration_min: int | None = None,
                                      google_signal_estimate: int | None = None,
                                      signal_source: str = "Estimated",
                                      google_elevation_gain: int | None = None,
                                      elevation_source: str = "Estimated",
                                      weather_score_input: float = 5.0,
                                      time_factor_override: float | None = None,
                                      travel_mode: str = "driving") -> dict:
    signals = get_signals_safe(
        distance_km, start_lat, start_lon, end_lat, end_lon,
        step_count=step_count,
        google_signal_estimate=google_signal_estimate,
    )

    avg_lat = (start_lat + end_lat) / 2
    avg_lon = (start_lon + end_lon) / 2
    density_multiplier = 1.0
    if 114.15 < avg_lon < 114.20 and 22.27 < avg_lat < 22.33:
        density_multiplier = 1.3
    elif 113.90 < avg_lon < 114.05:
        density_multiplier = 0.7

    construction_detail = build_construction_detail_heuristic(distance_km, density_multiplier, avg_lat, avg_lon)
    construction = int(construction_detail["estimated_work_zones"])
    elevation = int(distance_km * 8 * density_multiplier)
    if google_elevation_gain is not None:
        elevation = int(google_elevation_gain)

    effective_time_factor = float(time_factor_override) if time_factor_override is not None else float(time_factor)
    base_complexity = 2.8 if travel_mode == "driving" else 2.1
    distance_factor = min(2.2, distance_km * 0.22)
    signals_per_km = signals / max(1.0, distance_km)
    short_route_smoothing = min(1.0, distance_km / 3.0)
    signal_factor = min(2.4, signals_per_km * 0.65 * (0.55 + 0.45 * short_route_smoothing))
    density_bonus = (density_multiplier - 1.0) * 1.0
    time_bonus = (effective_time_factor - 1.0) * 2.0 if effective_time_factor >= 1.0 else (effective_time_factor - 1.0) * 1.0

    road_complexity = min(10.0, base_complexity + distance_factor + signal_factor + density_bonus + time_bonus)
    weather_component = weather_score_to_component(weather_score_input)
    construction_component = min(1.6, construction * 0.8)
    elevation_component = min(1.2, elevation / 60.0)
    mode_adjustment = -0.45 if travel_mode == "walking" else 0.0

    risk_score = min(
        10.0,
        max(
            0.0,
            (road_complexity * 0.62)
            + weather_component
            + construction_component
            + elevation_component
            + mode_adjustment
        ),
    )
    risk_score = calibrate_risk_score(
        risk_score,
        distance_km=distance_km,
        weather_score=weather_score_input,
        travel_mode=travel_mode,
    )
    risk_level = risk_level_from_score(risk_score)
    est_time = max(3, int(round((distance_km / 28.0) * 60 + (signals * 0.18) + (construction * 0.4) + (elevation / 80.0))))
    if google_duration_min is not None:
        est_time = max(3, int(google_duration_min))

    return {
        "distance_km": round(distance_km, 1),
        "signals": int(signals),
        "signals_per_km": round(signals_per_km, 2),
        "construction": int(construction),
        "construction_detail": construction_detail,
        "elevation_gain": int(elevation),
        "elevation_source": elevation_source,
        "time_factor": round(effective_time_factor, 2),
        "road_complexity": round(road_complexity, 2),
        "road_breakdown": {
            "base": round(base_complexity, 2),
            "distance": round(distance_factor, 2),
            "signals": round(signal_factor, 2),
            "density": round(density_bonus, 2),
            "time": round(time_bonus, 2),
        },
        "est_time": est_time,
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "weather_impact": round(weather_score_input, 2),
        "weather_reason": "Fast quote mode uses baseline weather assumption.",
        "signal_source": signal_source,
        "model_used": "Fast Quote Engine",
        "risk_components": {
            "road_complexity_component": round(road_complexity * 0.62, 2),
            "weather_component": round(weather_component, 2),
            "construction_component": round(construction_component, 2),
            "elevation_component": round(elevation_component, 2),
            "mode_adjustment": round(mode_adjustment, 2),
        },
        "baseline_risk_score": round(risk_score, 2),
        "xgb_raw_score": None,
        "risk_formula": "Fast quote rule-based scoring",
        "risk_formula_weights": {"xgboost": 0.0, "baseline": 1.0},
    }


# ---------------- Main pipeline ---------------- #

def get_route_conditions(
    start_location: str,
    dest_location: str,
    travel_mode: str = "driving",
    interactive: bool = True,
):
    # 1) Geocode + confirm (Google-first)
    start_lat, start_lon, start_meta = geocode_confirmed(start_location, interactive=interactive)
    if start_lat is None:
        print(f"WARNING: Geocode failed for {start_location} -> using Central HK fallback")
        start_lat, start_lon = 22.3027, 114.1772
        start_meta = {"provider": "Fallback"}

    dest_lat, dest_lon, dest_meta = geocode_confirmed(dest_location, interactive=interactive)
    if dest_lat is None:
        print(f"WARNING: Geocode failed for {dest_location} -> using Central HK fallback")
        dest_lat, dest_lon = 22.3027, 114.1772
        dest_meta = {"provider": "Fallback"}

    if SHOW_DEBUG:
        print(
            f"DEBUG start geocode: {start_meta.get('provider')} {start_meta.get('location_type', '')} score={start_meta.get('score', '')}")
        print(
            f"DEBUG dest  geocode: {dest_meta.get('provider')} {dest_meta.get('location_type', '')} score={dest_meta.get('score', '')}")
    # District labels
    start_district, start_district_source = get_hk_district_with_source(start_lat, start_lon, start_meta)
    dest_district, dest_district_source = get_hk_district_with_source(dest_lat, dest_lon, dest_meta)
    start_resolved_address = start_meta.get("formatted_address", "")
    dest_resolved_address = dest_meta.get("formatted_address", "")
    start_place_id = start_meta.get("place_id", "")
    dest_place_id = dest_meta.get("place_id", "")

    # 2) Distance + duration
    google_route = get_google_route_metrics_client(
        start_lat, start_lon, dest_lat, dest_lon,
        travel_mode=travel_mode,
        origin_meta=start_meta,
        destination_meta=dest_meta,
    )
    if google_route:
        distance_km = google_route["distance_km"]
        route_duration_min = google_route["duration_min"]
        route_step_count = google_route["step_count"]
        route_signal_estimate = google_route.get("signal_estimate")
        route_elevation_gain = google_route.get("elevation_gain")
        route_elevation_source = google_route.get("elevation_source", "Estimated")
        route_signal_source = google_route.get("signal_source", "Google route intersection proxy")
        if route_signal_estimate is None and google_route.get("step_based_signal_ready"):
            route_signal_source = "Google route step estimate"
        route_provider = google_route["provider"]
        if google_route.get("resolved_start_address"):
            start_resolved_address = google_route.get("resolved_start_address") or start_resolved_address
        if google_route.get("resolved_end_address"):
            dest_resolved_address = google_route.get("resolved_end_address") or dest_resolved_address
        start_place_id = google_route.get("origin_place_id") or start_place_id
        dest_place_id = google_route.get("destination_place_id") or dest_place_id
    else:
        if travel_mode == "walking":
            straight_km = calculate_straight_distance(start_lat, start_lon, dest_lat, dest_lon)
            distance_km = round(straight_km * 1.2, 1)
            route_duration_min = max(3, int(round((distance_km / 4.8) * 60)))
        else:
            distance_km = calculate_route_distance(start_lat, start_lon, dest_lat, dest_lon)
            route_duration_min = None
        route_step_count = 0
        route_signal_estimate = None
        route_elevation_gain = None
        route_elevation_source = "Estimated"
        route_signal_source = "Baseline estimate"
        route_provider = "GraphHopper/estimate"

    routing_confidence = build_routing_confidence(start_meta, dest_meta, google_route if google_route else None)

    if travel_mode == "driving":
        print_section_header("Weather Summary")
        print_weather_summary(start_lat, start_lon, dest_lat, dest_lon)

    print_section_header("Input")
    print_two_column_panel(
        "Route Input",
        [("Route Query", start_location.title())],
        "Destination Input",
        [("Destination Query", dest_location.title())],
    )

    print_section_header("Google Resolution")
    route_resolution_rows = [
        ("Resolved Address", format_display_value(start_resolved_address or "Not available")),
    ]
    if start_place_id:
        route_resolution_rows.append(("Place ID", start_place_id))
    route_resolution_rows.append(("District Fallback", format_display_value(start_district)))
    route_resolution_rows.append(("District Source", format_display_value(start_district_source)))

    destination_resolution_rows = [
        ("Resolved Address", format_display_value(dest_resolved_address or "Not available")),
    ]
    if dest_place_id:
        destination_resolution_rows.append(("Place ID", dest_place_id))
    destination_resolution_rows.append(("District Fallback", format_display_value(dest_district)))
    destination_resolution_rows.append(("District Source", format_display_value(dest_district_source)))

    print_two_column_panel(
        "Route Resolution",
        route_resolution_rows,
        "Destination Resolution",
        destination_resolution_rows,
    )

    print_section_header("Routing Confidence")
    print_report_line("Routing Source", routing_confidence.get('source', route_provider))
    print_report_line("Origin Resolution", format_display_value(routing_confidence.get('origin_resolution', 'Unknown')))
    print_report_line("Destination Resolution", format_display_value(routing_confidence.get('destination_resolution', 'Unknown')))
    print()
    print_report_line("POI Priority Applied", routing_confidence.get('poi_priority_applied', 'No'))
    print_report_line("Confidence Level", format_display_value(routing_confidence.get('level', 'LOW')))

    print_section_header("Routing Target")
    routing_left_rows = [
        ("Mode", travel_mode),
        ("Distance", f"{distance_km:.1f}km"),
    ]
    routing_right_rows = [
        ("Start Coordinates", f"({start_lat:.4f}, {start_lon:.4f})"),
        ("End Coordinates", f"({dest_lat:.4f}, {dest_lon:.4f})"),
    ]
    if route_duration_min is not None:
        routing_left_rows.append(("Route Time", f"{route_duration_min} min"))

    print_two_column_panel(
        "Route Overview",
        routing_left_rows,
        "Resolved Coordinates",
        routing_right_rows,
    )

    # 3) Weather risk (kept)
    w_start = get_current_weather(start_lat, start_lon)
    w_dest = get_current_weather(dest_lat, dest_lon)
    weather_source, weather_confidence = get_weather_data_quality(w_start, w_dest)
    if w_start and w_dest:
        combined_weather_risk = calculate_route_risk(
            w_start, w_dest,
            {"complexity_score": 5.0},
            {"complexity_score": 5.0},
        )
    else:
        combined_weather_risk = {"score": 5.0, "level": "MEDIUM", "reason": "No weather data"}

    # 4) Load or train model (OSM disabled during training)
    model_score, model_time, meta = load_xgb_models()
    model_status = "Loaded existing model"
    if model_score is None:
        print("\nNo saved XGBoost model found. Training a mimic model now (one-time)...")
        global DISABLE_OSM
        DISABLE_OSM = True
        model_score, model_time = train_xgb_models_synthetic(n_samples=DEFAULT_TRAINING_SAMPLES, seed=42)
        DISABLE_OSM = False
        meta = joblib.load(META_PATH)
        model_status = "Trained new model (no saved model found)"

    # 5) Predict (fallback to teacher)
    try:
        route_risk = calculate_route_risk_factors_xgb(
            distance_km, start_lat, start_lon, dest_lat, dest_lon,
            model_score, model_time, meta,
            step_count=route_step_count,
            google_duration_min=route_duration_min,
            google_signal_estimate=route_signal_estimate,
            signal_source=route_signal_source,
            google_elevation_gain=route_elevation_gain,
            elevation_source=route_elevation_source,
            weather_score_input=combined_weather_risk["score"],
            time_factor_override=time_factor,
            travel_mode=travel_mode,
        )
    except Exception as e:
        if "Feature mismatch" in str(e):
            print("\nXGBoost model metadata is outdated. Retraining model to match the latest risk formula...")
            DISABLE_OSM = True
            model_score, model_time = train_xgb_models_synthetic(n_samples=DEFAULT_TRAINING_SAMPLES, seed=42)
            DISABLE_OSM = False
            meta = joblib.load(META_PATH)
            model_status = "Retrained model (feature set updated)"
            route_risk = calculate_route_risk_factors_xgb(
                distance_km, start_lat, start_lon, dest_lat, dest_lon,
                model_score, model_time, meta,
                step_count=route_step_count,
                google_duration_min=route_duration_min,
                google_signal_estimate=route_signal_estimate,
                signal_source=route_signal_source,
                google_elevation_gain=route_elevation_gain,
                elevation_source=route_elevation_source,
                weather_score_input=combined_weather_risk["score"],
                time_factor_override=time_factor,
                travel_mode=travel_mode,
            )
        else:
            print(f"\nXGBoost prediction failed ({e}). Falling back to teacher rules.")
            route_risk = calculate_route_risk_factors_teacher(
                distance_km, start_lat, start_lon, dest_lat, dest_lon,
                step_count=route_step_count,
                google_duration_min=route_duration_min,
                google_signal_estimate=route_signal_estimate,
                signal_source=route_signal_source,
                google_elevation_gain=route_elevation_gain,
                elevation_source=route_elevation_source,
                weather_score_input=combined_weather_risk["score"],
                time_factor_override=time_factor,
                travel_mode=travel_mode,
            )
            route_risk["model_used"] = "TeacherRules"
            model_status = "Teacher-rule fallback"

    signal_label, default_signal_source = get_signal_display_config(travel_mode)
    overall_recommendation = build_overall_recommendation(route_risk, routing_confidence, travel_mode)
    ai_risk_explanation = build_ai_risk_explanation(
        route_risk=route_risk,
        routing_confidence=routing_confidence,
        weather_risk=combined_weather_risk,
        overall_recommendation=overall_recommendation,
        travel_mode=travel_mode,
        distance_km=distance_km,
        route_duration_min=route_duration_min,
    )

    print_executive_summary_header("Executive Summary", leading_newline=False)
    print_centered_line(
        f"{format_display_value(route_risk['risk_level'])} Risk | "
        f"Score {route_risk['risk_score']:.2f} | "
        f"{overall_recommendation['action']}"
    )
    print()
    print_centered_line(
        f"{travel_mode.title()} route | {distance_km:.1f} km | "
        f"{route_duration_min or route_risk['est_time']} min | "
        f"Routing Confidence {format_display_value(routing_confidence.get('level', 'LOW'))}"
    )

    # Print summary
    print_section_header("Route Risk Factors")
    print_report_line("Model Used", format_display_value(route_risk.get('model_used', 'Unknown')))
    print_report_line(signal_label, f"{route_risk['signals']} ({route_risk['signals']/max(1.0, distance_km):.1f}/km)")
    print_report_line("Signal Data Source", route_risk.get('signal_source', default_signal_source))
    print()
    print_report_line("Construction Count", route_risk['construction'])
    construction_detail = route_risk.get("construction_detail", {})
    if construction_detail:
        print_report_line(
            "Construction Detail",
            f"zones={construction_detail.get('estimated_work_zones', 0)}, "
            f"affected_distance={construction_detail.get('estimated_affected_distance_km', 0):.2f}km, "
            f"density_factor={construction_detail.get('route_density_factor', 0):.2f}"
        )
        print_report_line("Construction Data Source", construction_detail.get('source', 'Estimated'))
        print_report_line("Construction Confidence", format_display_value(construction_detail.get('confidence', 'LOW')))
        print_report_line("Construction Area Context", format_display_value(construction_detail.get('area_type', 'Unknown')))
        matched_roadworks = construction_detail.get("matched_roadworks", [])
        for idx, work in enumerate(matched_roadworks[:3], start=1):
            print_report_line(
                f"Matched Work {idx}",
                f"{work.get('location', 'Unknown location')} | "
                f"status={work.get('status', 'Unknown')} | "
                f"lane={work.get('affected_lane', 'Unknown')} | "
                f"dist={work.get('distance_to_route_km', 0):.3f}km"
            )
    print()
    print_report_line("Elevation Gain", f"{route_risk['elevation']}m")
    print_report_line("Elevation Data Source", route_risk.get('elevation_source', 'Estimated'))
    elevation_confidence = "HIGH" if "Google Elevation API" in str(route_risk.get("elevation_source", "")) else "MEDIUM" if route_risk.get("elevation_source") else "LOW"
    print_report_line("Elevation Confidence", format_display_value(elevation_confidence))
    print_report_line("Time Factor", f"{now.strftime('%H:%M')} ({get_time_band(now.hour)}) x{time_factor:.2f}")
    print_report_line("Road Complexity", f"{route_risk['road_complexity']}/10")
    breakdown = route_risk.get("road_breakdown", {})
    if breakdown:
        print_report_line(
            "Complexity Breakdown",
            f"base={breakdown.get('base', 0):.2f}, "
            f"distance={breakdown.get('distance', 0):.2f}, "
            f"signals={breakdown.get('signals_km', 0):.2f}, "
            f"density={breakdown.get('density', 0):.2f}, "
            f"time={breakdown.get('time', 0):.2f}"
        )
    print()
    print_report_line("Estimated Travel Time", f"{route_risk['est_time']} min")
    print_report_line("Weather Impact", f"{combined_weather_risk['score']:.1f}/10")
    print_report_line("Weather Explanation", combined_weather_risk['reason'])
    print_report_line("Weather Data Source", weather_source)
    print_report_line("Weather Confidence", format_display_value(weather_confidence))

    print_section_header("Overall Recommendation")
    print_report_line("Recommended Action", overall_recommendation['action'])
    print_report_line("Reason", overall_recommendation['reason'])

    print_section_header("AI Risk Explanation")
    print()
    print_report_line("Summary", ai_risk_explanation["summary"], label_width=12, value_width=86)
    print()
    print_report_line("Guidance", ai_risk_explanation["caution"], label_width=12, value_width=86)

    print_section_header("Risk Score Summary")
    print_report_line("Final Risk Score", route_risk['risk_score'])
    print_report_line("Risk Level", format_display_value(route_risk['risk_level']))
    print_report_line("Risk Score Scale", "0 (lowest) to 10 (highest)")
    print_report_line("Risk Level Thresholds", format_risk_threshold_text())

    print_section_header("Model Info")
    model_info_indent = 8
    print_report_line("Model Status", model_status, label_width=22, value_width=78, indent=model_info_indent)
    print_report_line("Model Version", meta.get('model_version', MODEL_VERSION) if meta else MODEL_VERSION, label_width=22, value_width=78, indent=model_info_indent)
    print_report_line("Feature Set Version", meta.get('feature_set_version', FEATURE_SET_VERSION) if meta else FEATURE_SET_VERSION, label_width=22, value_width=78, indent=model_info_indent)
    print()
    print_report_line("Training Source", format_display_value(meta.get('training_source', 'Not available') if meta else 'Not available'), label_width=22, value_width=78, indent=model_info_indent)
    print_report_line(
        "Interpretability Note",
        "We used a hybrid approach (XGBoost Model + rule-based baseline) for stability and interpretability in this early prototype. Full SHAP values can be added in the next iteration once more training data becomes available.",
        label_width=22,
        value_width=78,
        indent=model_info_indent,
    )
    if meta:
        print()
        if meta.get("risk_score_mae") is not None:
            print_report_line("Risk Score MAE", f"{meta.get('risk_score_mae'):.3f}", label_width=22, value_width=78, indent=model_info_indent)
        if meta.get("est_time_mae") is not None:
            print_report_line("Est. Time MAE", f"{meta.get('est_time_mae'):.3f} min", label_width=22, value_width=78, indent=model_info_indent)
        if meta.get("total_samples") is not None:
            print_report_line(
                "Training Samples",
                f"{meta.get('total_samples')} "
                f"(train={meta.get('train_samples', 'n/a')}, test={meta.get('test_samples', 'n/a')})",
                label_width=22,
                value_width=78,
                indent=model_info_indent,
            )
        if meta.get("feature_count") is not None:
            print_report_line("Feature Count", meta.get('feature_count'), label_width=22, value_width=78, indent=model_info_indent)

    print_section_header("Risk Score Calculation")
    if route_risk.get("model_used") == "XGBoost":
        print_report_line("Model Output (XGBoost Model)", f"{route_risk.get('xgb_raw_score', route_risk['risk_score']):.2f}")
        print_report_line("Baseline Rule Score", f"{route_risk.get('baseline_risk_score', route_risk['risk_score']):.2f}")
        print()
        print_report_line("Final Score Formula", route_risk.get('risk_formula', 'final = model score'))
        formula_weights = route_risk.get("risk_formula_weights", {})
        if formula_weights:
            print_report_line(
                "Hybrid Weights",
                f"XGBoost Model={formula_weights.get('xgboost', 0):.2f}, "
                f"Baseline={formula_weights.get('baseline', 0):.2f}"
            )
    else:
        print_report_line(
            f"Model Output ({format_display_value(route_risk.get('model_used', 'TeacherRules'))})",
            f"{route_risk['risk_score']:.2f}"
        )
        print_report_line("Baseline Rule Score", "same as model output")
        print()
        print_report_line("Final Score Formula", "final = baseline score")

    print()
    print_report_line("Final Risk Score", f"{route_risk['risk_score']:.2f}")

    components = route_risk.get("risk_components", {})
    if components:
        print()
        print("Baseline Contribution")
        print_report_line(
            "Explanation",
            "These are approximate contributions to the final score based on the hybrid rule-based explanation layer."
        )
        print()
        print_report_line("Road Complexity", f"{components.get('complexity_component', 0):.2f}")
        print_report_line("Weather", f"{components.get('weather_component', 0):.2f}")
        print_report_line("Construction", f"{components.get('construction_component', 0):.2f}")
        print_report_line("Elevation", f"{components.get('elevation_component', 0):.2f}")
        print_report_line("Mode Adjustment", f"{components.get('mode_adjustment', 0):.2f}")
        print()
        print_report_line("Note", "These values support interpretation of the hybrid baseline layer and are not SHAP values.")

    append_route_analysis_log({
        "timestamp_hkt": datetime.now(hkt_tz).isoformat(),
        "input_route_query": start_location,
        "input_destination_query": dest_location,
        "resolved_route_address": start_resolved_address or "Not available",
        "resolved_destination_address": dest_resolved_address or "Not available",
        "route_place_id": start_place_id or "",
        "destination_place_id": dest_place_id or "",
        "route_district_fallback": start_district,
        "route_district_source": start_district_source,
        "destination_district_fallback": dest_district,
        "destination_district_source": dest_district_source,
        "mode": travel_mode,
        "distance_km": round(distance_km, 2),
        "route_time_min": route_duration_min,
        "routing_source": routing_confidence.get("source", route_provider),
        "routing_confidence": routing_confidence.get("level", "LOW"),
        "model_used": route_risk.get("model_used", "Unknown"),
        "model_status": model_status,
        "risk_score": route_risk.get("risk_score"),
        "risk_level": route_risk.get("risk_level"),
        "overall_recommendation": overall_recommendation.get("action"),
        "ai_risk_explanation": ai_risk_explanation.get("summary"),
    })
    return {
        "mode": travel_mode,
        "distance_km": round(distance_km, 2),
        "route_time_min": route_duration_min or route_risk.get("est_time"),
        "risk_score": route_risk.get("risk_score"),
        "risk_level": route_risk.get("risk_level"),
        "road_complexity": route_risk.get("road_complexity"),
        "routing_confidence": routing_confidence.get("level", "LOW"),
        "recommendation": overall_recommendation.get("action"),
        "ai_explanation": ai_risk_explanation.get("summary"),
    }


def get_route_quote_summary(start_location: str, dest_location: str, travel_mode: str = "driving") -> dict:
    start_lat, start_lon, start_meta = geocode_confirmed(start_location, interactive=False)
    if start_lat is None:
        start_lat, start_lon = 22.3027, 114.1772
        start_meta = {"provider": "Fallback"}

    dest_lat, dest_lon, dest_meta = geocode_confirmed(dest_location, interactive=False)
    if dest_lat is None:
        dest_lat, dest_lon = 22.3027, 114.1772
        dest_meta = {"provider": "Fallback"}

    google_route = get_google_route_metrics_client(
        start_lat,
        start_lon,
        dest_lat,
        dest_lon,
        travel_mode=travel_mode,
        origin_meta=start_meta,
        destination_meta=dest_meta,
        include_elevation=True,
    )

    if google_route:
        distance_km = google_route["distance_km"]
        route_duration_min = google_route["duration_min"]
        route_step_count = google_route["step_count"]
        route_signal_estimate = google_route.get("signal_estimate")
        route_signal_source = google_route.get("signal_source", "Google route intersection proxy")
        route_elevation_gain = google_route.get("elevation_gain")
        route_elevation_source = google_route.get("elevation_source", "Estimated")
        route_provider = google_route["provider"]
    else:
        if travel_mode == "walking":
            straight_km = calculate_straight_distance(start_lat, start_lon, dest_lat, dest_lon)
            distance_km = round(straight_km * 1.2, 1)
            route_duration_min = max(3, int(round((distance_km / 4.8) * 60)))
        else:
            distance_km = calculate_route_distance(start_lat, start_lon, dest_lat, dest_lon)
            route_duration_min = max(3, int(round((distance_km / 28.0) * 60)))
        route_step_count = 0
        route_signal_estimate = None
        route_signal_source = "Baseline estimate"
        route_elevation_gain = None
        route_elevation_source = "Estimated"
        route_provider = "GraphHopper/estimate"

    routing_confidence = build_routing_confidence(start_meta, dest_meta, google_route if google_route else None)
    w_start = get_current_weather(start_lat, start_lon)
    w_dest = get_current_weather(dest_lat, dest_lon)
    weather_source, weather_confidence = get_weather_data_quality(w_start, w_dest)
    if w_start and w_dest:
        combined_weather_risk = calculate_route_risk(
            w_start,
            w_dest,
            {"complexity_score": 5.0},
            {"complexity_score": 5.0},
        )
    else:
        combined_weather_risk = {"score": 5.0, "level": "MEDIUM", "reason": "No weather data"}

    model_score, model_time, meta = load_xgb_models()
    model_status = "Loaded existing model"
    if model_score is not None:
        try:
            route_risk = calculate_route_risk_factors_xgb(
                distance_km,
                start_lat,
                start_lon,
                dest_lat,
                dest_lon,
                model_score,
                model_time,
                meta,
                step_count=route_step_count,
                google_duration_min=route_duration_min,
                google_signal_estimate=route_signal_estimate,
                signal_source=route_signal_source,
                google_elevation_gain=route_elevation_gain,
                elevation_source=route_elevation_source,
                weather_score_input=combined_weather_risk["score"],
                time_factor_override=time_factor,
                travel_mode=travel_mode,
            )
        except Exception:
            route_risk = calculate_route_risk_factors_fast(
                distance_km,
                start_lat,
                start_lon,
                dest_lat,
                dest_lon,
                step_count=route_step_count,
                google_duration_min=route_duration_min,
                google_signal_estimate=route_signal_estimate,
                signal_source=route_signal_source,
                google_elevation_gain=route_elevation_gain,
                elevation_source=route_elevation_source,
                weather_score_input=combined_weather_risk["score"],
                time_factor_override=time_factor,
                travel_mode=travel_mode,
            )
            route_risk["model_used"] = "Fast Quote Engine"
            model_status = "Fallback to fast quote engine"
    else:
        route_risk = calculate_route_risk_factors_fast(
            distance_km,
            start_lat,
            start_lon,
            dest_lat,
            dest_lon,
            step_count=route_step_count,
            google_duration_min=route_duration_min,
            google_signal_estimate=route_signal_estimate,
            signal_source=route_signal_source,
            google_elevation_gain=route_elevation_gain,
            elevation_source=route_elevation_source,
            weather_score_input=combined_weather_risk["score"],
            time_factor_override=time_factor,
            travel_mode=travel_mode,
        )
        route_risk["model_used"] = "Fast Quote Engine"
        model_status = "No saved model available"
    overall_recommendation = build_overall_recommendation(route_risk, routing_confidence, travel_mode)
    ai_risk_explanation = build_ai_risk_explanation(
        route_risk=route_risk,
        routing_confidence=routing_confidence,
        weather_risk=combined_weather_risk,
        overall_recommendation=overall_recommendation,
        travel_mode=travel_mode,
        distance_km=distance_km,
        route_duration_min=route_duration_min,
    )

    return {
        "mode": travel_mode,
        "distance_km": round(distance_km, 2),
        "route_time_min": route_duration_min,
        "risk_score": route_risk.get("risk_score"),
        "risk_level": route_risk.get("risk_level"),
        "road_complexity": route_risk.get("road_complexity"),
        "routing_confidence": routing_confidence.get("level", "LOW"),
        "recommendation": overall_recommendation.get("action"),
        "ai_explanation": ai_risk_explanation.get("summary"),
        "routing_source": routing_confidence.get("source", route_provider),
        "signal_source": route_signal_source,
        "model_used": route_risk.get("model_used", "Fast Quote Engine"),
        "model_status": model_status,
        "signals": route_risk.get("signals", 0),
        "construction": route_risk.get("construction", 0),
        "construction_source": route_risk.get("construction_detail", {}).get("source", "Estimated"),
        "construction_confidence": route_risk.get("construction_detail", {}).get("confidence", "LOW"),
        "weather": combined_weather_risk.get("reason", "No weather data"),
        "weather_impact": combined_weather_risk.get("score", route_risk.get("weather_impact", 5.0)),
        "weather_source": weather_source,
        "weather_confidence": weather_confidence,
        "elevation_gain": route_risk.get("elevation", route_risk.get("elevation_gain", 0)),
        "elevation_source": route_risk.get("elevation_source", "Estimated"),
    }

def main_cli():
    print("Hong Kong Route Risk Analyzer")
    print("=" * 50)
    print("Technical Console Edition")
    print("Interface demo version is provided by Interface_launcher.exe")
    print()

    mode_input = input("Select mode [1=Walking, 2=Driving, 3=Compare Both]: ").strip()
    travel_mode = normalize_travel_mode(mode_input)
    start = input("Enter PICKUP Point (No. / Road): ").strip()
    dest = input("Enter DROPOFF Point (No. / Road): ").strip()

    if not travel_mode:
        print("Please choose 1 for Walking, 2 for Driving, or 3 for Compare Both!")
        return

    if not start or not dest:
        print("Please enter both locations!")
        return

    if travel_mode == "compare":
        print_section_header("Comparison Mode - Driving")
        driving_result = get_route_conditions(start, dest, travel_mode="driving")
        print_section_header("Comparison Mode - Walking")
        walking_result = get_route_conditions(start, dest, travel_mode="walking")
        print_compare_summary(driving_result, walking_result)
        return

    get_route_conditions(start, dest, travel_mode=travel_mode)


if __name__ == "__main__":
    try:
        main_cli()
    finally:
        try:
            input("\nPress Enter to exit...")
        except Exception:
            pass

