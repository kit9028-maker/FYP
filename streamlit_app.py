import io
from contextlib import redirect_stdout

import streamlit as st

from Main_program import format_display_value, get_route_conditions, get_route_quote_summary


DELIVERY_TYPE_FACTOR = {
    "Food Delivery": 0.18,
    "Parcel Delivery": 0.12,
    "Document Delivery": 0.08,
    "Grocery Delivery": 0.22,
}

DELIVERY_SIZE_FACTOR = {
    "Small": 0.10,
    "Medium": 0.24,
    "Large": 0.42,
}


def build_app_css() -> str:
    return """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, #e8f4fb 0%, #f6fbff 38%, #edf6fa 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1120px;
    }
    .sim-title {
        text-align: center;
        color: #12384d;
        font-size: 2.15rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .sim-subtitle {
        text-align: center;
        color: #517081;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .sim-card {
        background: linear-gradient(180deg, #176285 0%, #165877 100%);
        border: 3px solid #173646;
        border-radius: 28px;
        padding: 1.4rem 1.4rem 1.6rem 1.4rem;
        box-shadow: 0 18px 38px rgba(22, 76, 102, 0.18);
        margin-bottom: 1.75rem;
    }
    .sim-card-header {
        width: 84%;
        margin: 0 auto 1.15rem auto;
        text-align: center;
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
        padding: 0.8rem 1rem;
        border-radius: 22px;
        border: 3px solid #1b3440;
        background: linear-gradient(180deg, rgba(40,126,167,0.95), rgba(30,95,128,0.95));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.14);
    }
    .helper-card {
        background: #fff0e5;
        border: 2px solid #1d5068;
        border-radius: 18px;
        color: #234455;
        padding: 1.2rem 1.1rem;
        font-size: 1rem;
        line-height: 1.7;
        box-shadow: 0 10px 26px rgba(25, 79, 103, 0.08);
        margin-top: 9rem;
    }
    .result-panel {
        background: rgba(11, 55, 75, 0.16);
        border: 3px solid rgba(14, 39, 52, 0.85);
        border-radius: 14px;
        padding: 1.3rem 1.1rem 1.35rem 1.1rem;
        color: #f3fbff;
        min-height: 365px;
    }
    .result-title {
        text-align: center;
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.85rem;
    }
    .result-subtitle {
        text-align: center;
        font-size: 1.15rem;
        color: #dff1fb;
        margin-bottom: 1.25rem;
    }
    .summary-chip {
        display: inline-block;
        background: rgba(226, 244, 255, 0.14);
        border: 1px solid rgba(228, 244, 255, 0.35);
        color: #f7fdff;
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        font-size: 0.95rem;
        margin: 0 0.35rem 0.55rem 0;
    }
    .metric-strip {
        background: rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin: 1rem 0 1rem 0;
        color: #f3fbff;
        font-size: 1rem;
        line-height: 1.7;
    }
    .risk-heading {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.45rem;
        margin-bottom: 0.45rem;
    }
    .factor-summary-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem 0.9rem;
        margin-top: 0.35rem;
        margin-bottom: 1rem;
    }
    .factor-card {
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(233, 245, 252, 0.22);
        border-radius: 12px;
        padding: 0.9rem 0.95rem;
        color: #f4fbff;
    }
    .factor-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #dff1fb;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .factor-card-value {
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .factor-card-caption {
        font-size: 0.92rem;
        color: #d2e8f4;
        line-height: 1.55;
    }
    .result-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.55rem 0.85rem;
        margin-bottom: 1rem;
    }
    .result-grid-row {
        display: contents;
    }
    .result-label {
        background: rgba(242, 247, 251, 0.96);
        color: #173d4f;
        border-radius: 10px;
        padding: 0.72rem 0.9rem;
        font-size: 1rem;
        font-weight: 700;
    }
    .result-value {
        background: rgba(255,255,255,0.96);
        color: #234557;
        border-radius: 10px;
        padding: 0.72rem 0.9rem;
        font-size: 1rem;
        font-weight: 600;
    }
    .risk-list {
        color: #f5fbff;
        font-size: 1.06rem;
        line-height: 1.85;
        margin-bottom: 1.1rem;
    }
    .premium-box {
        background: linear-gradient(180deg, #e8edf5 0%, #a6bdd1 100%);
        border-radius: 20px;
        padding: 1rem 1.2rem;
        color: #16384b;
        text-align: center;
        margin-top: 1.2rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.68);
    }
    .premium-box .label {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .premium-box .value {
        font-size: 2.1rem;
        font-weight: 800;
    }
    .caption-note {
        color: #d8edf9;
        font-size: 0.92rem;
        margin-top: 0.7rem;
        text-align: center;
    }
    .ai-section-title {
        color: #ffffff;
        font-size: 1.18rem;
        font-weight: 700;
        margin-top: 0.95rem;
        margin-bottom: 0.5rem;
    }
    .ai-explanation-box {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(226, 242, 251, 0.22);
        border-radius: 12px;
        padding: 0.95rem 1rem;
        color: #f3fbff;
        font-size: 1rem;
        line-height: 1.75;
        margin-bottom: 0.9rem;
    }
    .detail-three-col {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 0.3rem;
        margin-bottom: 1rem;
    }
    .detail-section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(233, 245, 252, 0.18);
        border-radius: 14px;
        padding: 0.85rem 0.85rem 0.15rem 0.85rem;
    }
    .detail-group-title {
        color: #dff1fb;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    @media (max-width: 980px) {
        .factor-summary-grid,
        .detail-three-col,
        .result-grid {
            grid-template-columns: 1fr;
        }
    }
    div[data-testid="stForm"] {
        background: linear-gradient(180deg, #176285 0%, #165877 100%);
        border: 3px solid #173646;
        border-radius: 28px;
        padding: 1.4rem 1.4rem 1.6rem 1.4rem;
        box-shadow: 0 18px 38px rgba(22, 76, 102, 0.18);
        margin-bottom: 1.75rem;
    }
    div[data-testid="stForm"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label {
        color: #f7fdff !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        border-radius: 10px !important;
    }
    .stButton button,
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        border-radius: 20px;
        border: 0;
        color: #14374a;
        font-weight: 800;
        font-size: 1.05rem;
        background: linear-gradient(180deg, #e4edf7 0%, #a7bfd5 100%);
        padding: 0.85rem 1.2rem;
        box-shadow: 0 10px 20px rgba(7, 41, 57, 0.20);
    }
    </style>
    """


@st.cache_data(show_spinner=False, ttl=900)
def analyse_route_summary(start: str, dest: str, mode: str) -> dict:
    return get_route_quote_summary(start, dest, travel_mode=mode)


@st.cache_data(show_spinner=False, ttl=900)
def analyse_route_output(start: str, dest: str, mode: str) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        get_route_conditions(start, dest, travel_mode=mode, interactive=False)
    return buffer.getvalue()


def calculate_premium(result: dict, delivery_type: str, delivery_size: str, previous_claims: int) -> float:
    base_price = 1.20
    risk_component = float(result.get("risk_score", 0.0) or 0.0) * 0.22
    delivery_component = DELIVERY_TYPE_FACTOR.get(delivery_type, 0.12)
    size_component = DELIVERY_SIZE_FACTOR.get(delivery_size, 0.10)
    claims_component = min(max(previous_claims, 0), 5) * 0.14
    mode_component = 0.10 if result.get("mode") == "driving" else 0.04
    confidence_component = {
        "High": 0.06,
        "Medium": 0.10,
        "Low": 0.16,
    }.get(str(result.get("routing_confidence", "Medium")).title(), 0.10)
    premium = base_price + risk_component + delivery_component + size_component + claims_component + mode_component + confidence_component
    return round(premium, 2)


def build_risk_factors(result: dict, delivery_type: str, delivery_size: str, previous_claims: int) -> list[str]:
    factors = [
        f"Route risk is assessed as {result.get('risk_level', 'Medium')} with a score of {float(result.get('risk_score', 0.0)):.2f}.",
        f"Travel context shows {float(result.get('distance_km', 0.0)):.1f} km, approximately {int(result.get('route_time_min', 0) or 0)} minutes, and routing confidence rated {result.get('routing_confidence', 'Medium')}.",
        f"Weather condition is based on {result.get('weather', 'baseline weather assumption')} with a weather impact score of {float(result.get('weather_impact', 5.0) or 5.0):.1f}/10.",
        f"Route factors include {int(result.get('signals', 0) or 0)} estimated intersections/signals, construction count {int(result.get('construction', 0) or 0)}, and elevation gain {int(result.get('elevation_gain', 0) or 0)} m.",
    ]
    factors.append(
        f"Operational profile records {delivery_type.lower()}, {delivery_size.lower()} item size, and {previous_claims} previous claim(s)."
    )
    recommendation = result.get("recommendation")
    if recommendation:
        factors.append(f"Recommended action: {recommendation}.")
    return factors


def render_result_card(result: dict, premium: float, delivery_type: str, delivery_size: str, previous_claims: int) -> None:
    risk_factors = build_risk_factors(result, delivery_type, delivery_size, previous_claims)
    risk_level = format_display_value(result.get("risk_level", "Medium"))
    risk_score = float(result.get("risk_score", 0.0) or 0.0)
    distance_km = float(result.get("distance_km", 0.0) or 0.0)
    route_time = int(result.get("route_time_min", 0) or 0)
    routing_confidence = format_display_value(result.get("routing_confidence", "Medium"))
    road_complexity = float(result.get("road_complexity", 0.0) or 0.0)
    recommendation = result.get("recommendation", "Proceed with caution")
    ai_explanation = result.get("ai_explanation", "The current route has been analysed using the hybrid risk engine.")
    model_used = format_display_value(result.get("model_used", "Fast Quote Engine"))
    model_status = result.get("model_status", "")
    weather_text = result.get("weather", "No weather data")
    weather_impact = float(result.get("weather_impact", 5.0) or 5.0)
    weather_source = result.get("weather_source", "Not Available")
    weather_confidence = format_display_value(result.get("weather_confidence", "Low"))
    signals = int(result.get("signals", 0) or 0)
    construction = int(result.get("construction", 0) or 0)
    construction_source = result.get("construction_source", "Estimated")
    construction_confidence = format_display_value(result.get("construction_confidence", "Low"))
    elevation_gain = int(result.get("elevation_gain", 0) or 0)
    elevation_source = result.get("elevation_source", "Estimated")
    signal_source = result.get("signal_source", "Estimated")
    routing_source = result.get("routing_source", "Google Directions")

    route_detail_rows = [
        ("Mode", f"{result.get('mode', 'driving').title()}"),
        ("Distance", f"{distance_km:.1f} km"),
        ("Route Time", f"{route_time} min"),
        ("Routing Confidence", routing_confidence),
        ("Routing Source", routing_source),
        ("Recommended Action", recommendation),
    ]
    model_detail_rows = [
        ("Risk Score", f"{risk_score:.2f}"),
        ("Road Complexity", f"{road_complexity:.2f}/10"),
        ("Model Engine", model_used),
        ("Model Status", model_status or "Not Available"),
        ("Weather Explanation", weather_text),
        ("Weather Impact", f"{weather_impact:.1f}/10"),
        ("Weather Source", weather_source),
        ("Weather Confidence", weather_confidence),
        ("Estimated Signals / Intersections", str(signals)),
        ("Signal Source", signal_source),
        ("Construction Count", str(construction)),
        ("Construction Source", construction_source),
        ("Construction Confidence", construction_confidence),
        ("Elevation Gain", f"{elevation_gain} m"),
        ("Elevation Source", elevation_source),
    ]
    profile_detail_rows = [
        ("Delivery Type", delivery_type),
        ("Delivery Size", delivery_size),
        ("Previous Claims", str(previous_claims)),
    ]

    def _rows_to_html(rows: list[tuple[str, str]]) -> str:
        return "".join(
            (
                '<div class="result-grid-row">'
                f'<div class="result-label">{label}</div>'
                f'<div class="result-value">{value}</div>'
                '</div>'
            )
            for label, value in rows
        )

    route_rows_html = _rows_to_html(route_detail_rows)
    model_rows_html = _rows_to_html(model_detail_rows)
    profile_rows_html = _rows_to_html(profile_detail_rows)
    risk_html = "".join(f"{index}) {factor}<br>" for index, factor in enumerate(risk_factors, start=1))
    factor_cards_html = (
        f"""
        <div class="factor-card">
            <div class="factor-card-title">Weather</div>
            <div class="factor-card-value">{weather_impact:.1f}/10</div>
            <div class="factor-card-caption">{weather_text}<br>Source: {weather_source}</div>
        </div>
        <div class="factor-card">
            <div class="factor-card-title">Estimated Signals</div>
            <div class="factor-card-value">{signals}</div>
            <div class="factor-card-caption">Source: {signal_source}</div>
        </div>
        <div class="factor-card">
            <div class="factor-card-title">Construction</div>
            <div class="factor-card-value">{construction}</div>
            <div class="factor-card-caption">Source: {construction_source}</div>
        </div>
        <div class="factor-card">
            <div class="factor-card-title">Elevation</div>
            <div class="factor-card-value">{elevation_gain} m</div>
            <div class="factor-card-caption">Source: {elevation_source}</div>
        </div>
        """
    )
    result_html = f"""
    <div class="sim-card">
        <div class="sim-card-header">GigShield Risk Simulator</div>
        <div class="result-panel">
            <div class="result-title">Risk Calculation Result</div>
            <div class="result-subtitle">{risk_level} Risk | Score {risk_score:.2f} | {recommendation}</div>
            <div class="risk-heading">Risk Factors Summary</div>
            <div class="factor-summary-grid">{factor_cards_html}</div>
            <div class="ai-section-title">Risk Explanation</div>
            <div class="ai-explanation-box">{ai_explanation}</div>
            <div class="risk-heading">Core Analysis Details</div>
            <div class="detail-three-col">
                <div class="detail-section-card">
                    <div class="detail-group-title">Route Details</div>
                    <div class="result-grid">{route_rows_html}</div>
                </div>
                <div class="detail-section-card">
                    <div class="detail-group-title">Risk Model Details</div>
                    <div class="result-grid">{model_rows_html}</div>
                </div>
                <div class="detail-section-card">
                    <div class="detail-group-title">Operational Profile</div>
                    <div class="result-grid">{profile_rows_html}</div>
                </div>
            </div>
            <div class="risk-heading">Risk Factors</div>
            <div class="risk-list">{risk_html}</div>
        </div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="GigShield Risk Simulator", layout="wide")
    st.markdown(build_app_css(), unsafe_allow_html=True)

    st.markdown('<div class="sim-title">GigShield Risk Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sim-subtitle">Retrieve route data from open APIs and calculate the delivery route risk with AI-driven computation.</div>',
        unsafe_allow_html=True,
    )

    if "quotation_result" not in st.session_state:
        st.session_state.quotation_result = None
    if "quotation_error" not in st.session_state:
        st.session_state.quotation_error = ""
    if "technical_output" not in st.session_state:
        st.session_state.technical_output = None

    st.markdown('<div class="sim-card-header">GigShield Risk Simulator</div>', unsafe_allow_html=True)
    with st.form("premium_simulator_form", clear_on_submit=False):
        start = st.text_input("Start Location", value="", placeholder="Enter the pickup point")
        dest = st.text_input("Destination", value="", placeholder="Enter the destination")
        delivery_type = st.selectbox("Delivery", list(DELIVERY_TYPE_FACTOR.keys()))
        delivery_size = st.selectbox("Delivery Size", list(DELIVERY_SIZE_FACTOR.keys()))
        previous_claims = st.number_input("Previous Claims", min_value=0, max_value=10, value=0, step=1)
        mode = st.selectbox("Mode", ["Driving", "Walking"])
        submitted = st.form_submit_button("Risk Calculation")

    if submitted:
        st.session_state.quotation_error = ""
        if not start.strip() or not dest.strip():
            st.session_state.quotation_error = "Please enter both start location and destination."
        else:
            mode_value = mode.strip().lower()
            try:
                with st.spinner("Calculating route risk result..."):
                    result = analyse_route_summary(start.strip(), dest.strip(), mode_value)
                    st.session_state.quotation_result = {
                        "result": result,
                        "delivery_type": delivery_type,
                        "delivery_size": delivery_size,
                        "previous_claims": int(previous_claims),
                        "start": start.strip(),
                        "dest": dest.strip(),
                        "mode": mode_value,
                    }
                    st.session_state.technical_output = None
            except Exception as exc:
                st.session_state.quotation_result = None
                st.session_state.technical_output = None
                st.session_state.quotation_error = f"Unable to calculate the route risk right now: {exc}"

    if st.session_state.quotation_error:
        st.error(st.session_state.quotation_error)

    if st.session_state.quotation_result:
        stored = st.session_state.quotation_result
        render_result_card(
            stored["result"],
            0.0,
            stored["delivery_type"],
            stored["delivery_size"],
            stored["previous_claims"],
        )
        with st.expander("View Full Technical Analysis"):
            if st.session_state.technical_output is None:
                if st.button("Generate Technical Analysis", use_container_width=True):
                    with st.spinner("Preparing full technical analysis..."):
                        st.session_state.technical_output = analyse_route_output(
                            stored["start"],
                            stored["dest"],
                            stored["mode"],
                        )
                    st.rerun()
                else:
                    st.info("Generate the detailed analysis only when you need it. This keeps the main risk result loading faster.")
            else:
                st.code(st.session_state.technical_output, language="text")


if __name__ == "__main__":
    main()
