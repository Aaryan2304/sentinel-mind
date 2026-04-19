"""Sentinel Mind — Streamlit Dashboard.

Run with: streamlit run src/dashboard/app.py

Dashboard components:
- Satellite orbit map (folium)
- Current captured image with band toggle
- Spectral analysis overlays (NDVI, NDWI, NBR heatmaps)
- VLM investigation report
- Downlink priority queue with compression stats
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Sentinel Mind",
        page_icon="🛰️",
        layout="wide",
    )

    st.title("Sentinel Mind — Autonomous Satellite Intelligence")
    st.caption("Decide. Investigate. Prioritize. Downlink.")

    # Sidebar: simulation controls
    with st.sidebar:
        st.header("Simulation Controls")
        if st.button("Start Simulation"):
            st.session_state["running"] = True
        if st.button("Stop Simulation"):
            st.session_state["running"] = False

        speed = st.slider("Replay Speed", min_value=1, max_value=20, value=10)
        region = st.selectbox("Region", ["Amazon Basin", "Congo Basin", "Southeast Asia"])

    # Main area placeholders
    col_map, col_image = st.columns(2)
    with col_map:
        st.subheader("Satellite Orbit")
        st.info("Map component (streamlit-folium) — Phase 6")

    with col_image:
        st.subheader("Current Capture")
        st.info("Image display — Phase 6")

    st.subheader("Spectral Analysis")
    col_ndvi, col_ndwi, col_nbr = st.columns(3)
    with col_ndvi:
        st.metric("NDVI", "—")
    with col_ndwi:
        st.metric("NDWI", "—")
    with col_nbr:
        st.metric("NBR", "—")

    st.subheader("VLM Investigation Report")
    st.info("Investigation output — Phase 5/6")

    st.subheader("Downlink Priority Queue")
    st.info("Queue table — Phase 6")


if __name__ == "__main__":
    main()
