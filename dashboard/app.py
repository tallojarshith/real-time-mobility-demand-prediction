from pathlib import Path
import sys

# =========================================================
# PROJECT ROOT / IMPORT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORTS
# =========================================================

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st

from src.prediction.predictor import DemandPredictor
from src.fleet.recommendations import (
    add_zone_metadata,
    create_fleet_recommendations,
)


# =========================================================
# PATHS
# =========================================================

ZONE_LOOKUP_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "taxi_zone_lookup.csv"
)

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "yellow_tripdata_2025-01.parquet"
)

TAXI_ZONE_SHAPEFILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "taxi_zones"
    / "taxi_zones"
    / "taxi_zones.shp"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NYC Taxi Demand Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_zone_lookup():
    return pd.read_csv(
        ZONE_LOOKUP_PATH
    )


@st.cache_data
def load_demand_data():
    """
    Load January 2025 Yellow Taxi pickup data and construct
    a complete zone-hour demand table.
    """

    df = pd.read_parquet(
        RAW_DATA_PATH,
        columns=[
            "tpep_pickup_datetime",
            "PULocationID",
        ],
    )

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"]
    )

    df = df[
        (df["tpep_pickup_datetime"] >= "2025-01-01")
        & (df["tpep_pickup_datetime"] < "2025-02-01")
        & (df["PULocationID"].between(1, 265))
    ].copy()

    df["pickup_hour_timestamp"] = (
        df["tpep_pickup_datetime"].dt.floor("h")
    )

    df["date"] = (
        df["tpep_pickup_datetime"].dt.date
    )

    df["hour"] = (
        df["tpep_pickup_datetime"].dt.hour
    )

    observed_demand = (
        df.groupby(
            [
                "pickup_hour_timestamp",
                "PULocationID",
            ]
        )
        .size()
        .reset_index(name="demand")
    )

    # -----------------------------------------------------
    # COMPLETE ZERO-DEMAND GRID
    # -----------------------------------------------------

    active_zones = sorted(
        df["PULocationID"].unique()
    )

    all_hours = pd.date_range(
        start="2025-01-01 00:00:00",
        end="2025-02-01 00:00:00",
        freq="h",
        inclusive="left",
    )

    complete_grid = (
        pd.MultiIndex.from_product(
            [
                all_hours,
                active_zones,
            ],
            names=[
                "pickup_hour_timestamp",
                "PULocationID",
            ],
        )
        .to_frame(index=False)
    )

    hourly_zone = complete_grid.merge(
        observed_demand,
        on=[
            "pickup_hour_timestamp",
            "PULocationID",
        ],
        how="left",
    )

    hourly_zone["demand"] = (
        hourly_zone["demand"]
        .fillna(0)
        .astype("int32")
    )

    return df, hourly_zone


@st.cache_data
def load_taxi_zone_geometry():
    """
    Load official NYC TLC taxi-zone polygons.

    Geometry is converted to WGS84 because Plotly expects
    longitude/latitude coordinates.
    """

    zones = gpd.read_file(
        TAXI_ZONE_SHAPEFILE
    )

    # Standardize column names from the TLC shapefile.
    rename_map = {}

    if "location_i" in zones.columns:
        rename_map["location_i"] = "LocationID"

    if "LocationID" not in zones.columns:
        for column in zones.columns:
            if column.lower() == "locationid":
                rename_map[column] = "LocationID"

    zones = zones.rename(
        columns=rename_map
    )

    if "LocationID" not in zones.columns:
        raise ValueError(
            "LocationID was not found in taxi-zone shapefile"
        )

    zones["LocationID"] = (
        pd.to_numeric(
            zones["LocationID"],
            errors="coerce",
        )
    )

    zones = zones.dropna(
        subset=["LocationID"]
    ).copy()

    zones["LocationID"] = (
        zones["LocationID"]
        .astype(int)
    )

    # Plotly choropleth maps require longitude/latitude.
    zones = zones.to_crs(
        epsg=4326
    )

    return zones


@st.cache_resource
def load_predictor():
    """
    Load trained Extra Trees model only once.
    """

    return DemandPredictor()


# =========================================================
# LOAD CORE DATA
# =========================================================

try:

    zone_lookup = (
        load_zone_lookup()
    )

    trips, hourly_zone = (
        load_demand_data()
    )

except Exception as error:

    st.error(
        f"Unable to load dashboard data: {error}"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🚕 NYC Taxi Intelligence"
)

st.sidebar.markdown(
    """
**Project**

Real-Time NYC Taxi Demand Forecasting  
& Fleet Decision Support
"""
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Demand Overview",
        "Zone Explorer",
        "Model Performance",
        "Fleet Intelligence",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Data: NYC TLC Yellow Taxi — January 2025"
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "🚕 NYC Taxi Demand Forecasting"
)

st.caption(
    "Zone-level demand intelligence, "
    "next-hour forecasting and fleet decision support"
)


# =========================================================
# PAGE 1 — DEMAND OVERVIEW
# =========================================================

if page == "Demand Overview":

    st.subheader(
        "Demand Overview"
    )

    total_trips = len(
        trips
    )

    active_zones = (
        trips["PULocationID"]
        .nunique()
    )

    busiest_hour = (
        trips["hour"]
        .value_counts()
        .idxmax()
    )

    average_daily = (
        trips.groupby("date")
        .size()
        .mean()
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "January Trips",
        f"{total_trips:,}",
    )

    col2.metric(
        "Active Taxi Zones",
        f"{active_zones}",
    )

    col3.metric(
        "Busiest Hour",
        f"{busiest_hour}:00",
    )

    col4.metric(
        "Average Daily Trips",
        f"{average_daily:,.0f}",
    )

    st.divider()

    # -----------------------------------------------------
    # HOURLY DEMAND
    # -----------------------------------------------------

    hourly_demand = (
        trips.groupby("hour")
        .size()
        .reset_index(
            name="pickups"
        )
    )

    fig_hour = px.line(
        hourly_demand,
        x="hour",
        y="pickups",
        markers=True,
        title=(
            "Total Pickup Demand by Hour of Day"
        ),
        labels={
            "hour":
                "Hour of Day",
            "pickups":
                "Taxi Pickups",
        },
    )

    fig_hour.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # DAILY DEMAND
    # -----------------------------------------------------

    daily_demand = (
        trips.groupby("date")
        .size()
        .reset_index(
            name="pickups"
        )
    )

    fig_daily = px.line(
        daily_demand,
        x="date",
        y="pickups",
        markers=True,
        title=(
            "Daily Yellow Taxi Pickup Demand"
        ),
        labels={
            "date":
                "Date",
            "pickups":
                "Taxi Pickups",
        },
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True,
    )


# =========================================================
# PAGE 2 — ZONE EXPLORER
# =========================================================

elif page == "Zone Explorer":

    st.subheader(
        "Taxi Zone Demand Explorer"
    )

    zone_demand = (
        trips.groupby(
            "PULocationID"
        )
        .size()
        .reset_index(
            name="total_pickups"
        )
    )

    zone_demand = (
        zone_demand.merge(
            zone_lookup[
                [
                    "LocationID",
                    "Borough",
                    "Zone",
                ]
            ],
            left_on="PULocationID",
            right_on="LocationID",
            how="left",
        )
    )

    zone_demand["Zone"] = (
        zone_demand["Zone"]
        .fillna("Unknown Zone")
    )

    zone_demand["Borough"] = (
        zone_demand["Borough"]
        .fillna("Unknown")
    )

    zone_demand = (
        zone_demand.sort_values(
            "total_pickups",
            ascending=False,
        )
    )

    top_n = st.slider(
        "Number of zones to display",
        min_value=5,
        max_value=30,
        value=15,
    )

    top_zones = (
        zone_demand.head(
            top_n
        )
    )

    fig_zones = px.bar(
        top_zones,
        x="total_pickups",
        y="Zone",
        orientation="h",
        title=(
            f"Top {top_n} Pickup Zones"
        ),
        labels={
            "total_pickups":
                "Total Pickups",
            "Zone":
                "Taxi Zone",
        },
        hover_data=[
            "Borough",
            "PULocationID",
        ],
    )

    fig_zones.update_layout(
        yaxis={
            "categoryorder":
                "total ascending"
        }
    )

    st.plotly_chart(
        fig_zones,
        use_container_width=True,
    )

    st.subheader(
        "Zone Ranking"
    )

    st.dataframe(
        zone_demand[
            [
                "PULocationID",
                "Zone",
                "Borough",
                "total_pickups",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# PAGE 3 — MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.subheader(
        "Forecasting Model Performance"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Test MAE",
        "3.220",
    )

    col2.metric(
        "Test RMSE",
        "10.004",
    )

    col3.metric(
        "Test R²",
        "0.974",
    )

    st.info(
        "Metrics are from the held-out "
        "January 28–31, 2025 test window "
        "using rolling one-hour-ahead evaluation."
    )

    st.divider()

    model_results = pd.DataFrame({
        "Model": [
            "Naive Lag-24",
            "Linear Regression",
            "Random Forest",
            "Extra Trees",
            "XGBoost",
            "LightGBM",
            "Tuned Extra Trees",
        ],

        "MAE": [
            8.317,
            5.019,
            4.449,
            4.125,
            4.274,
            4.371,
            4.054,
        ],
    })

    fig_models = px.bar(
        model_results.sort_values(
            "MAE",
            ascending=True,
        ),
        x="MAE",
        y="Model",
        orientation="h",
        title=(
            "Validation MAE — Model Comparison"
        ),
    )

    st.plotly_chart(
        fig_models,
        use_container_width=True,
    )

    st.markdown(
        """
**Final model:** Tuned Extra Trees Regressor

The model uses historical demand lags, rolling demand
statistics, calendar features, taxi-zone ID and weather
variables.

Historical demand features were the dominant predictors.
Weather had only a small incremental effect during the
January validation experiment.
"""
    )


# =========================================================
# PAGE 4 — FLEET INTELLIGENCE
# =========================================================

elif page == "Fleet Intelligence":

    st.subheader(
        "Next-Hour Fleet Intelligence"
    )

    st.info(
        "This page runs the trained Extra Trees model "
        "for each eligible taxi zone and converts predicted "
        "next-hour demand into proportional fleet-allocation "
        "recommendations."
    )

    # -----------------------------------------------------
    # FLEET SIZE
    # -----------------------------------------------------

    available_vehicles = int(
        st.number_input(
            "Available fleet size",
            min_value=1,
            max_value=10000,
            value=500,
            step=50,
        )
    )

    # -----------------------------------------------------
    # TARGET TIMESTAMP
    # -----------------------------------------------------

    latest_observed_timestamp = (
        hourly_zone[
            "pickup_hour_timestamp"
        ].max()
    )

    target_timestamp = (
        latest_observed_timestamp
        + pd.Timedelta(hours=1)
    )

    st.caption(
        "Forecast target: "
        f"{target_timestamp:%Y-%m-%d %H:%M}"
    )

    # -----------------------------------------------------
    # TARGET-HOUR WEATHER
    # -----------------------------------------------------

    with st.expander(
        "Target-hour weather",
        expanded=False,
    ):

        weather_col1, weather_col2 = (
            st.columns(2)
        )

        temperature = (
            weather_col1.number_input(
                "Temperature (°C)",
                value=5.0,
            )
        )

        humidity = (
            weather_col2.number_input(
                "Relative humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
            )
        )

        precipitation = (
            weather_col1.number_input(
                "Precipitation (mm)",
                min_value=0.0,
                value=0.0,
            )
        )

        rain = (
            weather_col2.number_input(
                "Rain (mm)",
                min_value=0.0,
                value=0.0,
            )
        )

        snowfall = (
            weather_col1.number_input(
                "Snowfall (cm)",
                min_value=0.0,
                value=0.0,
            )
        )

        weather_code = int(
            weather_col2.number_input(
                "Weather code",
                min_value=0,
                value=1,
                step=1,
            )
        )

        wind_speed = (
            weather_col1.number_input(
                "Wind speed (km/h)",
                min_value=0.0,
                value=10.0,
            )
        )

    weather_row = pd.DataFrame({
        "temperature_2m": [
            temperature
        ],
        "relative_humidity_2m": [
            humidity
        ],
        "precipitation": [
            precipitation
        ],
        "rain": [
            rain
        ],
        "snowfall": [
            snowfall
        ],
        "weather_code": [
            weather_code
        ],
        "wind_speed_10m": [
            wind_speed
        ],
    })

    # -----------------------------------------------------
    # FORECAST BUTTON
    # -----------------------------------------------------

    if st.button(
        "Generate Next-Hour Forecast",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Running zone-level demand forecasts..."
            ):

                predictor = (
                    load_predictor()
                )

                # -----------------------------------------
                # ELIGIBLE ZONES
                # -----------------------------------------

                zone_counts = (
                    hourly_zone.groupby(
                        "PULocationID"
                    )
                    .size()
                )

                eligible_zones = (
                    zone_counts[
                        zone_counts >= 168
                    ]
                    .index
                    .tolist()
                )

                # -----------------------------------------
                # MODEL PREDICTIONS
                # -----------------------------------------

                predictions = (
                    predictor.predict_multiple_zones(
                        demand_history=(
                            hourly_zone
                        ),
                        weather_row=(
                            weather_row
                        ),
                        target_timestamp=(
                            target_timestamp
                        ),
                        zone_ids=(
                            eligible_zones
                        ),
                    )
                )

                # -----------------------------------------
                # ZONE METADATA
                # -----------------------------------------

                predictions = (
                    add_zone_metadata(
                        predictions=(
                            predictions
                        ),
                        zone_lookup=(
                            zone_lookup
                        ),
                    )
                )

                # -----------------------------------------
                # FLEET ALLOCATION
                # -----------------------------------------

                recommendations = (
                    create_fleet_recommendations(
                        predictions=(
                            predictions
                        ),
                        available_vehicles=(
                            available_vehicles
                        ),
                    )
                )

                # -----------------------------------------
                # SESSION STATE
                # -----------------------------------------

                st.session_state[
                    "fleet_forecast"
                ] = recommendations

                st.session_state[
                    "forecast_timestamp"
                ] = target_timestamp

                st.session_state[
                    "forecast_fleet_size"
                ] = available_vehicles

        except Exception as error:

            st.error(
                "Forecast generation failed: "
                f"{error}"
            )

    # -----------------------------------------------------
    # DISPLAY FORECAST
    # -----------------------------------------------------

    if (
        "fleet_forecast"
        in st.session_state
    ):

        recommendations = (
            st.session_state[
                "fleet_forecast"
            ]
        )

        forecast_timestamp = (
            st.session_state[
                "forecast_timestamp"
            ]
        )

        forecast_fleet_size = (
            st.session_state[
                "forecast_fleet_size"
            ]
        )

        total_predicted_demand = (
            recommendations[
                "predicted_demand"
            ].sum()
        )

        high_priority_zones = (
            recommendations[
                "priority"
            ]
            .astype(str)
            .eq("High")
            .sum()
        )

        # -------------------------------------------------
        # KPIs
        # -------------------------------------------------

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Available Vehicles",
            f"{forecast_fleet_size:,}",
        )

        col2.metric(
            "Predicted Next-Hour Pickups",
            f"{total_predicted_demand:,.0f}",
        )

        col3.metric(
            "Zones Forecasted",
            f"{len(recommendations)}",
        )

        col4.metric(
            "High-Priority Zones",
            f"{high_priority_zones}",
        )

        st.caption(
            "Forecast generated for "
            f"{forecast_timestamp:%Y-%m-%d %H:%M}"
        )

        st.divider()

        # =================================================
        # INTERACTIVE NYC DEMAND MAP
        # =================================================

        st.subheader(
            "NYC Spatial Demand Forecast"
        )

        st.caption(
            "Taxi-zone polygons are colored by predicted "
            "next-hour pickup demand. Hover over a zone "
            "for forecast and fleet-allocation details."
        )

        try:

            taxi_zones = (
                load_taxi_zone_geometry()
            )

            # ---------------------------------------------
            # MERGE FORECAST WITH GEOMETRY
            # ---------------------------------------------

            map_data = (
                taxi_zones.merge(
                    recommendations[
                        [
                            "PULocationID",
                            "Zone",
                            "Borough",
                            "predicted_demand",
                            "recommended_vehicles",
                            "demand_share",
                            "priority",
                        ]
                    ],
                    left_on="LocationID",
                    right_on="PULocationID",
                    how="left",
                )
            )

            # Zones not forecasted remain visible with 0.
            map_data[
                "predicted_demand"
            ] = (
                map_data[
                    "predicted_demand"
                ]
                .fillna(0)
            )

            map_data[
                "recommended_vehicles"
            ] = (
                map_data[
                    "recommended_vehicles"
                ]
                .fillna(0)
                .astype(int)
            )

            map_data[
                "demand_share"
            ] = (
                map_data[
                    "demand_share"
                ]
                .fillna(0)
            )

            map_data[
                "Zone"
            ] = (
                map_data[
                    "Zone"
                ]
                .fillna(
                    map_data.get(
                        "zone",
                        pd.Series(
                            "Unknown Zone",
                            index=map_data.index,
                        ),
                    )
                )
                .fillna("Unknown Zone")
            )

            map_data[
                "Borough"
            ] = (
                map_data[
                    "Borough"
                ]
                .fillna(
                    map_data.get(
                        "borough",
                        pd.Series(
                            "Unknown",
                            index=map_data.index,
                        ),
                    )
                )
                .fillna("Unknown")
            )

            map_data[
                "priority"
            ] = (
                map_data[
                    "priority"
                ]
                .astype(str)
                .replace(
                    "nan",
                    "Low",
                )
            )

            # ---------------------------------------------
            # GEOJSON
            # ---------------------------------------------

            geojson = (
                map_data.__geo_interface__
            )

            # ---------------------------------------------
            # PLOTLY CHOROPLETH
            # ---------------------------------------------

            fig_map = (
                px.choropleth_map(
                    map_data,
                    geojson=geojson,
                    locations=map_data.index,
                    color="predicted_demand",
                    hover_name="Zone",
                    hover_data={
                        "Borough": True,
                        "PULocationID": True,
                        "predicted_demand":
                            ":.2f",
                        "recommended_vehicles":
                            True,
                        "demand_share":
                            ":.2%",
                        "priority":
                            True,
                    },
                    color_continuous_scale=(
                        "YlOrRd"
                    ),
                    map_style=(
                        "carto-darkmatter"
                    ),
                    center={
                        "lat": 40.7128,
                        "lon": -74.0060,
                    },
                    zoom=9,
                    opacity=0.75,
                    labels={
                        "predicted_demand":
                            "Predicted Pickups",
                        "PULocationID":
                            "Zone ID",
                        "recommended_vehicles":
                            "Recommended Vehicles",
                        "demand_share":
                            "Demand Share",
                        "priority":
                            "Priority",
                    },
                )
            )

            fig_map.update_layout(
                height=700,
                margin={
                    "r": 0,
                    "t": 20,
                    "l": 0,
                    "b": 0,
                },
                coloraxis_colorbar={
                    "title":
                        "Predicted<br>Pickups"
                },
            )

            st.plotly_chart(
                fig_map,
                use_container_width=True,
            )

        except Exception as error:

            st.warning(
                "NYC map could not be rendered: "
                f"{error}"
            )

        st.divider()

        # =================================================
        # TOP PREDICTED DEMAND
        # =================================================

        top_demand = (
            recommendations
            .sort_values(
                "predicted_demand",
                ascending=False,
            )
            .head(15)
        )

        fig_demand = px.bar(
            top_demand,
            x="predicted_demand",
            y="Zone",
            orientation="h",
            title=(
                "Top 15 Zones — Predicted "
                "Next-Hour Demand"
            ),
            labels={
                "predicted_demand":
                    "Predicted Pickups",
                "Zone":
                    "Taxi Zone",
            },
            hover_data=[
                "Borough",
                "PULocationID",
                "priority",
            ],
        )

        fig_demand.update_layout(
            yaxis={
                "categoryorder":
                    "total ascending"
            }
        )

        st.plotly_chart(
            fig_demand,
            use_container_width=True,
        )

        # =================================================
        # FLEET ALLOCATION
        # =================================================

        top_fleet = (
            recommendations
            .sort_values(
                "recommended_vehicles",
                ascending=False,
            )
            .head(15)
        )

        fig_fleet = px.bar(
            top_fleet,
            x="recommended_vehicles",
            y="Zone",
            orientation="h",
            title=(
                "Recommended Vehicle Distribution "
                "— Top 15 Zones"
            ),
            labels={
                "recommended_vehicles":
                    "Recommended Vehicles",
                "Zone":
                    "Taxi Zone",
            },
            hover_data=[
                "Borough",
                "predicted_demand",
                "demand_share",
                "priority",
            ],
        )

        fig_fleet.update_layout(
            yaxis={
                "categoryorder":
                    "total ascending"
            }
        )

        st.plotly_chart(
            fig_fleet,
            use_container_width=True,
        )

        # =================================================
        # TABLE
        # =================================================

        st.subheader(
            "Fleet Recommendation Table"
        )

        display_df = (
            recommendations[
                [
                    "PULocationID",
                    "Zone",
                    "Borough",
                    "predicted_demand",
                    "demand_share",
                    "priority",
                    "recommended_vehicles",
                ]
            ]
            .copy()
        )

        display_df[
            "predicted_demand"
        ] = (
            display_df[
                "predicted_demand"
            ].round(2)
        )

        display_df[
            "demand_share"
        ] = (
            display_df[
                "demand_share"
            ]
            * 100
        ).round(2)

        display_df = (
            display_df.rename(
                columns={
                    "PULocationID":
                        "Zone ID",
                    "predicted_demand":
                        "Predicted Pickups",
                    "demand_share":
                        "Demand Share (%)",
                    "priority":
                        "Priority",
                    "recommended_vehicles":
                        "Recommended Vehicles",
                }
            )
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # =================================================
        # DOWNLOAD
        # =================================================

        csv_data = (
            display_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label=(
                "Download Fleet Recommendations"
            ),
            data=csv_data,
            file_name=(
                "fleet_recommendations.csv"
            ),
            mime="text/csv",
        )