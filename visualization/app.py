from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
from dash import Input, Output, State, callback
import plotly.express as px

# from app import app

import sys
sys.path.append("../notebooks")

from week4 import get_all_crime_probabilities
df = pd.read_parquet("../data/crime_clean.parquet", columns=["date", "primary_type", "arrest", "district", "block", "latitude", "longitude"])

crime_options = [{"label": crime, "value": crime} for crime in sorted(df["primary_type"].unique())]

overview_tab = dbc.Container([
    html.H3("Chicago Crime Overview", className="mt-3 mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Date Range"),
            html.Br(),
            dcc.DatePickerRange(
                id="overview-date-range",
                start_date=df["date"].min(),
                end_date=df["date"].max(),
                display_format="YYYY-MM-DD",
                clearable=True
            )
        ], width=3),
        dbc.Col([
                html.Label("Crime Type"),
                dcc.Dropdown(
                    id="filter-crime-type",
                    options=crime_options,
                    multi=True,
                )
            ], width=6),
        dbc.Col([
            html.Br(),
            dbc.Button("Refresh", id="overview-refresh-btn", color="primary", className="mt-1")
        ], width=3),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Crimes"), html.H3(id="kpi-total-crimes")])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("YoY Change"), html.H3(id="kpi-yoy-change")])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Most Common Crime"), html.H3(id="kpi-common-crime")])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Arrest Rate"), html.H3(id="kpi-arrest-rate")])), width=3),
    ], className="mb-5"),

    # Charts
    dbc.Row([
        dbc.Col([html.H5("Crime Trend Over Time"), dcc.Graph(id="crime-trend-chart", style={"height": "400px"})], width=7),
        dbc.Col([html.H5("Top Crime Types"), dcc.Graph(id="top-crime-types-chart", style={"height": "400px"})], width=5),
    ], className="mb-5"),

], fluid=True)


# FILTER FUNCTION
def filter_df(df, start=None, end=None, crime=None):
    temp = df.copy()
    
    if start:
        temp = temp[temp["date"] >= pd.to_datetime(start)]
    if end:
        temp = temp[temp["date"] <= pd.to_datetime(end)]
    if crime:
        temp = temp[temp["primary_type"].isin(crime)]
    
    return temp

# KPI CALLBACK
@callback(
    Output("kpi-total-crimes", "children"),
    Output("kpi-yoy-change", "children"),
    Output("kpi-common-crime", "children"),
    Output("kpi-arrest-rate", "children"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
    Input("filter-crime-type", "value"),
)
def update_kpis(start, end, crime):
    filt = filter_df(df, start, end, crime)
    if len(filt) == 0:
        filt = df.copy()

    total = len(filt)
    common = filt["primary_type"].mode()[0] if len(filt) else "—"
    arrest_rate = round(filt["arrest"].mean() * 100, 2) if len(filt) else 0

    filt["year"] = filt["date"].dt.year
    curr = filt[filt["year"] == filt["year"].max()]
    prev = filt[filt["year"] == filt["year"].max() - 1]
    yoy = round(((len(curr) - len(prev)) / len(prev)) * 100, 2) if len(prev) else 0

    return total, f"{yoy}%", common, f"{arrest_rate}%"


# CRIME TREND CALLBACK
@callback(
    Output("crime-trend-chart", "figure"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
    Input("filter-crime-type", "value"),
)
def update_trend(start, end, crime):
    filt = filter_df(df, start, end, crime)
    if len(filt) == 0:
        filt = df.copy()
    filt["Year"] = filt["date"].dt.to_period("M").astype(str)
    trend = filt.groupby("Year").size().reset_index(name="Count")
    return px.line(trend, x="Year", y="Count", title="Crime Trend Over Time")


# TOP CRIME TYPES CALLBACK
@callback(
    Output("top-crime-types-chart", "figure"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
    Input("filter-crime-type", "value"),
)
def update_top_crimes(start, end, crime):
    filt = filter_df(df, start, end, crime)
    if len(filt) == 0:
        filt = df.copy()
    top = filt["primary_type"].value_counts().head(10)
    
    return px.bar(top, x=top.values, y=top.index, orientation="h", title="Top 10 Crime Types",
        color_discrete_sequence=px.colors.qualitative.Vivid)

# 2. HOTSPOT MAP TAB
hotspot_tab = dbc.Container([
    html.H3("Hotspot Analysis"),
    html.Br(),

    dbc.Row([
        dbc.Col([
            html.Iframe(
                id="folium-map",
                srcDoc=open("../outputs/h3_hex_hotspots.html", "r").read(),
                width="100%",
                height="600"
            ),
        ])
    ]),
])

# 3. ML PREDICTION TAB
prediction_tab = dbc.Container([
    html.H3("Crime Risk Prediction (ML Models)", className="mt-2 mb-4"),

    dbc.Row([

        dbc.Col([
            html.Label("Select Date & Time"),
            html.Br(),
            dcc.DatePickerSingle(id="pred-date"),
            html.Br(), html.Br(),
            html.Label("Hour"),
            html.Br(),
            dcc.Input(
                id="pred-hour", type="number", min=0, max=23,
                placeholder="Hour",
                className="mb-3",
            ),
            html.Br(),
            html.Label("Location (Lat, Long)", style={"margin-right": "130px"}),
            dcc.Input(id="pred-lat", type="number", placeholder="Latitude"),
            html.Br(),
            dcc.Input(id="pred-lng", type="number", placeholder="Longitude"),

            html.Br(), html.Br(),

            dbc.Button(
                "Predict Crime Probability",
                id="predict-btn",
                color="primary"
            ),

            html.Br(), html.Br(),

            html.Div(
                id="prediction-output",
                className="p-3 bg-light border rounded"
            )
        ], width=4),

        dbc.Col([
            dcc.Graph(
                id="probability-chart",
                style={"height": "450px"}
            )
        ], width=8),
    ], className="mb-4"),

])

@app.callback(
    Output("prediction-output", "children"),
    Output("probability-chart", "figure"),
    Input("predict-btn", "n_clicks"),
    State("pred-date", "date"),
    State("pred-hour", "value"),
    State("pred-lat", "value"),
    State("pred-lng", "value")
)
def predict_callback(n_clicks, date, hour, lat, lng):

    if not n_clicks:
        return "Enter inputs above to generate prediction.", {}

    if None in [date, hour, lat, lng]:
        return "Please fill all fields.", {}

    # Convert date
    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")

    # Raw output (logits / scores)
    prob_raw = get_all_crime_probabilities(date_str, hour, lat, lng)

    # Normalize using softmax
    crimes = list(prob_raw.keys())
    vals = np.array(list(prob_raw.values()), dtype=float)
    exp_vals = np.exp(vals - np.max(vals))
    probs = exp_vals / exp_vals.sum()
    prob_dict = dict(zip(crimes, probs))

    # Build DataFrame
    df_probs = (
        pd.DataFrame(prob_dict.items(), columns=["Crime", "Probability"])
        .sort_values(by="Probability", ascending=False)
    )

    # Top prediction
    top1 = df_probs.iloc[0]
    top3 = df_probs.head(3)

    # LEFT PANEL OUTPUT (Top Crime + Top 3 list)
    output_div = html.Div([
        html.H4(f"Predicted Crime Type: {top1.Crime}", className="text-primary mb-3"),

        html.H5("Top 3 Predicted Crimes"),
        html.Ol([
            html.Li(f"{row.Crime} — {row.Probability * 100:.2f}%")
            for _, row in top3.iterrows()
        ])
    ])

    df_plot = df_probs.copy()
    df_plot["Color"] = ["rgba(0,0,150,0.55)"] * len(df_plot)
    df_plot.loc[df_plot["Probability"].idxmax(), "Color"] = "rgba(200,0,0,0.85)"

    prob_fig = px.bar(
        df_plot,
        x="Crime",
        y="Probability",
        color="Color",
        color_discrete_map="identity",
        text=df_plot["Probability"].apply(lambda p: f"{p*100:.1f}%"),
        title="Probability Distribution Across Crime Types"
    )

    prob_fig.update_layout(
        showlegend=False,
        xaxis_title="Crime Category",
        yaxis_title="Predicted Probability",
        margin=dict(l=20, r=20, t=60, b=120)
    )
    prob_fig.update_traces(textposition="outside")


    df_temp = df.copy()
    df_temp["Hour"] = df_temp["date"].dt.hour

    return output_div, prob_fig

# 4. EXPLORATORY ANALYSIS TAB
eda_tab = dbc.Container([
    html.H3("Exploratory Data Analysis (EDA)", className="mt-3 mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Date Range"),
            dcc.DatePickerRange(
                id="eda-date-range",
                start_date=df["date"].min(),
                end_date=df["date"].max(),
                display_format="YYYY-MM-DD"
            )
        ], width=4),

        dbc.Col([
            html.Label("Crime Type"),
            dcc.Dropdown(
                id="eda-crime-type",
                options=[{"label": c, "value": c} for c in sorted(df["primary_type"].unique())],
                multi=True,
                placeholder="Select crime types..."
            )
        ], width=6),
    ], className="mb-4"),

    # Row 1 — Year & Month
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5("Crime Count by Year", className="mb-3"),
                    dcc.Graph(id="eda-yearly-chart", style={"height": "350px"})
                ])
            )
        ], width=6),

        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5("Crime Count by Month", className="mb-3"),
                    dcc.Graph(id="eda-monthly-chart", style={"height": "350px"})
                ])
            )
        ], width=6),
    ], className="mb-4"),

    # Row 2 — Arrest Pie
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5("Arrest Ratio", className="mb-3"),
                    dcc.Graph(id="eda-arrest-pie", style={"height": "350px"})
                ])
            )
        ], width=6),

        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5("Crime Frequency by Hour", className="mb-3"),
                    dcc.Graph(id="eda-hourly-chart", style={"height": "350px"})
                ])
            )
        ]),
    ], className="mb-4"),

], fluid=True)

# -----------------------------
# EDA CALLBACKS
# -----------------------------
@app.callback(
    Output("eda-yearly-chart", "figure"),
    Output("eda-monthly-chart", "figure"),
    Output("eda-arrest-pie", "figure"),
    Output("eda-hourly-chart", "figure"),
    Input("eda-date-range", "start_date"),
    Input("eda-date-range", "end_date"),
    Input("eda-crime-type", "value")
)
def update_eda(start_date, end_date, crime_list):

    # FILTER DATA
    df2 = df.copy()

    if start_date:
        df2 = df2[df2["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df2 = df2[df2["date"] <= pd.to_datetime(end_date)]
    if crime_list:
        df2 = df2[df2["primary_type"].isin(crime_list)]

    # Yearly Trend
    df2["Year"] = df2["date"].dt.year
    yearly = df2.groupby("Year").size().reset_index(name="Count")
    fig_year = px.line(yearly, x="Year", y="Count", title="Crime Count by Year")
    fig_year.update_layout(height=300)

    # Monthly Trend
    df2["Month"] = df2["date"].dt.month
    monthly = df2.groupby("Month").size().reset_index(name="Count")
    fig_month = px.bar(monthly, x="Month", y="Count", title="Crime Count by Month")
    fig_month.update_layout(height=300)

    # Arrest Ratio
    fig_arrest = px.pie(
        df2,
        names="arrest",
        title="Arrest vs Non-Arrest Ratio"
    )
    fig_arrest.update_layout(height=300)

    # Hourly Pattern
    df2["Hour"] = df2["date"].dt.hour
    hourly = df2.groupby("Hour").size().reset_index(name="Count")
    fig_hour = px.line(hourly, x="Hour", y="Count", markers=True, title="Crime Frequency by Hour")
    fig_hour.update_layout(height=300)

    return fig_year, fig_month, fig_arrest, fig_hour


app.layout = dbc.Container([
    html.H1("Chicago Crime Analytics Dashboard", className="text-center my-4"),

    dbc.Tabs([
        dbc.Tab(overview_tab, label="Overview"),
        dbc.Tab(eda_tab, label="Exploratory Analysis"),
        dbc.Tab(hotspot_tab, label="Hotspots"),
        dbc.Tab(prediction_tab, label="ML Prediction"),
    ])
], fluid=True)


if __name__ == "__main__":
    app.run(debug=True)