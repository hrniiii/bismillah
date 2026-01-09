
import os
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.io as pio

# === Render tanpa WebGL (aman untuk server/headless) ===
pio.renderers.default = "svg"

# === Load data dengan path robust ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
]

def load_data():
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            df_local = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(
            "File CSV 'WA_Fn-UseC_-Telco-Customer-Churn.csv' tidak ditemukan "
            "di root atau folder 'data'. Pastikan nama persis sama (huruf besar/kecil)."
        )

    # --- ETL ---
    df_local["TotalCharges"] = pd.to_numeric(df_local["TotalCharges"], errors="coerce")
    df_local["ChurnFlag"] = df_local["Churn"].map({"Yes": 1, "No": 0})
    df_local.dropna(inplace=True)
    for col in ["gender", "Contract", "PaymentMethod"]:
        df_local[col] = df_local[col].str.lower().str.strip()

    return df_local

df = load_data()

# === KPI ===
churn_rate = df["ChurnFlag"].mean() * 100
avg_tenure = df["tenure"].mean()
avg_monthly = df["MonthlyCharges"].mean()
top_contract = (
    df.loc[df["Churn"] == "Yes", "Contract"].mode().iat[0].title()
    if not df.loc[df["Churn"] == "Yes", "Contract"].mode().empty
    else "N/A"
)

# === Dash App (tanpa external_stylesheets) ===
app = Dash(__name__)
server = app.server  # penting untuk Gunicorn / Railway
app.title = "Customer Churn Interactive Dashboard"
GRAPH_STYLE = {"height": "420px"}

# === Layout ===
app.layout = html.Div(
    className="page",
    children=[
        html.H2("Customer Churn Interactive Analysis Dashboard 🔍", className="dashboard-title"),

        # KPI Cards (gunakan CSS grid/flex dari file assets/style.css)
        html.Div(
            className="kpi-row",
            children=[
                html.Div(
                    className="card kpi-card",
                    children=[
                        html.Div("Churn Rate", className="kpi-title"),
                        html.Div(f"{churn_rate:.1f}%", className="kpi-value"),
                    ],
                ),
                html.Div(
                    className="card kpi-card",
                    children=[
                        html.Div("Avg Tenure", className="kpi-title"),
                        html.Div(f"{avg_tenure:.0f} months", className="kpi-value"),
                    ],
                ),
                html.Div(
                    className="card kpi-card",
                    children=[
                        html.Div("Avg Monthly Charges", className="kpi-title"),
                        html.Div(f"${avg_monthly:.2f}", className="kpi-value"),
                    ],
                ),
                html.Div(
                    className="card kpi-card",
                    children=[
                        html.Div("High Risk Contract", className="kpi-title"),
                        html.Div(top_contract, className="kpi-value"),
                    ],
                ),
            ],
        ),

        # Tabs
        dcc.Tabs(
            id="tabs",
            value="overview",
            className="custom-tabs",
            children=[
                # Tab 1: Overview
                dcc.Tab(
                    label="Overview",
                    value="overview",
                    className="tab",
                    children=[
                        html.Div(
                            className="grid-2",
                            children=[
                                html.Div(
                                    className="card",
                                    children=dcc.Graph(
                                        figure=px.pie(
                                            df,
                                            names="Churn",
                                            title="Overall Churn Distribution",
                                            color="Churn",
                                            color_discrete_map={"Yes": "#bc1313", "No": "#22c55e"},
                                        ),
                                        style=GRAPH_STYLE,
                                    ),
                                ),
                                html.Div(
                                    className="card",
                                    children=dcc.Graph(
                                        figure=px.histogram(
                                            df,
                                            x="Contract",
                                            color="Churn",
                                            barmode="group",
                                            title="Churn by Contract Type",
                                            color_discrete_map={"Yes": "#bc1313", "No": "#22c55e"},
                                        ),
                                        style=GRAPH_STYLE,
                                    ),
                                ),
                            ],
                        )
                    ],
                ),

                # Tab 2: EDA
                dcc.Tab(
                    label="EDA Explorer",
                    value="eda",
                    className="tab",
                    children=[
                        html.Div(
                            className="controls-row",
                            children=[
                                html.Div(
                                    className="control",
                                    children=[
                                        html.Label("X Variable", className="label"),
                                        dcc.Dropdown(
                                            id="eda-x",
                                            className="dropdown",
                                            options=[{"label": c, "value": c} for c in ["tenure", "MonthlyCharges", "TotalCharges"]],
                                            value="tenure",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control",
                                    children=[
                                        html.Label("Y Variable", className="label"),
                                        dcc.Dropdown(
                                            id="eda-y",
                                            className="dropdown",
                                            options=[{"label": c, "value": c} for c in ["MonthlyCharges", "TotalCharges", "ChurnFlag"]],
                                            value="MonthlyCharges",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(className="card", children=dcc.Graph(id="eda-graph")),
                    ],
                ),

                # Tab 3: Regression
                dcc.Tab(
                    label="Linear Regression",
                    value="regression",
                    className="tab",
                    children=[
                        html.Div(
                            className="controls-row",
                            children=[
                                html.Div(
                                    className="control",
                                    children=[
                                        html.Label("Independent Variable", className="label"),
                                        dcc.Dropdown(
                                            id="reg-x",
                                            className="dropdown",
                                            options=[{"label": c, "value": c} for c in ["tenure", "MonthlyCharges"]],
                                            value="tenure",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control",
                                    children=[
                                        html.Label("Dependent Variable", className="label"),
                                        dcc.Dropdown(
                                            id="reg-y",
                                            className="dropdown",
                                            options=[{"label": "TotalCharges", "value": "TotalCharges"}],
                                            value="TotalCharges",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(className="card", children=dcc.Graph(id="reg-graph")),
                        html.Div(id="reg-summary", className="insight-box"),
                    ],
                ),

                # Tab 4: Cluster
                dcc.Tab(
                    label="Customer Segmentation",
                    value="cluster",
                    className="tab",
                    children=[
                        html.Div(
                            className="controls-row",
                            children=[
                                html.Div(
                                    className="control wide",
                                    children=[
                                        html.Label("Number of Clusters (k)", className="label"),
                                        dcc.Slider(
                                            id="k-slider",
                                            min=2, max=6, step=1, value=3,
                                            marks={i: str(i) for i in range(2, 7)},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(className="card", children=dcc.Graph(id="cluster-graph")),
                    ],
                ),

                # Tab 5: Time Series
                dcc.Tab(
                    label="Time Series",
                    value="timeseries",
                    className="tab",
                    children=[html.Div(className="card", children=dcc.Graph(id="ts-graph"))],
                ),
            ],
        ),

        html.Div("Customer Churn Analysis Dashboard • Business Intelligence Project", className="footer"),
    ],
)

# === Callbacks ===
@app.callback(
    Output("eda-graph", "figure"),
    Input("eda-x", "value"),
    Input("eda-y", "value"),
)
def update_eda(x, y):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="Churn",
        title=f"{x} vs {y}",
        color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
        template="plotly_white",
    )
    return fig

@app.callback(
    Output("reg-graph", "figure"),
    Output("reg-summary", "children"),
    Input("reg-x", "value"),
    Input("reg-y", "value"),
)
def update_regression(x, y):
    X = df[[x]].values
    y_data = df[y].values
    model = LinearRegression().fit(X, y_data)
    r2 = model.score(X, y_data)

    fig = px.scatter(df, x=x, y=y, title="Linear Regression Result", template="plotly_white")
    x_sorted = np.sort(df[x].values)
    y_pred = model.predict(x_sorted.reshape(-1, 1))
    fig.add_trace(go.Scatter(x=x_sorted, y=y_pred, name="Regression line", mode="lines", line=dict(color="#1f77b4")))

    summary = html.Div(
        [
            html.P(f"💡 Setiap peningkatan 1 unit pada {x} menaikkan {y} ≈ {model.coef_[0]:.2f}."),
            html.P(f"📊 R² = {r2*100:.1f}% dari variasi dijelaskan oleh model."),
        ]
    )
    return fig, summary

@app.callback(
    Output("cluster-graph", "figure"),
    Input("k-slider", "value"),
)
def update_cluster(k):
    X = df[["tenure", "MonthlyCharges"]]
    df_cluster = df.copy()
    df_cluster["Cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    fig = px.scatter(
        df_cluster,
        x="tenure",
        y="MonthlyCharges",
        color="Cluster",
        title=f"Customer Segmentation (k={k})",
        template="plotly_white",
    )
    return fig

@app.callback(
    Output("ts-graph", "figure"),
    Input("tabs", "value"),
)
def update_ts(_tab_value):
    ts = df.groupby("tenure")["ChurnFlag"].mean().reset_index()
    fig = px.line(ts, x="tenure", y="ChurnFlag", title="Churn Rate over Tenure", template="plotly_white")
    return fig

# === Entry point lokal (Railway pakai Gunicorn) ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    app.run_server(host="0.0.0.0", port=port, debug=debug)
