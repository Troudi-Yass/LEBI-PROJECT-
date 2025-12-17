"""Dash application to visualize enriched job dataset.

Run with: `python -m src.dashboard.app` or `python src/dashboard/app.py`
"""
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from src.utils.config import CLEAN_CSV, ENRICHED_CSV, get_logger

logger = get_logger("dashboard")


def load_data(path: Path = CLEAN_CSV) -> pd.DataFrame:
    """Load data from enriched CSV if available, otherwise cleaned CSV."""
    try:
        # Prefer enriched file when available
        use_path = ENRICHED_CSV if ENRICHED_CSV.exists() else path
        df = pd.read_csv(use_path, encoding="utf-8")
        
        # Parse publication_date as datetime if it exists
        if "publication_date" in df.columns:
            df["publication_date"] = pd.to_datetime(df["publication_date"], errors='coerce')
            logger.info("✓ Parsed publication_date: %d valid dates", df["publication_date"].notna().sum())
            if df["publication_date"].notna().any():
                logger.info("  Date range: %s to %s", 
                           df["publication_date"].min(), 
                           df["publication_date"].max())
        
        # Convert salary_monthly to numeric if it exists
        if "salary_monthly" in df.columns:
            df["salary_monthly"] = pd.to_numeric(df["salary_monthly"], errors='coerce')
            logger.info("✓ Converted salary_monthly to numeric: %d valid values", df["salary_monthly"].notna().sum())
        
        logger.info("Loaded dataset from %s (%d rows)", use_path, len(df))
        return df
    except Exception as e:
        logger.error("Failed loading CSV %s: %s", path, e)
        return pd.DataFrame()


def create_app(df: pd.DataFrame) -> Dash:
    """Create and configure the Dash application."""
    app = Dash(__name__)
    
    # Extract unique values for filters
    sectors = sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns else []
    locations = sorted(df["location"].dropna().unique().tolist()) if "location" in df.columns else []
    contract_types = sorted(df["contract_type"].dropna().unique().tolist()) if "contract_type" in df.columns else []

    # Layout
    app.layout = html.Div([
        html.H1("LEBI - Job Offers Explorer", style={"textAlign": "center", "marginBottom": "30px"}),
        
        # Filters Panel
        html.Div([
            html.H3("Filters"),
            html.Label("Sector"),
            dcc.Dropdown(
                id="sector-filter", 
                options=[{"label": s, "value": s} for s in sectors], 
                multi=True,
                placeholder="Select sectors..."
            ),
            html.Br(),
            html.Label("Location"),
            dcc.Dropdown(
                id="location-filter", 
                options=[{"label": l, "value": l} for l in locations], 
                multi=True,
                placeholder="Select locations..."
            ),
            html.Br(),
            html.Label("Contract Type"),
            dcc.Dropdown(
                id="contract-filter", 
                options=[{"label": c, "value": c} for c in contract_types], 
                multi=True,
                placeholder="Select contract types..."
            ),
            html.Br(),
            html.Label("Cluster (if available)"),
            dcc.Input(
                id="cluster-filter", 
                type="number", 
                placeholder="Enter cluster ID",
                style={"width": "100%"}
            ),
            html.Br(),
            html.Br(),
            html.Label("Salary Range (monthly €)"),
            dcc.RangeSlider(
                id="salary-range", 
                min=0, 
                max=20000, 
                step=100, 
                value=[0, 20000],
                marks={0: "0€", 5000: "5k€", 10000: "10k€", 15000: "15k€", 20000: "20k€"},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={
            "width": "25%", 
            "display": "inline-block", 
            "verticalAlign": "top", 
            "padding": "20px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "5px"
        }),
        
        # Graphs Panel
        html.Div([
            dcc.Graph(id="jobs-by-sector"),
            dcc.Graph(id="salary-dist"),
            dcc.Graph(id="cluster-viz"),
            dcc.Graph(id="top-companies"),
            dcc.Graph(id="temporal-trend"),
        ], style={
            "width": "70%", 
            "display": "inline-block", 
            "padding": "20px"
        }),
    ])

    @app.callback(
        Output("jobs-by-sector", "figure"),
        Output("salary-dist", "figure"),
        Output("cluster-viz", "figure"),
        Output("top-companies", "figure"),
        Output("temporal-trend", "figure"),
        Input("sector-filter", "value"),
        Input("location-filter", "value"),
        Input("contract-filter", "value"),
        Input("cluster-filter", "value"),
        Input("salary-range", "value"),
    )
    def update(sectors_sel, locations_sel, contracts_sel, cluster_sel, salary_range):
        """Update all graphs based on filter selections."""
        dff = df.copy()
        
        # Apply filters
        if sectors_sel:
            dff = dff[dff["sector"].isin(sectors_sel)]
        if locations_sel:
            dff = dff[dff["location"].isin(locations_sel)]
        if contracts_sel:
            dff = dff[dff["contract_type"].isin(contracts_sel)]
        if cluster_sel is not None and "cluster" in dff.columns:
            try:
                cluster_sel = int(cluster_sel)
                dff = dff[dff["cluster"] == cluster_sel]
            except Exception:
                pass
        # Handle both 'cluster' and 'job_cluster' column names
        if cluster_sel is not None:
            if "job_cluster" in dff.columns:
                try:
                    cluster_sel = int(cluster_sel)
                    dff = dff[dff["job_cluster"] == cluster_sel]
                except Exception:
                    pass
            elif "cluster" in dff.columns:
                try:
                    cluster_sel = int(cluster_sel)
                    dff = dff[dff["cluster"] == cluster_sel]
                except Exception:
                    pass
        if salary_range and "salary_monthly" in dff.columns:
            # Filter only numeric salary values
            dff = dff[dff["salary_monthly"].notna()]
            if not dff.empty:
                dff = dff[(dff["salary_monthly"] >= salary_range[0]) & (dff["salary_monthly"] <= salary_range[1])]

        # Graph 1: Jobs by sector
        if "sector" in dff.columns and not dff.empty:
            sector_counts = dff["sector"].value_counts().reset_index()
            sector_counts.columns = ["sector", "count"]
            fig_sector = px.bar(
                sector_counts.head(15), 
                x="sector", 
                y="count", 
                title="Job Distribution by Sector (Top 15)",
                labels={"sector": "Sector", "count": "Number of Jobs"}
            )
            fig_sector.update_xaxes(tickangle=45)
        else:
            fig_sector = go.Figure()
            fig_sector.update_layout(title="Job Distribution by Sector (No Data)")

        # Graph 2: Salary distribution
        if "salary_monthly" in dff.columns and dff["salary_monthly"].notna().any():
            fig_salary = px.histogram(
                dff[dff["salary_monthly"].notna()], 
                x="salary_monthly", 
                nbins=50, 
                title="Salary Distribution (Monthly €)",
                labels={"salary_monthly": "Monthly Salary (€)"}
            )
        else:
            fig_salary = go.Figure()
            fig_salary.update_layout(title="Salary Distribution (No Data)")

        # Graph 3: Cluster visualization (support both 'cluster' and 'job_cluster')
        cluster_col = None
        if "job_cluster" in dff.columns:
            cluster_col = "job_cluster"
        elif "cluster" in dff.columns:
            cluster_col = "cluster"
        
        if cluster_col and "salary_monthly" in dff.columns:
            try:
                cluster_data = dff[dff[cluster_col].notna() & dff["salary_monthly"].notna()].copy()
                if not cluster_data.empty:
                    cluster_data[cluster_col] = cluster_data[cluster_col].astype(str)
                    fig_cluster = px.scatter(
                        cluster_data, 
                        x="salary_monthly", 
                        y=cluster_col, 
                        color=cluster_col,
                        title="Cluster Distribution (Salary vs Cluster)",
                        labels={"salary_monthly": "Monthly Salary (€)", cluster_col: "Cluster ID"}
                    )
                else:
                    fig_cluster = go.Figure()
                    fig_cluster.update_layout(title="Cluster Visualization (No Data)")
            except Exception as e:
                logger.warning("Cluster visualization error: %s", e)
                fig_cluster = go.Figure()
                fig_cluster.update_layout(title="Cluster Visualization (Error)")
        else:
            fig_cluster = go.Figure()
            fig_cluster.update_layout(title="Cluster Visualization (Not Available)")

        # Graph 4: Top companies
        if "company" in dff.columns and not dff.empty:
            top = dff["company"].value_counts().nlargest(10).reset_index()
            top.columns = ["company", "count"]
            fig_companies = px.bar(
                top, 
                x="company", 
                y="count", 
                title="Top 10 Companies",
                labels={"company": "Company", "count": "Number of Jobs"}
            )
            fig_companies.update_xaxes(tickangle=45)
        else:
            fig_companies = go.Figure()
            fig_companies.update_layout(title="Top Companies (No Data)")

        # Graph 5: Temporal trend (weekly job postings)
        if "publication_date" in dff.columns and dff["publication_date"].notna().any():
            try:
                dff_temporal = dff[dff["publication_date"].notna()].copy()
                dff_temporal = dff_temporal.set_index("publication_date")
                df_trend = dff_temporal.resample('W').size().reset_index(name='count')
                
                fig_temporal = px.line(
                    df_trend, 
                    x="publication_date", 
                    y="count",
                    title="Job Postings Over Time (Weekly)",
                    labels={"publication_date": "Week", "count": "Number of Jobs"},
                    markers=True
                )
                logger.info("✓ Temporal chart: %d weeks, %d total jobs", 
                           len(df_trend), df_trend['count'].sum())
            except Exception as e:
                logger.warning("Temporal trend error: %s", e)
                fig_temporal = px.line(title=f"Temporal Trend (Error: {str(e)})")
        else:
            fig_temporal = px.line(title="Temporal Trend (No Date Data)")

        return fig_sector, fig_salary, fig_cluster, fig_companies, fig_temporal

    return app


def run():
    """Load data and run the Dash application."""
    df = load_data()
    if df.empty:
        logger.error("No data available. Please run ETL pipeline first.")
        return
    app = create_app(df)
    logger.info("Starting Dash server at http://127.0.0.1:8050/")
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    run()
