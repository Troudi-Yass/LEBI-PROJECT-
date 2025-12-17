"""Dash application to visualize enriched job dataset.

Run with: `python -m src.dashboard.app` or `python src/dashboard/app.py`
"""
import logging
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from src.utils.config import CLEAN_CSV, ENRICHED_CSV, get_logger

logger = get_logger("dashboard")

# Modern color palette
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'accent': '#E74C3C',
    'success': '#27AE60',
    'warning': '#F39C12',
    'background': '#ECF0F1',
    'card': '#FFFFFF',
    'text': '#2C3E50',
    'border': '#BDC3C7'
}

# Chart template
CHART_TEMPLATE = 'plotly_white'


def normalize_location(value: str) -> str:
    """Normalize location labels (collapse arrondissements into city-level entries)."""
    if pd.isna(value):
        return ""

    loc = str(value).strip()
    if not loc:
        return ""

    # Collapse Paris/Lyon/Marseille arrondissements to a city-level label
    match = re.match(r"^(?P<city>Paris|Lyon|Marseille)\s+\d+(?:er|e)?\s*-\s*(?P<dept>\d{2})$", loc, flags=re.IGNORECASE)
    if match:
        city = match.group("city").title()
        dept = match.group("dept")
        return f"{city} - {dept}"

    # Normalize variant with direct city-dept
    match = re.match(r"^(?P<city>Paris|Lyon|Marseille)\s*-\s*(?P<dept>\d{2})$", loc, flags=re.IGNORECASE)
    if match:
        city = match.group("city").title()
        dept = match.group("dept")
        return f"{city} - {dept}"

    return loc


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

        # Normalize locations to reduce duplicates (e.g., Paris arrondissements)
        if "location" in df.columns:
            df["location"] = df["location"].apply(normalize_location)
            logger.info("✓ Normalized locations: %d unique", df["location"].nunique())
        
        logger.info("Loaded dataset from %s (%d rows)", use_path, len(df))
        return df
    except Exception as e:
        logger.error("Failed loading CSV %s: %s", path, e)
        return pd.DataFrame()


def create_kpi_card(title, value, icon="📊", color=COLORS['secondary']):
    """Create a KPI card component."""
    return html.Div([
        html.Div([
            html.Div(icon, style={
                'fontSize': '2.5rem',
                'marginBottom': '10px'
            }),
            html.H3(title, style={
                'margin': '10px 0',
                'fontSize': '0.9rem',
                'color': COLORS['text'],
                'fontWeight': '500',
                'textTransform': 'uppercase',
                'letterSpacing': '0.5px'
            }),
            html.H2(value, style={
                'margin': '5px 0',
                'fontSize': '2rem',
                'color': color,
                'fontWeight': 'bold'
            })
        ], style={'textAlign': 'center'})
    ], style={
        'backgroundColor': COLORS['card'],
        'padding': '25px',
        'borderRadius': '12px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
        'transition': 'transform 0.2s, box-shadow 0.2s',
        'cursor': 'pointer',
        'border': f'1px solid {COLORS["border"]}',
        'flex': '1',
        'minWidth': '200px',
        'margin': '10px'
    })


def create_app(df: pd.DataFrame) -> Dash:
    """Create and configure the Dash application."""
    app = Dash(__name__, suppress_callback_exceptions=True)
    
    # Extract unique values for filters
    sectors = sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns else []
    locations = sorted(df["location"].dropna().unique().tolist()) if "location" in df.columns else []
    contract_types = sorted(df["contract_type"].dropna().unique().tolist()) if "contract_type" in df.columns else []

    # Calculate KPIs
    total_jobs = len(df)
    avg_salary = f"€{df['salary_monthly'].mean():,.0f}" if 'salary_monthly' in df.columns and df['salary_monthly'].notna().any() else "N/A"
    total_sectors = df['sector'].nunique() if 'sector' in df.columns else 0
    total_companies = df['company'].nunique() if 'company' in df.columns else 0

    # Layout with modern styling
    app.layout = html.Div([
        # Header
        html.Div([
            html.Div([
                html.H1("🎯 LEBI Job Market Dashboard", style={
                    'color': COLORS['card'],
                    'margin': '0',
                    'fontSize': '2.5rem',
                    'fontWeight': '700',
                    'letterSpacing': '-0.5px'
                }),
                html.P("Real-time insights into job market trends and opportunities", style={
                    'color': COLORS['card'],
                    'margin': '10px 0 0 0',
                    'fontSize': '1.1rem',
                    'opacity': '0.9'
                })
            ], style={'textAlign': 'center'})
        ], style={
            'backgroundColor': COLORS['primary'],
            'padding': '40px 20px',
            'marginBottom': '30px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
        }),

        # KPI Cards
        html.Div([
            create_kpi_card("Total Jobs", f"{total_jobs:,}", "💼", COLORS['secondary']),
            create_kpi_card("Avg Salary", avg_salary, "💰", COLORS['success']),
            create_kpi_card("Sectors", total_sectors, "🏢", COLORS['warning']),
            create_kpi_card("Companies", total_companies, "🏭", COLORS['accent']),
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'justifyContent': 'center',
            'margin': '0 auto 30px auto',
            'maxWidth': '1400px',
            'padding': '0 20px'
        }),

        # Main Content Container
        html.Div([
            # Filters Sidebar
            html.Div([
                html.Div([
                    html.H3("🔍 Filters", style={
                        'color': COLORS['primary'],
                        'marginBottom': '25px',
                        'fontSize': '1.5rem',
                        'fontWeight': '600',
                        'borderBottom': f'3px solid {COLORS["secondary"]}',
                        'paddingBottom': '10px'
                    }),
                    
                    html.Div([
                        html.Label("🏢 Sector", style={'fontWeight': '600', 'color': COLORS['text'], 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Dropdown(
                            id="sector-filter", 
                            options=[{"label": s, "value": s} for s in sectors], 
                            multi=True,
                            placeholder="All sectors...",
                            style={'marginBottom': '20px'}
                        ),
                    ]),
                    
                    html.Div([
                        html.Label("📍 Location", style={'fontWeight': '600', 'color': COLORS['text'], 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Dropdown(
                            id="location-filter", 
                            options=[{"label": l, "value": l} for l in locations], 
                            multi=True,
                            placeholder="All locations...",
                            style={'marginBottom': '20px'}
                        ),
                    ]),
                    
                    html.Div([
                        html.Label("📝 Contract Type", style={'fontWeight': '600', 'color': COLORS['text'], 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Dropdown(
                            id="contract-filter", 
                            options=[{"label": c, "value": c} for c in contract_types], 
                            multi=True,
                            placeholder="All contract types...",
                            style={'marginBottom': '20px'}
                        ),
                    ]),
                    
                    html.Div([
                        html.Label("🔢 Cluster ID", style={'fontWeight': '600', 'color': COLORS['text'], 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(
                            id="cluster-filter", 
                            type="number", 
                            placeholder="Enter cluster...",
                            style={
                                'width': '100%',
                                'padding': '10px',
                                'borderRadius': '5px',
                                'border': f'1px solid {COLORS["border"]}',
                                'marginBottom': '20px'
                            }
                        ),
                    ]),
                    
                    html.Div([
                        html.Label("💵 Salary Range (monthly)", style={'fontWeight': '600', 'color': COLORS['text'], 'marginBottom': '15px', 'display': 'block'}),
                        dcc.RangeSlider(
                            id="salary-range", 
                            min=0, 
                            max=5000, 
                            step=100, 
                            value=[0, 5000],
                            marks={
                                0: {'label': '0€', 'style': {'fontSize': '0.85rem'}}, 
                                1000: {'label': '1k€', 'style': {'fontSize': '0.85rem'}}, 
                                2000: {'label': '2k€', 'style': {'fontSize': '0.85rem'}}, 
                                3000: {'label': '3k€', 'style': {'fontSize': '0.85rem'}}, 
                                4000: {'label': '4k€', 'style': {'fontSize': '0.85rem'}},
                                5000: {'label': '5k€', 'style': {'fontSize': '0.85rem'}}
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], style={'marginBottom': '30px'}),
                    
                    html.Div([
                        html.Button('🔄 Reset Filters', id='reset-btn', n_clicks=0, style={
                            'width': '100%',
                            'padding': '12px',
                            'backgroundColor': COLORS['accent'],
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '8px',
                            'fontSize': '1rem',
                            'fontWeight': '600',
                            'cursor': 'pointer',
                            'transition': 'all 0.3s',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
                        })
                    ])
                ], style={
                    'backgroundColor': COLORS['card'],
                    'padding': '30px',
                    'borderRadius': '12px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'border': f'1px solid {COLORS["border"]}',
                    'position': 'sticky',
                    'top': '20px'
                })
            ], style={
                'width': '28%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '0 15px'
            }),
            
            # Charts Panel
            html.Div([
                # Row 1: Sector and Salary charts
                html.Div([
                    html.Div([
                        dcc.Graph(id="jobs-by-sector", config={'displayModeBar': True, 'displaylogo': False})
                    ], style={
                        'width': '48%',
                        'display': 'inline-block',
                        'backgroundColor': COLORS['card'],
                        'padding': '20px',
                        'borderRadius': '12px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                        'marginRight': '4%',
                        'border': f'1px solid {COLORS["border"]}'
                    }),
                    html.Div([
                        dcc.Graph(id="salary-dist", config={'displayModeBar': True, 'displaylogo': False})
                    ], style={
                        'width': '48%',
                        'display': 'inline-block',
                        'backgroundColor': COLORS['card'],
                        'padding': '20px',
                        'borderRadius': '12px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                        'border': f'1px solid {COLORS["border"]}'
                    })
                ], style={'marginBottom': '30px'}),
                
                # Row 2: Location Distribution
                html.Div([
                    dcc.Graph(id="jobs-by-location", config={'displayModeBar': True, 'displaylogo': False})
                ], style={
                    'backgroundColor': COLORS['card'],
                    'padding': '20px',
                    'borderRadius': '12px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'marginBottom': '30px',
                    'border': f'1px solid {COLORS["border"]}'
                }),
                
                # Row 3: Cluster visualization (full width)
                html.Div([
                    dcc.Graph(id="cluster-viz", config={'displayModeBar': True, 'displaylogo': False})
                ], style={
                    'backgroundColor': COLORS['card'],
                    'padding': '20px',
                    'borderRadius': '12px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'marginBottom': '30px',
                    'border': f'1px solid {COLORS["border"]}'
                }),
                
                # Row 4: Top companies (full width)
                html.Div([
                    dcc.Graph(id="top-companies", config={'displayModeBar': True, 'displaylogo': False})
                ], style={
                    'backgroundColor': COLORS['card'],
                    'padding': '20px',
                    'borderRadius': '12px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'border': f'1px solid {COLORS["border"]}'
                })
            ], style={
                'width': '68%',
                'display': 'inline-block',
                'padding': '0 15px'
            })
        ], style={
            'maxWidth': '1600px',
            'margin': '0 auto',
            'padding': '20px'
        }),

        # Footer
        html.Div([
            html.P("© 2025 LEBI Project | Job Market Intelligence Platform", style={
                'textAlign': 'center',
                'color': COLORS['card'],
                'margin': '0',
                'fontSize': '0.9rem'
            })
        ], style={
            'backgroundColor': COLORS['primary'],
            'padding': '20px',
            'marginTop': '50px'
        })
    ], style={
        'backgroundColor': COLORS['background'],
        'minHeight': '100vh',
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    })
    # Reset button callback
    @app.callback(
        Output("sector-filter", "value"),
        Output("location-filter", "value"),
        Output("contract-filter", "value"),
        Output("cluster-filter", "value"),
        Output("salary-range", "value"),
        Input("reset-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def reset_filters(n_clicks):
        """Reset all filters to default values."""
        return [], [], [], None, [0, 5000]
    @app.callback(
        Output("jobs-by-sector", "figure"),
        Output("salary-dist", "figure"),
        Output("jobs-by-location", "figure"),
        Output("cluster-viz", "figure"),
        Output("top-companies", "figure"),
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

        # Graph 1: Jobs by sector (Enhanced)
        if "sector" in dff.columns and not dff.empty:
            sector_counts = dff["sector"].value_counts().reset_index()
            sector_counts.columns = ["sector", "count"]
            fig_sector = px.bar(
                sector_counts.head(15), 
                x="sector", 
                y="count", 
                title="📊 Job Distribution by Sector (Top 15)",
                labels={"sector": "Sector", "count": "Number of Jobs"},
                color="count",
                color_continuous_scale="Blues",
                template=CHART_TEMPLATE
            )
            fig_sector.update_traces(
                texttemplate='%{y}',
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Jobs: %{y}<extra></extra>'
            )
            fig_sector.update_xaxes(tickangle=45)
            fig_sector.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font=dict(size=16, color=COLORS['primary']),
                margin=dict(t=60, l=50, r=30, b=100)
            )
        else:
            fig_sector = go.Figure()
            fig_sector.update_layout(
                title="📊 Job Distribution by Sector (No Data)",
                template=CHART_TEMPLATE,
                annotations=[dict(text="No data available", showarrow=False, font=dict(size=14))]
            )

        # Graph 2: Salary distribution (Enhanced)
        if "salary_monthly" in dff.columns and dff["salary_monthly"].notna().any():
            salary_data = dff[dff["salary_monthly"].notna()]
            fig_salary = px.histogram(
                salary_data, 
                x="salary_monthly", 
                nbins=40, 
                title="💰 Salary Distribution",
                labels={"salary_monthly": "Monthly Salary (€)", "count": "Frequency"},
                color_discrete_sequence=[COLORS['success']],
                template=CHART_TEMPLATE
            )
            
            # Add mean and median lines
            mean_salary = salary_data["salary_monthly"].mean()
            median_salary = salary_data["salary_monthly"].median()
            
            fig_salary.add_vline(
                x=mean_salary, 
                line_dash="dash", 
                line_color=COLORS['accent'], 
                annotation_text=f"Mean: €{mean_salary:,.0f}",
                annotation_position="top"
            )
            fig_salary.add_vline(
                x=median_salary, 
                line_dash="dot", 
                line_color=COLORS['warning'], 
                annotation_text=f"Median: €{median_salary:,.0f}",
                annotation_position="bottom"
            )
            
            fig_salary.update_traces(
                hovertemplate='Salary: €%{x:,.0f}<br>Count: %{y}<extra></extra>'
            )
            fig_salary.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font=dict(size=16, color=COLORS['primary']),
                margin=dict(t=60, l=50, r=30, b=50)
            )
        else:
            fig_salary = go.Figure()
            fig_salary.update_layout(
                title="💰 Salary Distribution (No Data)",
                template=CHART_TEMPLATE,
                annotations=[dict(text="No salary data available", showarrow=False, font=dict(size=14))]
            )

        # Graph 3: Location Distribution (Enhanced)
        if "location" in dff.columns and not dff.empty:
            location_counts = dff["location"].value_counts().reset_index()
            location_counts.columns = ["location", "count"]
            
            # Take top 20 locations
            location_counts = location_counts.head(20)
            
            fig_location = px.bar(
                location_counts,
                x="count",
                y="location",
                orientation='h',
                title="📍 Top 20 Job Locations",
                labels={"location": "Location", "count": "Number of Jobs"},
                color="count",
                color_continuous_scale="Teal",
                template=CHART_TEMPLATE
            )
            fig_location.update_traces(
                texttemplate='%{x}',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Jobs: %{x}<extra></extra>'
            )
            fig_location.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font=dict(size=16, color=COLORS['primary']),
                margin=dict(t=60, l=200, r=30, b=50),
                yaxis=dict(autorange="reversed"),
                height=600
            )
        else:
            fig_location = go.Figure()
            fig_location.update_layout(
                title="📍 Job Distribution by Location (No Data)",
                template=CHART_TEMPLATE,
                annotations=[dict(text="No location data available", showarrow=False, font=dict(size=14))]
            )

        # Graph 4: Cluster visualization (Enhanced with better colors)
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
                    
                    # Create box plot instead of scatter for better visualization
                    fig_cluster = px.box(
                        cluster_data, 
                        x=cluster_col, 
                        y="salary_monthly",
                        color=cluster_col,
                        title="🎯 Salary Distribution by Cluster",
                        labels={"salary_monthly": "Monthly Salary (€)", cluster_col: "Cluster ID"},
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        template=CHART_TEMPLATE
                    )
                    fig_cluster.update_traces(
                        hovertemplate='Cluster %{x}<br>Salary: €%{y:,.0f}<extra></extra>'
                    )
                    fig_cluster.update_layout(
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12),
                        title_font=dict(size=16, color=COLORS['primary']),
                        margin=dict(t=60, l=50, r=30, b=50)
                    )
                else:
                    fig_cluster = go.Figure()
                    fig_cluster.update_layout(
                        title="🎯 Cluster Visualization (No Data)",
                        template=CHART_TEMPLATE,
                        annotations=[dict(text="No cluster data available", showarrow=False, font=dict(size=14))]
                    )
            except Exception as e:
                logger.warning("Cluster visualization error: %s", e)
                fig_cluster = go.Figure()
                fig_cluster.update_layout(
                    title="🎯 Cluster Visualization (Error)",
                    template=CHART_TEMPLATE,
                    annotations=[dict(text=f"Error: {str(e)}", showarrow=False, font=dict(size=14))]
                )
        else:
            fig_cluster = go.Figure()
            fig_cluster.update_layout(
                title="🎯 Cluster Visualization (Not Available)",
                template=CHART_TEMPLATE,
                annotations=[dict(text="Cluster data not available", showarrow=False, font=dict(size=14))]
            )

        # Graph 5: Top companies (Enhanced)
        if "company" in dff.columns and not dff.empty:
            top = dff["company"].value_counts().nlargest(10).reset_index()
            top.columns = ["company", "count"]
            fig_companies = px.bar(
                top, 
                y="company", 
                x="count",
                orientation='h',
                title="🏢 Top 10 Hiring Companies",
                labels={"company": "Company", "count": "Number of Jobs"},
                color="count",
                color_continuous_scale="Viridis",
                template=CHART_TEMPLATE
            )
            fig_companies.update_traces(
                texttemplate='%{x}',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Jobs: %{x}<extra></extra>'
            )
            fig_companies.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font=dict(size=16, color=COLORS['primary']),
                margin=dict(t=60, l=150, r=30, b=50),
                yaxis=dict(autorange="reversed")
            )
        else:
            fig_companies = go.Figure()
            fig_companies.update_layout(
                title="🏢 Top Companies (No Data)",
                template=CHART_TEMPLATE,
                annotations=[dict(text="No company data available", showarrow=False, font=dict(size=14))]
            )

        # Graph 6: Temporal trend (Enhanced with area chart)
        return fig_sector, fig_salary, fig_location, fig_cluster, fig_companies

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
