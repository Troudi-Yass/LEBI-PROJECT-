
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import os

# ==========================================
# Phase 4 - Dashboard Interactif (Dash)
# ==========================================

# 4.3 Application Dash (app.py)

# Initialisation de l'application
app = dash.Dash(__name__)

# Chargement des données enrichies
file_path = "hellowork_ml_enriched.csv"
if not os.path.exists(file_path):
    print(f"Erreur : Le fichier {file_path} est introuvable. Veuillez exécuter la Phase 3 d'abord.")
    exit(1)

df = pd.read_csv(file_path)

# Ensure Date column is datetime
if 'Publication_Date' in df.columns:
    df['Publication_Date'] = pd.to_datetime(df['Publication_Date'])
    print(f"✅ Publication_Date loaded: {df['Publication_Date'].notna().sum()} valid dates")
    print(f"   Date range: {df['Publication_Date'].min()} to {df['Publication_Date'].max()}")
else:
    print("❌ WARNING: Publication_Date column NOT FOUND in CSV!")


# Layout de l'application
app.layout = html.Div([
    html.H1("Analyse du Marché de l'Emploi - Hellowork"),
    
    dcc.Dropdown(
        id='sector_filter',
        options=[{'label': s, 'value': s} for s in df['Sector'].unique()],
        multi=True,
        placeholder="Filtrer par secteur"
    ),
    
    html.Div([
        # Row 1
        dcc.Graph(id='cluster_chart', style={'width': '49%', 'display': 'inline-block'}),
        dcc.Graph(id='salary_chart', style={'width': '49%', 'display': 'inline-block'}),
    ]),

    # Row 2 - Temporal Analysis
    html.H2("Analyse Temporelle (Tendances/Semaine)"),
    dcc.Graph(id='trend_chart')
])

# Callback pour la mise à jour des graphiques
@app.callback(
    [Output('cluster_chart', 'figure'), 
     Output('salary_chart', 'figure'),
     Output('trend_chart', 'figure')],
    [Input('sector_filter', 'value')]
)
def update_charts(sectors):
    dff = df if not sectors else df[df['Sector'].isin(sectors)]
    
    # Graphique 1 : Clusters Métiers
    fig1 = px.histogram(dff, x='Job_Cluster', 
                        title='Répartition des clusters métiers')
    
    # Graphique 2 : Classification Salariale
    fig2 = px.histogram(dff, x='Salary_Class_Pred', 
                        title='Classification salariale')

    # Graphique 3 : Tendance Temporelle
    if 'Publication_Date' in dff.columns and dff['Publication_Date'].notna().any():
        try:
            # Ensure datetime type
            dff_copy = dff.copy()
            dff_copy['Publication_Date'] = pd.to_datetime(dff_copy['Publication_Date'])
            
            # Group by week
            dff_copy = dff_copy.set_index('Publication_Date')
            df_trend = dff_copy.resample('W').size().reset_index(name='Count')
            
            print(f"📊 Temporal chart data: {len(df_trend)} weeks, {df_trend['Count'].sum()} total jobs")
            
            fig3 = px.line(df_trend, x='Publication_Date', y='Count', 
                           title="Nombre d'offres publiées par semaine", markers=True)
        except Exception as e:
            print(f"❌ Error creating temporal chart: {e}")
            fig3 = px.histogram(x=[], title=f"Erreur rendu graphique: {str(e)}")
    else:
        print(f"❌ No valid dates found. Has column: {'Publication_Date' in dff.columns}, Has data: {dff['Publication_Date'].notna().any() if 'Publication_Date' in dff.columns else False}")
        fig3 = px.histogram(x=[], title="Données temporelles non disponibles (Extraction échouée)")
    
    return fig1, fig2, fig3

# Lancement du serveur
if __name__ == '__main__':
    print("Lancement du serveur Dash sur http://127.0.0.1:8050/")
    app.run(debug=True, port=8050)
