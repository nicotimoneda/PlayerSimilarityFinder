"""
🔍 PLAYER SIMILARITY FINDER - STREAMLIT APP
Football Player Recommendation System powered by Machine Learning

Autor: Data Science Team
Versión: 2.0 (Production Ready)
Última actualización: Octubre 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# 🎨 CONFIGURACIÓN INICIAL Y ESTILOS
# ============================================================================

st.set_page_config(
    page_title="🔍 Player Similarity Finder",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    .main-header {
        text-align: center;
        color: #0066cc;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# 📊 FUNCIONES DE CARGA Y CACHÉ
# ============================================================================

@st.cache_resource
def load_data():
    """
    Carga el dataset con clustering.
    Busca el archivo en múltiples rutas posibles.
    Si no existe, intenta generarlo desde el raw.
    """
    # Rutas candidatas
    candidate_paths = [
        "data/processed/players_with_clusters_k6.csv",
        "../data/processed/players_with_clusters_k6.csv",
        "../../data/processed/players_with_clusters_k6.csv"
    ]

    # Buscar en rutas candidatas
    for path in candidate_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.success(f"✅ Dataset encontrado: {path}")
            return df

    # Si no existe, intenta generar desde raw
    st.warning("⚠️ Dataset procesado no encontrado. Generando desde raw...")

    raw_paths = [
        "data/raw/big5_leagues_2024_2025_stats.csv",
        "../data/raw/big5_leagues_2024_2025_stats.csv",
        "../../data/raw/big5_leagues_2024_2025_stats.csv"
    ]

    for raw_path in raw_paths:
        if os.path.exists(raw_path):
            st.info(f"📂 Leyendo desde: {raw_path}")
            df = _process_raw_data(raw_path)
            return df

    # Error final
    st.error(
        """
        ❌ No se encontró ningún archivo de datos.
        
        **Por favor:**
        1. Asegúrate de tener `data/raw/big5_leagues_2024_2025_stats.csv` en la raíz del proyecto
        2. O ejecuta primero el notebook de clustering para generar `players_with_clusters_k6.csv`
        3. Ejecuta desde la raíz del proyecto: `streamlit run src/streamlit_app.py`
        """
    )
    st.stop()

def _process_raw_data(raw_path):
    """Procesa el dataset raw si es necesario"""
    df_raw = pd.read_csv(raw_path, skiprows=1)

    # Renombrar columnas
    column_names = [
        'Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born',
        'MP', 'Starts', 'Min', '90s',
        'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR',
        'xG', 'npxG', 'xAG', 'npxG+xAG',
        'PrgC', 'PrgP', 'PrgR',
        'Gls_per90', 'Ast_per90', 'G+A_per90', 'G-PK_per90', 'G+A-PK_per90',
        'xG_per90', 'xAG_per90', 'xG+xAG_per90', 'npxG_per90', 'npxG+xAG_per90',
        'Matches'
    ]
    df_raw.columns = column_names

    # Limpiar
    df = df_raw[df_raw['Player'] != 'Player'].reset_index(drop=True)
    df = df.drop(columns=['Rk', 'Matches'])

    # Limpiar Nation y Comp
    df['Nation'] = df['Nation'].astype(str).str.extract(r'([A-Z]{3})')[0]
    df['Comp'] = df['Comp'].astype(str).apply(
        lambda x: x.split(' ', 1)[1] if ' ' in str(x) else x
    )

    # Convertir a numérico
    numeric_cols = [
        'Age', 'Born', 'MP', 'Starts', 'Min', '90s',
        'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR',
        'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR',
        'Gls_per90', 'Ast_per90', 'G+A_per90',
        'G-PK_per90', 'G+A-PK_per90',
        'xG_per90', 'xAG_per90', 'xG+xAG_per90',
        'npxG_per90', 'npxG+xAG_per90'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Agregar cluster dummy
    df['Cluster_KMeans'] = 0

    return df

@st.cache_resource
def prepare_knn_model(df):
    """Prepara el modelo KNN y el escalador"""
    similarity_features = [
        'Gls_per90', 'Ast_per90', 'G+A_per90', 'xG_per90', 'xAG_per90', 'npxG_per90',
        'PrgC', 'PrgP', 'PrgR', 'CrdY', 'CrdR', 'Age'
    ]

    X_similarity = df[similarity_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_similarity)

    knn_model = NearestNeighbors(n_neighbors=11, metric='euclidean')
    knn_model.fit(X_scaled)

    return knn_model, scaler, similarity_features, X_scaled

def find_similar_players(df, knn_model, scaler, X_scaled, player_name, n_similar=5):
    """Encuentra jugadores similares a uno dado"""

    player_matches = df[df['Player'].str.contains(player_name, case=False, na=False)]

    if len(player_matches) == 0:
        return None, None, None, None

    player_idx = player_matches.index[0]
    player_ref = df.loc[player_idx]

    distances, indices = knn_model.kneighbors(X_scaled[player_idx].reshape(1, -1))
    distances = distances[0]
    indices = indices[0]

    similar_indices = indices[1:]
    similar_distances = distances[1:]

    similar_players = df.iloc[similar_indices].copy()
    similar_players['Distance'] = similar_distances
    similar_players['Similarity_Score'] = 100 * (1 - similar_distances / similar_distances.max())

    result = similar_players.head(n_similar)

    return player_ref, result, similar_distances, similar_players

def create_radar_chart(player_ref, similar_players, n_similar=5):
    """Crea un radar chart comparativo"""

    metrics = ['Gls_per90', 'Ast_per90', 'G+A_per90', 'xG_per90', 'xAG_per90', 'PrgR']
    labels = ['Goles/90', 'Asist/90', 'G+A/90', 'xG/90', 'xAG/90', 'Prog. Recib.']
    n_metrics = len(metrics)

    all_players = pd.concat([pd.DataFrame([player_ref]), similar_players.head(n_similar)])
    stats = all_players[metrics].values

    stats_min = stats.min(axis=0)
    stats_range = np.ptp(stats, axis=0)
    stats_norm = (stats - stats_min) / (stats_range + 1e-6)

    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    stats_norm = np.concatenate([stats_norm, stats_norm[:, [0]]], axis=1)
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_players)))

    for idx, (player_name, color) in enumerate(zip(all_players['Player'], colors)):
        ax.plot(angles, stats_norm[idx], linewidth=2, label=player_name, color=color)
        ax.fill(angles, stats_norm[idx], alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.grid(True)

    plt.title("Comparativa de perfiles", fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1), fontsize=9)

    return fig

# ============================================================================
# 🎯 INTERFAZ PRINCIPAL
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">⚽ Player Similarity Finder</div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Machine Learning System for Football Player Recommendation")
    st.markdown("---")

    # Cargar datos
    with st.spinner("⏳ Cargando datos..."):
        df = load_data()
        knn_model, scaler, similarity_features, X_scaled = prepare_knn_model(df)

    # Sidebar
    st.sidebar.markdown("## ⚙️ CONTROLES")
    st.sidebar.markdown("---")

    # Búsqueda de jugador
    st.sidebar.markdown("### 🔎 Buscar Jugador")
    player_search = st.sidebar.text_input(
        "Escribe el nombre del jugador:",
        placeholder="Ej: Erling Haaland",
        help="Búsqueda parcial (sin acentos). Ej: 'Mbappe' en lugar de 'Mbappé'"
    )

    n_similar = st.sidebar.slider(
        "Número de similares a mostrar:",
        min_value=3,
        max_value=15,
        value=7,
        step=1
    )

    st.sidebar.markdown("---")

    # Información general
    st.sidebar.markdown("### 📊 INFORMACIÓN GENERAL")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Total Jugadores", len(df))
    col2.metric("Clusters", df['Cluster_KMeans'].nunique())

    col3, col4 = st.sidebar.columns(2)
    col3.metric("Ligas", df['Comp'].nunique())
    col4.metric("Equipos", df['Squad'].nunique())

    # Contenido principal
    if not player_search:
        st.info("👈 Escribe el nombre de un jugador en el panel lateral para encontrar similares")

        # Mostrar estadísticas generales
        st.markdown("## 📈 Estadísticas Generales del Dataset")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Edad Promedio", f"{df['Age'].mean():.1f} años")
        col2.metric("G+A Promedio", f"{df['G+A_per90'].mean():.3f} por 90")
        col3.metric("PrgR Promedio", f"{df['PrgR'].mean():.1f}")
        col4.metric("Minutos Promedio", f"{df['90s'].mean():.1f}")

        # Top jugadores por métrica
        st.markdown("## 🏆 Top Jugadores por Métrica")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### ⭐ Top Goleadores")
            top_scorers = df.nlargest(5, 'Gls_per90')[['Player', 'Squad', 'Gls_per90']]
            for idx, row in top_scorers.iterrows():
                st.write(f"• {row['Player']} ({row['Squad']}) - {row['Gls_per90']:.3f}")

        with col2:
            st.markdown("### 🎯 Top Asistidores")
            top_assists = df.nlargest(5, 'Ast_per90')[['Player', 'Squad', 'Ast_per90']]
            for idx, row in top_assists.iterrows():
                st.write(f"• {row['Player']} ({row['Squad']}) - {row['Ast_per90']:.3f}")

        with col3:
            st.markdown("### 🚀 Top Progresión")
            top_prog = df.nlargest(5, 'PrgR')[['Player', 'Squad', 'PrgR']]
            for idx, row in top_prog.iterrows():
                st.write(f"• {row['Player']} ({row['Squad']}) - {row['PrgR']:.0f}")

    else:
        # Búsqueda de similares
        player_ref, similar, distances, all_similar = find_similar_players(
            df, knn_model, scaler, X_scaled, player_search, n_similar
        )

        if player_ref is None:
            st.error(f"❌ No se encontró ningún jugador con '{player_search}' en su nombre")
            st.info("💡 Intenta con otro nombre o un nombre parcial")

        else:
            # Información del jugador de referencia
            st.markdown(f"## 🎯 Jugador de Referencia: **{player_ref['Player']}**")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Equipo", player_ref['Squad'])
            col2.metric("Liga", player_ref['Comp'])
            col3.metric("Posición", player_ref['Pos'])
            col4.metric("Edad", f"{player_ref['Age']:.0f} años")
            col5.metric("Cluster", f"C{player_ref['Cluster_KMeans']}")

            # Estadísticas del jugador
            st.markdown("### 📊 Perfil Estadístico")

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("G+A/90", f"{player_ref['G+A_per90']:.3f}")
            col2.metric("Goles/90", f"{player_ref['Gls_per90']:.3f}")
            col3.metric("Asist/90", f"{player_ref['Ast_per90']:.3f}")
            col4.metric("xG/90", f"{player_ref['xG_per90']:.3f}")
            col5.metric("xAG/90", f"{player_ref['xAG_per90']:.3f}")
            col6.metric("PrgR", f"{player_ref['PrgR']:.0f}")

            st.markdown("---")

            # Tabla de similares
            st.markdown(f"## 🔍 {len(similar)} Jugadores Más Similares")

            display_cols = ['Player', 'Squad', 'Comp', 'Pos', 'G+A_per90', 'Similarity_Score']
            display_df = similar[display_cols].copy()
            display_df.columns = ['Jugador', 'Equipo', 'Liga', 'Posición', 'G+A/90', 'Similitud (%)']
            display_df['Similitud (%)'] = display_df['Similitud (%)'].apply(lambda x: f"{x:.1f}%")
            display_df['G+A/90'] = display_df['G+A/90'].apply(lambda x: f"{x:.3f}")

            st.dataframe(display_df, use_container_width=True)

            st.markdown("---")

            # Visualizaciones
            st.markdown("## 📈 Visualizaciones Comparativas")

            tab1, tab2, tab3 = st.tabs(["Radar Chart", "Comparativa de Métricas", "Información Detallada"])

            with tab1:
                fig = create_radar_chart(player_ref, similar, n_similar)
                st.pyplot(fig)

            with tab2:
                # Gráfico comparativo de métricas clave
                metrics_to_plot = ['Gls_per90', 'Ast_per90', 'G+A_per90', 'xG_per90', 'PrgR']

                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                fig.suptitle(f"Comparativa de {player_ref['Player']} vs Similares",
                             fontsize=16, fontweight='bold')

                for idx, metric in enumerate(metrics_to_plot):
                    ax = axes[idx // 3, idx % 3]

                    data_to_plot = pd.concat([
                        pd.DataFrame([player_ref]),
                        similar.head(n_similar)
                    ])[['Player', metric]]

                    colors = ['#0066cc'] + ['#ff6b6b'] * len(similar)
                    ax.bar(range(len(data_to_plot)), data_to_plot[metric], color=colors)
                    ax.set_title(metric, fontweight='bold')
                    ax.set_xticklabels([p[:15] for p in data_to_plot['Player']], rotation=45, ha='right')
                    ax.grid(alpha=0.3, axis='y')

                axes[-1, -1].axis('off')

                plt.tight_layout()
                st.pyplot(fig)

            with tab3:
                st.markdown("### 📋 Detalles Completos de los Similares")

                detail_cols = ['Player', 'Squad', 'Comp', 'Pos', 'Age', 'Gls_per90',
                               'Ast_per90', 'G+A_per90', 'xG_per90', 'PrgR', 'Similarity_Score']

                detail_df = similar[detail_cols].copy()

                for col in ['Gls_per90', 'Ast_per90', 'G+A_per90', 'xG_per90']:
                    detail_df[col] = detail_df[col].apply(lambda x: f"{x:.3f}")

                detail_df['PrgR'] = detail_df['PrgR'].apply(lambda x: f"{x:.0f}")
                detail_df['Similarity_Score'] = detail_df['Similarity_Score'].apply(lambda x: f"{x:.1f}%")

                st.dataframe(detail_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 12px;'>
        🔍 Player Similarity Finder v2.0 | Built with Streamlit & Machine Learning<br>
        Data Source: FBref.com | Big 5 European Leagues 2024/2025
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 🚀 EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()