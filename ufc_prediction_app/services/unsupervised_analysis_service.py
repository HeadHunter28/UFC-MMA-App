"""
Unsupervised Analysis Service Module.

Comprehensive unsupervised machine learning analysis for UFC data including:
- Fighter style clustering (KMeans, DBSCAN, Hierarchical)
- Dimensionality reduction (PCA, t-SNE)
- Anomaly detection (Isolation Forest, LOF)
- Division pattern analysis
- Career trajectory analysis
- Fight outcome patterns
"""

import logging
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CACHE_DIR, DATA_DIR
from services.data_service import DataService

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Cache settings
ANALYSIS_CACHE_FILE = CACHE_DIR / "unsupervised_analysis_cache.json"
ANALYSIS_CACHE_TTL_HOURS = 24


class UnsupervisedAnalysisService:
    """
    Comprehensive unsupervised ML analysis service for UFC data.

    Performs clustering, dimensionality reduction, anomaly detection,
    and pattern discovery across fighters, fights, and divisions.
    """

    # Feature sets for different analyses
    STRIKING_FEATURES = [
        'sig_strikes_landed_per_min',
        'sig_strikes_absorbed_per_min',
        'sig_strike_accuracy',
        'sig_strike_defense',
    ]

    GRAPPLING_FEATURES = [
        'takedowns_avg_per_15min',
        'takedown_accuracy',
        'takedown_defense',
        'submissions_avg_per_15min',
    ]

    PHYSICAL_FEATURES = [
        'height_cm',
        'reach_cm',
        'weight_kg',
    ]

    RECORD_FEATURES = [
        'wins',
        'losses',
        'win_rate',
    ]

    # Fighter style archetypes
    STYLE_LABELS = {
        0: 'Striker',
        1: 'Grappler',
        2: 'Balanced',
        3: 'Volume Striker',
        4: 'Counter Striker',
        5: 'Wrestler',
        6: 'Submission Specialist',
    }

    def __init__(self):
        """Initialize the analysis service."""
        self.data_service = DataService()
        self.scaler = StandardScaler()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def _load_fighter_data(self) -> pd.DataFrame:
        """Load and prepare fighter data for analysis."""
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT
                    f.fighter_id, f.name, f.height_cm, f.weight_kg, f.reach_cm,
                    f.stance, f.wins, f.losses, f.draws, f.is_active,
                    fs.sig_strikes_landed_per_min, fs.sig_strikes_absorbed_per_min,
                    fs.sig_strike_accuracy, fs.sig_strike_defense,
                    fs.takedowns_avg_per_15min, fs.takedown_accuracy,
                    fs.takedown_defense, fs.submissions_avg_per_15min
                FROM fighters f
                LEFT JOIN fighter_stats fs ON f.fighter_id = fs.fighter_id
                WHERE f.wins + f.losses >= 3
            ''', conn)

        # Calculate derived features
        df['win_rate'] = df['wins'] / (df['wins'] + df['losses']).replace(0, 1)
        df['total_fights'] = df['wins'] + df['losses'] + df['draws']

        # Calculate striking vs grappling tendency
        df['striking_tendency'] = (
            df['sig_strikes_landed_per_min'].fillna(0) * 0.5 +
            df['sig_strike_accuracy'].fillna(0) * 50
        )
        df['grappling_tendency'] = (
            df['takedowns_avg_per_15min'].fillna(0) * 2 +
            df['submissions_avg_per_15min'].fillna(0) * 3
        )

        return df

    def _load_fight_data(self) -> pd.DataFrame:
        """Load fight data for analysis."""
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT
                    f.fight_id, f.event_id, f.fighter_red_id, f.fighter_blue_id,
                    f.winner_id, f.weight_class, f.is_title_fight, f.is_main_event,
                    f.method, f.round, f.time,
                    e.date as event_date, e.name as event_name
                FROM fights f
                JOIN events e ON f.event_id = e.event_id
                WHERE f.winner_id IS NOT NULL
                ORDER BY e.date
            ''', conn)

        df['event_date'] = pd.to_datetime(df['event_date'])
        df['year'] = df['event_date'].dt.year
        df['month'] = df['event_date'].dt.month

        return df

    def _load_division_data(self) -> pd.DataFrame:
        """Load division-level aggregated data."""
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT
                    weight_class,
                    COUNT(*) as total_fights,
                    SUM(CASE WHEN method LIKE '%KO%' OR method LIKE '%TKO%' THEN 1 ELSE 0 END) as ko_count,
                    SUM(CASE WHEN method LIKE '%Sub%' THEN 1 ELSE 0 END) as sub_count,
                    SUM(CASE WHEN method LIKE '%Dec%' THEN 1 ELSE 0 END) as dec_count,
                    AVG(round) as avg_rounds,
                    SUM(CASE WHEN is_title_fight THEN 1 ELSE 0 END) as title_fights
                FROM fights
                WHERE weight_class IS NOT NULL AND winner_id IS NOT NULL
                GROUP BY weight_class
                HAVING COUNT(*) >= 50
            ''', conn)

        df['ko_rate'] = df['ko_count'] / df['total_fights']
        df['sub_rate'] = df['sub_count'] / df['total_fights']
        df['dec_rate'] = df['dec_count'] / df['total_fights']
        df['finish_rate'] = (df['ko_count'] + df['sub_count']) / df['total_fights']

        return df

    # =========================================================================
    # FIGHTER STYLE CLUSTERING
    # =========================================================================

    def cluster_fighter_styles(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster fighters by fighting style using KMeans.

        Returns cluster assignments and characteristics for each cluster.
        """
        df = self._load_fighter_data()

        # Select features for clustering
        feature_cols = self.STRIKING_FEATURES + self.GRAPPLING_FEATURES
        X = df[feature_cols].copy()

        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Find optimal clusters using silhouette score
        best_score = -1
        best_k = n_clusters
        scores = {}

        for k in range(3, 8):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

        # Fit final model
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        # Analyze each cluster
        clusters = []
        for i in range(best_k):
            cluster_df = df[df['cluster'] == i]

            # Get cluster characteristics
            cluster_info = {
                'cluster_id': i,
                'size': len(cluster_df),
                'avg_sig_strikes': cluster_df['sig_strikes_landed_per_min'].mean(),
                'avg_takedowns': cluster_df['takedowns_avg_per_15min'].mean(),
                'avg_submissions': cluster_df['submissions_avg_per_15min'].mean(),
                'avg_strike_accuracy': cluster_df['sig_strike_accuracy'].mean(),
                'avg_strike_defense': cluster_df['sig_strike_defense'].mean(),
                'avg_td_defense': cluster_df['takedown_defense'].mean(),
                'avg_win_rate': cluster_df['win_rate'].mean(),
                'top_fighters': cluster_df.nlargest(5, 'wins')['name'].tolist(),
            }

            # Determine style label
            if cluster_info['avg_takedowns'] > 2.0 and cluster_info['avg_submissions'] > 0.8:
                cluster_info['style_label'] = 'Submission Specialist'
                cluster_info['description'] = 'Aggressive grapplers who hunt for submissions with high takedown and submission attempt rates.'
            elif cluster_info['avg_takedowns'] > 2.5:
                cluster_info['style_label'] = 'Wrestler'
                cluster_info['description'] = 'Wrestling-heavy fighters who dominate through takedowns and ground control.'
            elif cluster_info['avg_sig_strikes'] > 5.0:
                cluster_info['style_label'] = 'Volume Striker'
                cluster_info['description'] = 'High-output strikers who overwhelm opponents with punch volume.'
            elif cluster_info['avg_strike_defense'] > 0.55 and cluster_info['avg_sig_strikes'] < 4.0:
                cluster_info['style_label'] = 'Counter Striker'
                cluster_info['description'] = 'Patient fighters with excellent defense who pick shots carefully.'
            elif cluster_info['avg_sig_strikes'] > 3.5 and cluster_info['avg_takedowns'] < 1.5:
                cluster_info['style_label'] = 'Striker'
                cluster_info['description'] = 'Stand-up focused fighters who prefer to keep the fight on the feet.'
            elif cluster_info['avg_takedowns'] > 1.5 and cluster_info['avg_sig_strikes'] > 3.0:
                cluster_info['style_label'] = 'Balanced'
                cluster_info['description'] = 'Well-rounded fighters comfortable in all areas of MMA.'
            else:
                cluster_info['style_label'] = 'Grappler'
                cluster_info['description'] = 'Ground-focused fighters who prefer grappling exchanges.'

            clusters.append(cluster_info)

        # Get all fighters with their clusters
        fighters_by_cluster = df[['fighter_id', 'name', 'cluster', 'wins', 'losses', 'win_rate']].to_dict('records')

        return {
            'optimal_clusters': best_k,
            'silhouette_score': best_score,
            'silhouette_scores_by_k': scores,
            'clusters': clusters,
            'fighters': fighters_by_cluster,
            'insights': self._generate_cluster_insights(clusters),
        }

    def cluster_fighters_dbscan(self, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """
        Cluster fighters using DBSCAN to find natural groupings and outliers.
        """
        df = self._load_fighter_data()

        feature_cols = self.STRIKING_FEATURES + self.GRAPPLING_FEATURES
        X = df[feature_cols].copy()

        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['cluster'] = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
        n_outliers = (df['cluster'] == -1).sum()

        # Get outlier fighters (unique/unusual styles)
        outliers = df[df['cluster'] == -1][['name', 'wins', 'losses',
                                            'sig_strikes_landed_per_min',
                                            'takedowns_avg_per_15min']].to_dict('records')

        return {
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'outlier_percentage': n_outliers / len(df) * 100,
            'unique_style_fighters': outliers[:20],
            'insight': f"DBSCAN identified {n_clusters} natural fighter groupings and {n_outliers} unique-style fighters ({n_outliers/len(df)*100:.1f}%) who don't fit standard archetypes."
        }

    def hierarchical_clustering(self) -> Dict[str, Any]:
        """
        Perform hierarchical clustering to show fighter style relationships.
        """
        df = self._load_fighter_data()

        # Use top 100 fighters for visualization
        top_fighters = df.nlargest(100, 'wins').copy()

        feature_cols = self.STRIKING_FEATURES + self.GRAPPLING_FEATURES
        X = top_fighters[feature_cols].copy()

        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Compute linkage
        linkage_matrix = linkage(X_scaled, method='ward')

        # Get cluster assignments at different levels
        clusters_3 = fcluster(linkage_matrix, t=3, criterion='maxclust')
        clusters_5 = fcluster(linkage_matrix, t=5, criterion='maxclust')

        top_fighters['cluster_3'] = clusters_3
        top_fighters['cluster_5'] = clusters_5

        return {
            'linkage_matrix': linkage_matrix.tolist(),
            'fighter_names': top_fighters['name'].tolist(),
            'clusters_3_way': top_fighters[['name', 'cluster_3']].to_dict('records'),
            'clusters_5_way': top_fighters[['name', 'cluster_5']].to_dict('records'),
            'insight': "Hierarchical clustering reveals the natural taxonomy of fighting styles, showing which fighters are most similar in approach."
        }

    # =========================================================================
    # DIMENSIONALITY REDUCTION
    # =========================================================================

    def pca_analysis(self) -> Dict[str, Any]:
        """
        Perform PCA to identify the most important fighting dimensions.
        """
        df = self._load_fighter_data()

        feature_cols = self.STRIKING_FEATURES + self.GRAPPLING_FEATURES
        X = df[feature_cols].copy()

        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Fit PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Get explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # Get component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(feature_cols))],
            index=feature_cols
        )

        # Interpret components
        component_interpretations = []
        for i in range(min(3, len(feature_cols))):
            pc_loadings = loadings[f'PC{i+1}'].sort_values(key=abs, ascending=False)
            top_positive = pc_loadings[pc_loadings > 0].head(2).to_dict()
            top_negative = pc_loadings[pc_loadings < 0].head(2).to_dict()

            interpretation = {
                'component': f'PC{i+1}',
                'variance_explained': explained_variance[i],
                'top_positive_loadings': top_positive,
                'top_negative_loadings': top_negative,
            }

            # Generate interpretation
            if i == 0:
                interpretation['interpretation'] = "Overall Fighting Activity - Separates active, aggressive fighters from defensive, patient ones."
            elif i == 1:
                interpretation['interpretation'] = "Striking vs Grappling - Distinguishes stand-up specialists from ground fighters."
            else:
                interpretation['interpretation'] = "Offensive vs Defensive - Separates volume attackers from counter-fighters."

            component_interpretations.append(interpretation)

        # Store PCA coordinates for visualization
        df['pca_1'] = X_pca[:, 0]
        df['pca_2'] = X_pca[:, 1]

        # Get extreme fighters on each axis
        extremes = {
            'most_aggressive': df.nlargest(5, 'pca_1')[['name', 'pca_1']].to_dict('records'),
            'most_defensive': df.nsmallest(5, 'pca_1')[['name', 'pca_1']].to_dict('records'),
            'most_striker': df.nlargest(5, 'pca_2')[['name', 'pca_2']].to_dict('records'),
            'most_grappler': df.nsmallest(5, 'pca_2')[['name', 'pca_2']].to_dict('records'),
        }

        return {
            'explained_variance': explained_variance.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'n_components_95_variance': int(np.argmax(cumulative_variance >= 0.95) + 1),
            'loadings': loadings.to_dict(),
            'component_interpretations': component_interpretations,
            'fighter_coordinates': df[['fighter_id', 'name', 'pca_1', 'pca_2', 'wins']].to_dict('records'),
            'extremes': extremes,
            'insight': f"PCA reveals that {explained_variance[0]*100:.1f}% of fighting style variation can be explained by a single dimension (activity level), and {cumulative_variance[1]*100:.1f}% by two dimensions (activity + striking/grappling preference)."
        }

    def tsne_visualization(self, perplexity: int = 30) -> Dict[str, Any]:
        """
        Perform t-SNE for 2D visualization of fighter similarities.
        """
        df = self._load_fighter_data()

        # Use top 500 fighters for t-SNE (computational constraints)
        top_fighters = df.nlargest(500, 'total_fights').copy()

        feature_cols = self.STRIKING_FEATURES + self.GRAPPLING_FEATURES
        X = top_fighters[feature_cols].copy()

        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Fit t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)

        top_fighters['tsne_1'] = X_tsne[:, 0]
        top_fighters['tsne_2'] = X_tsne[:, 1]

        # Add style labels based on stats
        def get_style(row):
            if row['takedowns_avg_per_15min'] > 2.5:
                return 'Wrestler'
            elif row['submissions_avg_per_15min'] > 1.0:
                return 'Submission Specialist'
            elif row['sig_strikes_landed_per_min'] > 5.0:
                return 'Volume Striker'
            elif row['sig_strike_defense'] > 0.55:
                return 'Counter Striker'
            else:
                return 'Balanced'

        top_fighters['style'] = top_fighters.apply(get_style, axis=1)

        return {
            'coordinates': top_fighters[['fighter_id', 'name', 'tsne_1', 'tsne_2', 'style', 'wins']].to_dict('records'),
            'insight': "t-SNE reveals natural clusters of similar fighters. Fighters close together in this visualization share similar fighting styles and tendencies."
        }

    # =========================================================================
    # ANOMALY DETECTION
    # =========================================================================

    def detect_anomalous_fighters(self, contamination: float = 0.05) -> Dict[str, Any]:
        """
        Detect fighters with unusual/anomalous statistical profiles using Isolation Forest.
        """
        df = self._load_fighter_data()

        feature_cols = self.STRIKING_FEATURES + self.GRAPPLING_FEATURES + ['win_rate']
        X = df[feature_cols].copy()

        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df['anomaly_score'] = iso_forest.fit_predict(X_scaled)
        df['anomaly_score_raw'] = iso_forest.decision_function(X_scaled)

        # Get anomalous fighters
        anomalies = df[df['anomaly_score'] == -1].copy()
        anomalies = anomalies.sort_values('anomaly_score_raw')

        # Analyze why they're anomalous
        anomaly_details = []
        for _, fighter in anomalies.head(15).iterrows():
            reasons = []

            if fighter['sig_strikes_landed_per_min'] > df['sig_strikes_landed_per_min'].quantile(0.95):
                reasons.append('Exceptionally high strike output')
            if fighter['sig_strikes_landed_per_min'] < df['sig_strikes_landed_per_min'].quantile(0.05):
                reasons.append('Unusually low strike output')
            if fighter['takedowns_avg_per_15min'] > df['takedowns_avg_per_15min'].quantile(0.95):
                reasons.append('Elite takedown rate')
            if fighter['submissions_avg_per_15min'] > df['submissions_avg_per_15min'].quantile(0.95):
                reasons.append('Exceptional submission threat')
            if fighter['sig_strike_defense'] > df['sig_strike_defense'].quantile(0.95):
                reasons.append('Outstanding striking defense')
            if fighter['win_rate'] > 0.85:
                reasons.append('Dominant win rate')
            if fighter['win_rate'] < 0.3:
                reasons.append('Poor win rate despite UFC tenure')

            anomaly_details.append({
                'name': fighter['name'],
                'wins': int(fighter['wins']),
                'losses': int(fighter['losses']),
                'win_rate': fighter['win_rate'],
                'reasons': reasons if reasons else ['Unique combination of stats'],
                'anomaly_score': fighter['anomaly_score_raw'],
            })

        return {
            'total_anomalies': len(anomalies),
            'anomaly_percentage': len(anomalies) / len(df) * 100,
            'anomalous_fighters': anomaly_details,
            'insight': f"Isolation Forest identified {len(anomalies)} fighters ({len(anomalies)/len(df)*100:.1f}%) with statistically unusual profiles. These fighters don't fit typical fighting patterns."
        }

    def detect_local_outliers(self) -> Dict[str, Any]:
        """
        Detect local outliers using LOF - fighters unusual within their peer group.
        """
        df = self._load_fighter_data()

        feature_cols = self.STRIKING_FEATURES + self.GRAPPLING_FEATURES
        X = df[feature_cols].copy()

        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        df['lof_label'] = lof.fit_predict(X_scaled)
        df['lof_score'] = -lof.negative_outlier_factor_

        outliers = df[df['lof_label'] == -1].nlargest(15, 'lof_score')

        return {
            'n_local_outliers': (df['lof_label'] == -1).sum(),
            'local_outliers': outliers[['name', 'wins', 'losses', 'lof_score']].to_dict('records'),
            'insight': "Local Outlier Factor finds fighters who are unusual compared to their most similar peers, revealing unique adaptations within fighting styles."
        }

    # =========================================================================
    # DIVISION PATTERN ANALYSIS
    # =========================================================================

    def analyze_division_patterns(self) -> Dict[str, Any]:
        """
        Cluster and analyze patterns across weight divisions.
        """
        df = self._load_division_data()

        feature_cols = ['ko_rate', 'sub_rate', 'dec_rate', 'avg_rounds', 'finish_rate']
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        # Cluster divisions
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        # Analyze each cluster
        cluster_analysis = []
        for i in range(3):
            cluster_df = df[df['cluster'] == i]

            avg_ko = cluster_df['ko_rate'].mean()
            avg_sub = cluster_df['sub_rate'].mean()
            avg_finish = cluster_df['finish_rate'].mean()

            if avg_ko > 0.35:
                label = "Power Divisions"
                desc = "High knockout rate divisions where power and striking dominate."
            elif avg_sub > 0.20:
                label = "Technical Divisions"
                desc = "Divisions with higher submission rates indicating technical grappling."
            else:
                label = "Competitive Divisions"
                desc = "Well-matched divisions where fights often go to decision."

            cluster_analysis.append({
                'cluster_id': i,
                'label': label,
                'description': desc,
                'divisions': cluster_df['weight_class'].tolist(),
                'avg_ko_rate': avg_ko,
                'avg_sub_rate': avg_sub,
                'avg_finish_rate': avg_finish,
            })

        # Division rankings
        rankings = {
            'most_finishes': df.nlargest(3, 'finish_rate')[['weight_class', 'finish_rate']].to_dict('records'),
            'most_decisions': df.nlargest(3, 'dec_rate')[['weight_class', 'dec_rate']].to_dict('records'),
            'most_kos': df.nlargest(3, 'ko_rate')[['weight_class', 'ko_rate']].to_dict('records'),
            'most_submissions': df.nlargest(3, 'sub_rate')[['weight_class', 'sub_rate']].to_dict('records'),
        }

        return {
            'division_clusters': cluster_analysis,
            'rankings': rankings,
            'division_data': df.to_dict('records'),
            'insight': "Weight divisions cluster into distinct patterns. Heavier divisions favor knockouts, lighter divisions show more technical variety, and women's divisions often show higher finish rates."
        }

    # =========================================================================
    # TEMPORAL PATTERN ANALYSIS
    # =========================================================================

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analyze how fighting patterns have evolved over time.
        """
        df = self._load_fight_data()

        # Aggregate by year
        yearly = df.groupby('year').agg({
            'fight_id': 'count',
            'method': lambda x: (x.str.contains('KO|TKO', na=False)).sum() / len(x) * 100,
            'round': 'mean',
            'is_title_fight': 'sum',
        }).rename(columns={
            'fight_id': 'total_fights',
            'method': 'ko_percentage',
            'round': 'avg_rounds',
            'is_title_fight': 'title_fights'
        })

        # Calculate submission rate separately
        yearly['sub_percentage'] = df.groupby('year')['method'].apply(
            lambda x: (x.str.contains('Sub', case=False, na=False)).sum() / len(x) * 100
        )

        # Calculate decision rate
        yearly['dec_percentage'] = df.groupby('year')['method'].apply(
            lambda x: (x.str.contains('Dec', case=False, na=False)).sum() / len(x) * 100
        )

        yearly = yearly.reset_index()

        # Trend analysis
        recent_years = yearly[yearly['year'] >= 2018]
        ko_trend = stats.linregress(recent_years['year'], recent_years['ko_percentage'])
        sub_trend = stats.linregress(recent_years['year'], recent_years['sub_percentage'])

        trends = {
            'ko_trend': 'increasing' if ko_trend.slope > 0.5 else ('decreasing' if ko_trend.slope < -0.5 else 'stable'),
            'ko_slope': ko_trend.slope,
            'sub_trend': 'increasing' if sub_trend.slope > 0.5 else ('decreasing' if sub_trend.slope < -0.5 else 'stable'),
            'sub_slope': sub_trend.slope,
        }

        # Find era transitions
        era_analysis = []
        if len(yearly) > 0:
            # Early UFC era
            early = yearly[yearly['year'] < 2010]
            if len(early) > 0:
                era_analysis.append({
                    'era': 'Early UFC (Pre-2010)',
                    'avg_ko_rate': early['ko_percentage'].mean(),
                    'avg_sub_rate': early['sub_percentage'].mean(),
                    'characteristics': 'Higher finish rates, less refined techniques'
                })

            # Modern era
            modern = yearly[(yearly['year'] >= 2010) & (yearly['year'] < 2020)]
            if len(modern) > 0:
                era_analysis.append({
                    'era': 'Modern Era (2010-2019)',
                    'avg_ko_rate': modern['ko_percentage'].mean(),
                    'avg_sub_rate': modern['sub_percentage'].mean(),
                    'characteristics': 'Rise of wrestling-heavy fighters, improved TDD'
                })

            # Current era
            current = yearly[yearly['year'] >= 2020]
            if len(current) > 0:
                era_analysis.append({
                    'era': 'Current Era (2020+)',
                    'avg_ko_rate': current['ko_percentage'].mean(),
                    'avg_sub_rate': current['sub_percentage'].mean(),
                    'characteristics': 'Well-rounded athletes, technical striking improvements'
                })

        return {
            'yearly_data': yearly.to_dict('records'),
            'trends': trends,
            'era_analysis': era_analysis,
            'insight': f"UFC fighting patterns have evolved significantly. KO rates are {trends['ko_trend']} while submission rates are {trends['sub_trend']}. Modern fighters are more well-rounded than ever."
        }

    def analyze_monthly_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns by month/season.
        """
        df = self._load_fight_data()

        monthly = df.groupby('month').agg({
            'fight_id': 'count',
            'is_title_fight': 'sum',
        }).rename(columns={
            'fight_id': 'total_fights',
            'is_title_fight': 'title_fights'
        }).reset_index()

        monthly['title_fight_rate'] = monthly['title_fights'] / monthly['total_fights'] * 100

        # Find peak months
        peak_month = monthly.loc[monthly['total_fights'].idxmax()]

        return {
            'monthly_data': monthly.to_dict('records'),
            'peak_month': int(peak_month['month']),
            'peak_fights': int(peak_month['total_fights']),
            'insight': f"Month {int(peak_month['month'])} historically has the most UFC events. Big PPV months (March, July, December) often feature more title fights."
        }

    # =========================================================================
    # FIGHT OUTCOME PATTERNS
    # =========================================================================

    def analyze_finish_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in how fights end.
        """
        df = self._load_fight_data()

        # Round distribution for finishes
        ko_fights = df[df['method'].str.contains('KO|TKO', na=False, case=False)]
        sub_fights = df[df['method'].str.contains('Sub', na=False, case=False)]

        ko_by_round = ko_fights['round'].value_counts().sort_index().to_dict()
        sub_by_round = sub_fights['round'].value_counts().sort_index().to_dict()

        # Calculate probabilities
        total_by_round = df['round'].value_counts().sort_index()

        ko_prob_by_round = {}
        sub_prob_by_round = {}

        for r in [1, 2, 3, 4, 5]:
            if r in total_by_round.index:
                ko_prob_by_round[r] = ko_by_round.get(r, 0) / total_by_round[r] * 100
                sub_prob_by_round[r] = sub_by_round.get(r, 0) / total_by_round[r] * 100

        # Title vs non-title finish rates
        title_fights = df[df['is_title_fight'] == True]
        non_title = df[df['is_title_fight'] == False]

        title_finish_rate = (
            title_fights['method'].str.contains('KO|TKO|Sub', na=False, case=False).sum() /
            len(title_fights) * 100
        ) if len(title_fights) > 0 else 0

        non_title_finish_rate = (
            non_title['method'].str.contains('KO|TKO|Sub', na=False, case=False).sum() /
            len(non_title) * 100
        ) if len(non_title) > 0 else 0

        return {
            'ko_by_round': ko_by_round,
            'sub_by_round': sub_by_round,
            'ko_probability_by_round': ko_prob_by_round,
            'sub_probability_by_round': sub_prob_by_round,
            'title_fight_finish_rate': title_finish_rate,
            'non_title_finish_rate': non_title_finish_rate,
            'insights': [
                f"Round 1 has the highest KO probability ({ko_prob_by_round.get(1, 0):.1f}%)",
                f"Submissions are most common in rounds 2-3",
                f"Title fights have a {title_finish_rate:.1f}% finish rate vs {non_title_finish_rate:.1f}% for non-title fights",
            ]
        }

    def analyze_upset_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in upsets (when less experienced fighter wins).
        """
        df = self._load_fight_data()
        fighter_df = self._load_fighter_data()

        # Join fighter data
        df = df.merge(
            fighter_df[['fighter_id', 'wins', 'losses', 'win_rate']],
            left_on='fighter_red_id',
            right_on='fighter_id',
            suffixes=('', '_red')
        ).merge(
            fighter_df[['fighter_id', 'wins', 'losses', 'win_rate']],
            left_on='fighter_blue_id',
            right_on='fighter_id',
            suffixes=('_red', '_blue')
        )

        # Define upset (underdog wins based on record)
        df['favorite'] = df.apply(
            lambda x: 'red' if x['wins_red'] > x['wins_blue'] else 'blue',
            axis=1
        )
        df['upset'] = df.apply(
            lambda x: (x['favorite'] == 'red' and x['winner_id'] == x['fighter_blue_id']) or
                      (x['favorite'] == 'blue' and x['winner_id'] == x['fighter_red_id']),
            axis=1
        )

        upset_rate = df['upset'].mean() * 100

        # Upset rate by division
        upset_by_division = df.groupby('weight_class')['upset'].mean().sort_values(ascending=False) * 100

        # Upset rate by year
        upset_by_year = df.groupby('year')['upset'].mean() * 100

        return {
            'overall_upset_rate': upset_rate,
            'upset_by_division': upset_by_division.to_dict(),
            'upset_by_year': upset_by_year.to_dict(),
            'most_upset_prone_division': upset_by_division.idxmax() if len(upset_by_division) > 0 else None,
            'insight': f"Overall upset rate is {upset_rate:.1f}%. The most unpredictable division is {upset_by_division.idxmax() if len(upset_by_division) > 0 else 'Unknown'} with a {upset_by_division.max():.1f}% upset rate."
        }

    # =========================================================================
    # CAREER TRAJECTORY ANALYSIS
    # =========================================================================

    def analyze_career_trajectories(self) -> Dict[str, Any]:
        """
        Cluster fighters by career trajectory patterns.
        """
        df = self._load_fighter_data()

        # Calculate career metrics
        df['career_length'] = df['total_fights']
        df['prime_indicator'] = df['win_rate'] * np.log1p(df['total_fights'])

        # Cluster by career pattern
        career_features = ['wins', 'losses', 'win_rate', 'total_fights']
        X = df[career_features].copy()

        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['career_cluster'] = kmeans.fit_predict(X_scaled)

        # Analyze career archetypes
        career_types = []
        for i in range(4):
            cluster_df = df[df['career_cluster'] == i]

            avg_wins = cluster_df['wins'].mean()
            avg_losses = cluster_df['losses'].mean()
            avg_win_rate = cluster_df['win_rate'].mean()
            avg_fights = cluster_df['total_fights'].mean()

            if avg_win_rate > 0.7 and avg_fights > 10:
                label = "Elite Champions"
                desc = "Dominant fighters with long, successful careers."
            elif avg_win_rate > 0.6:
                label = "Solid Veterans"
                desc = "Consistent performers who compete at a high level."
            elif avg_fights > 15:
                label = "UFC Journeymen"
                desc = "Long-tenured fighters with mixed results but staying power."
            else:
                label = "Prospects/Newcomers"
                desc = "Early career fighters still establishing themselves."

            career_types.append({
                'cluster_id': i,
                'label': label,
                'description': desc,
                'size': len(cluster_df),
                'avg_wins': avg_wins,
                'avg_losses': avg_losses,
                'avg_win_rate': avg_win_rate,
                'avg_fights': avg_fights,
                'top_fighters': cluster_df.nlargest(5, 'wins')['name'].tolist(),
            })

        return {
            'career_archetypes': career_types,
            'insight': "Fighter careers follow distinct patterns. Elite champions often have 70%+ win rates over 10+ fights, while journeymen may have 20+ fights with closer to 50% win rates."
        }

    # =========================================================================
    # AGE ANALYSIS
    # =========================================================================

    def analyze_age_performance(self) -> Dict[str, Any]:
        """
        Analyze how age affects fighter performance and identify peak age.
        """
        with self.data_service.get_connection() as conn:
            # Get fighters with DOB
            df = pd.read_sql('''
                SELECT f.fighter_id, f.name, f.dob, f.wins, f.losses,
                       fs.sig_strikes_landed_per_min, fs.sig_strike_accuracy,
                       fs.takedowns_avg_per_15min, fs.sig_strike_defense
                FROM fighters f
                LEFT JOIN fighter_stats fs ON f.fighter_id = fs.fighter_id
                WHERE f.dob IS NOT NULL AND f.wins + f.losses >= 3
            ''', conn)

        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df = df.dropna(subset=['dob'])
        df['age'] = (datetime.now() - df['dob']).dt.days / 365.25
        df['win_rate'] = df['wins'] / (df['wins'] + df['losses']).replace(0, 1)

        # Age brackets
        age_brackets = {
            '21-25': (21, 25),
            '26-30': (26, 30),
            '31-35': (31, 35),
            '36-40': (36, 40),
            '40+': (40, 100),
        }

        age_stats = []
        for bracket, (min_age, max_age) in age_brackets.items():
            bracket_df = df[(df['age'] >= min_age) & (df['age'] < max_age)]
            if len(bracket_df) > 10:
                age_stats.append({
                    'bracket': bracket,
                    'count': len(bracket_df),
                    'avg_win_rate': bracket_df['win_rate'].mean(),
                    'avg_strikes_per_min': bracket_df['sig_strikes_landed_per_min'].mean(),
                    'avg_strike_accuracy': bracket_df['sig_strike_accuracy'].mean(),
                    'avg_takedowns': bracket_df['takedowns_avg_per_15min'].mean(),
                })

        # Find peak age
        peak_bracket = max(age_stats, key=lambda x: x['avg_win_rate']) if age_stats else None

        # Correlation analysis
        age_win_corr = df[['age', 'win_rate']].corr().iloc[0, 1]
        age_strike_corr = df[['age', 'sig_strikes_landed_per_min']].dropna().corr().iloc[0, 1]

        # Cluster fighters by age and performance
        age_performance_df = df[['age', 'win_rate', 'sig_strikes_landed_per_min']].dropna()
        if len(age_performance_df) > 50:
            X = self.scaler.fit_transform(age_performance_df)
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            age_performance_df['cluster'] = kmeans.fit_predict(X)

            cluster_profiles = []
            for i in range(4):
                cluster_df = age_performance_df[age_performance_df['cluster'] == i]
                avg_age = cluster_df['age'].mean()
                avg_wr = cluster_df['win_rate'].mean()

                if avg_age < 28 and avg_wr > 0.6:
                    label = "Rising Stars"
                elif avg_age > 35 and avg_wr > 0.55:
                    label = "Veteran Elite"
                elif avg_age < 30 and avg_wr < 0.5:
                    label = "Developing Prospects"
                elif avg_age > 33 and avg_wr < 0.5:
                    label = "Declining Veterans"
                else:
                    label = "Prime Competitors"

                cluster_profiles.append({
                    'cluster': i,
                    'label': label,
                    'avg_age': avg_age,
                    'avg_win_rate': avg_wr,
                    'size': len(cluster_df),
                })
        else:
            cluster_profiles = []

        return {
            'age_bracket_stats': age_stats,
            'peak_age_bracket': peak_bracket['bracket'] if peak_bracket else 'Unknown',
            'peak_win_rate': peak_bracket['avg_win_rate'] if peak_bracket else 0,
            'age_win_correlation': age_win_corr,
            'age_strike_correlation': age_strike_corr,
            'cluster_profiles': cluster_profiles,
            'avg_fighter_age': df['age'].mean(),
            'median_fighter_age': df['age'].median(),
            'insights': [
                f"Peak performance age bracket: {peak_bracket['bracket'] if peak_bracket else 'Unknown'} with {peak_bracket['avg_win_rate']*100:.1f}% win rate" if peak_bracket else "Insufficient age data",
                f"Age-win rate correlation: {age_win_corr:.3f} ({'negative' if age_win_corr < 0 else 'positive'} - {'older fighters tend to have lower win rates' if age_win_corr < 0 else 'experience helps'})",
                f"Average UFC fighter age: {df['age'].mean():.1f} years",
            ]
        }

    def analyze_age_by_division(self) -> Dict[str, Any]:
        """
        Analyze how age affects performance differently across weight classes.
        """
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT f.fighter_id, f.name, f.dob, f.wins, f.losses,
                       fi.weight_class
                FROM fighters f
                JOIN fights fi ON (f.fighter_id = fi.fighter_red_id OR f.fighter_id = fi.fighter_blue_id)
                WHERE f.dob IS NOT NULL
                GROUP BY f.fighter_id
            ''', conn)

        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df = df.dropna(subset=['dob'])
        df['age'] = (datetime.now() - df['dob']).dt.days / 365.25
        df['win_rate'] = df['wins'] / (df['wins'] + df['losses']).replace(0, 1)

        # Analyze by weight class
        division_age_stats = []
        for wc in df['weight_class'].dropna().unique():
            wc_df = df[df['weight_class'] == wc]
            if len(wc_df) >= 20:
                # Find optimal age for this division
                young = wc_df[wc_df['age'] < 30]['win_rate'].mean()
                prime = wc_df[(wc_df['age'] >= 30) & (wc_df['age'] < 35)]['win_rate'].mean()
                veteran = wc_df[wc_df['age'] >= 35]['win_rate'].mean()

                if young > prime and young > veteran:
                    favors = "Young (under 30)"
                elif prime > young and prime > veteran:
                    favors = "Prime (30-35)"
                else:
                    favors = "Experience (35+)"

                division_age_stats.append({
                    'weight_class': wc,
                    'avg_age': wc_df['age'].mean(),
                    'young_win_rate': young,
                    'prime_win_rate': prime,
                    'veteran_win_rate': veteran,
                    'favors_age': favors,
                })

        # Find which divisions favor youth vs experience
        youth_divisions = [d for d in division_age_stats if 'Young' in d.get('favors_age', '')]
        experience_divisions = [d for d in division_age_stats if 'Experience' in d.get('favors_age', '')]

        return {
            'division_age_stats': division_age_stats,
            'youth_favoring_divisions': [d['weight_class'] for d in youth_divisions],
            'experience_favoring_divisions': [d['weight_class'] for d in experience_divisions],
            'insight': f"Age matters differently across divisions. {len(youth_divisions)} divisions favor younger fighters while {len(experience_divisions)} divisions reward experience."
        }

    # =========================================================================
    # HEIGHT AND REACH ANALYSIS
    # =========================================================================

    def analyze_height_reach_advantage(self) -> Dict[str, Any]:
        """
        Analyze how height and reach advantages affect fight outcomes.
        """
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT
                    f.fight_id, f.winner_id, f.method, f.weight_class,
                    fr.height_cm as red_height, fr.reach_cm as red_reach,
                    fb.height_cm as blue_height, fb.reach_cm as blue_reach,
                    f.fighter_red_id, f.fighter_blue_id
                FROM fights f
                JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
                JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
                WHERE f.winner_id IS NOT NULL
                  AND fr.height_cm IS NOT NULL AND fb.height_cm IS NOT NULL
                  AND fr.reach_cm IS NOT NULL AND fb.reach_cm IS NOT NULL
            ''', conn)

        # Calculate advantages
        df['height_diff'] = df['red_height'] - df['blue_height']
        df['reach_diff'] = df['red_reach'] - df['blue_reach']
        df['red_won'] = df['winner_id'] == df['fighter_red_id']

        # Height advantage analysis
        df['taller_won'] = (
            ((df['height_diff'] > 0) & df['red_won']) |
            ((df['height_diff'] < 0) & ~df['red_won'])
        )
        df['has_height_advantage'] = df['height_diff'].abs() > 2  # 2cm threshold

        height_fights = df[df['has_height_advantage']]
        taller_win_rate = height_fights['taller_won'].mean() * 100 if len(height_fights) > 0 else 50

        # Reach advantage analysis
        df['longer_reach_won'] = (
            ((df['reach_diff'] > 0) & df['red_won']) |
            ((df['reach_diff'] < 0) & ~df['red_won'])
        )
        df['has_reach_advantage'] = df['reach_diff'].abs() > 5  # 5cm threshold

        reach_fights = df[df['has_reach_advantage']]
        reach_win_rate = reach_fights['longer_reach_won'].mean() * 100 if len(reach_fights) > 0 else 50

        # Reach advantage by method
        reach_ko_fights = reach_fights[reach_fights['method'].str.contains('KO|TKO', na=False, case=False)]
        reach_ko_win_rate = reach_ko_fights['longer_reach_won'].mean() * 100 if len(reach_ko_fights) > 0 else 50

        reach_sub_fights = reach_fights[reach_fights['method'].str.contains('Sub', na=False, case=False)]
        reach_sub_win_rate = reach_sub_fights['longer_reach_won'].mean() * 100 if len(reach_sub_fights) > 0 else 50

        # By division
        reach_by_division = {}
        for wc in df['weight_class'].dropna().unique():
            wc_df = df[(df['weight_class'] == wc) & (df['has_reach_advantage'])]
            if len(wc_df) >= 20:
                reach_by_division[wc] = wc_df['longer_reach_won'].mean() * 100

        # Optimal reach advantage (clustering)
        reach_advantages = df[df['reach_diff'].abs() > 0][['reach_diff', 'red_won']].copy()
        reach_advantages['reach_diff_abs'] = reach_advantages['reach_diff'].abs()
        reach_advantages['advantage_won'] = (
            ((reach_advantages['reach_diff'] > 0) & reach_advantages['red_won']) |
            ((reach_advantages['reach_diff'] < 0) & ~reach_advantages['red_won'])
        )

        # Group by reach advantage size
        reach_brackets = {
            '0-5cm': (0, 5),
            '5-10cm': (5, 10),
            '10-15cm': (10, 15),
            '15+cm': (15, 100),
        }

        reach_bracket_stats = []
        for bracket, (min_r, max_r) in reach_brackets.items():
            bracket_df = reach_advantages[(reach_advantages['reach_diff_abs'] >= min_r) &
                                          (reach_advantages['reach_diff_abs'] < max_r)]
            if len(bracket_df) >= 10:
                reach_bracket_stats.append({
                    'bracket': bracket,
                    'win_rate': bracket_df['advantage_won'].mean() * 100,
                    'sample_size': len(bracket_df),
                })

        return {
            'taller_fighter_win_rate': taller_win_rate,
            'longer_reach_win_rate': reach_win_rate,
            'reach_advantage_in_kos': reach_ko_win_rate,
            'reach_advantage_in_subs': reach_sub_win_rate,
            'reach_by_division': reach_by_division,
            'reach_bracket_stats': reach_bracket_stats,
            'total_fights_analyzed': len(df),
            'insights': [
                f"Taller fighters win {taller_win_rate:.1f}% of fights with significant height difference",
                f"Longer reach wins {reach_win_rate:.1f}% of fights with 5cm+ reach advantage",
                f"Reach advantage is {'more' if reach_ko_win_rate > reach_sub_win_rate else 'less'} valuable for KOs ({reach_ko_win_rate:.1f}%) than submissions ({reach_sub_win_rate:.1f}%)",
            ]
        }

    def cluster_by_physical_attributes(self) -> Dict[str, Any]:
        """
        Cluster fighters by physical attributes and analyze success patterns.
        """
        df = self._load_fighter_data()

        # Filter for fighters with physical data
        phys_df = df[['fighter_id', 'name', 'height_cm', 'reach_cm', 'weight_kg', 'wins', 'losses', 'win_rate']].dropna()

        if len(phys_df) < 50:
            return {'error': 'Insufficient physical data'}

        # Normalize physical attributes
        X = phys_df[['height_cm', 'reach_cm', 'weight_kg']].values
        X_scaled = self.scaler.fit_transform(X)

        # Cluster
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        phys_df['cluster'] = kmeans.fit_predict(X_scaled)

        # Analyze clusters
        physical_archetypes = []
        for i in range(4):
            cluster_df = phys_df[phys_df['cluster'] == i]
            avg_height = cluster_df['height_cm'].mean()
            avg_reach = cluster_df['reach_cm'].mean()
            avg_weight = cluster_df['weight_kg'].mean()
            avg_wr = cluster_df['win_rate'].mean()

            # Determine archetype
            if avg_height > 185 and avg_reach > 190:
                label = "Long & Tall"
                desc = "Fighters who use length to control distance"
            elif avg_weight > 90 and avg_height < 180:
                label = "Compact Powerhouse"
                desc = "Dense, powerful fighters"
            elif avg_reach / avg_height > 1.05:
                label = "Reach Specialist"
                desc = "Fighters with disproportionately long reach"
            else:
                label = "Average Build"
                desc = "Standard physical proportions"

            physical_archetypes.append({
                'cluster': i,
                'label': label,
                'description': desc,
                'avg_height_cm': avg_height,
                'avg_reach_cm': avg_reach,
                'avg_weight_kg': avg_weight,
                'reach_to_height_ratio': avg_reach / avg_height,
                'avg_win_rate': avg_wr,
                'size': len(cluster_df),
                'top_fighters': cluster_df.nlargest(5, 'wins')['name'].tolist(),
            })

        # Find most successful archetype
        best_archetype = max(physical_archetypes, key=lambda x: x['avg_win_rate'])

        return {
            'physical_archetypes': physical_archetypes,
            'best_archetype': best_archetype['label'],
            'best_archetype_win_rate': best_archetype['avg_win_rate'],
            'insight': f"'{best_archetype['label']}' fighters have the highest win rate at {best_archetype['avg_win_rate']*100:.1f}%"
        }

    # =========================================================================
    # WEIGHT CLASS DEEP ANALYSIS
    # =========================================================================

    def analyze_weight_class_careers(self) -> Dict[str, Any]:
        """
        Analyze career longevity and patterns by weight class.
        """
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT
                    f.fighter_id, f.name, f.wins, f.losses, f.dob,
                    fi.weight_class,
                    MIN(e.date) as first_fight,
                    MAX(e.date) as last_fight,
                    COUNT(DISTINCT fi.fight_id) as ufc_fights
                FROM fighters f
                JOIN fights fi ON (f.fighter_id = fi.fighter_red_id OR f.fighter_id = fi.fighter_blue_id)
                JOIN events e ON fi.event_id = e.event_id
                WHERE fi.weight_class IS NOT NULL
                GROUP BY f.fighter_id, fi.weight_class
                HAVING ufc_fights >= 3
            ''', conn)

        df['first_fight'] = pd.to_datetime(df['first_fight'])
        df['last_fight'] = pd.to_datetime(df['last_fight'])
        df['career_span_days'] = (df['last_fight'] - df['first_fight']).dt.days
        df['career_span_years'] = df['career_span_days'] / 365.25
        df['win_rate'] = df['wins'] / (df['wins'] + df['losses']).replace(0, 1)

        # Analyze by weight class
        division_careers = []
        for wc in df['weight_class'].dropna().unique():
            wc_df = df[df['weight_class'] == wc]
            if len(wc_df) >= 20:
                division_careers.append({
                    'weight_class': wc,
                    'avg_career_years': wc_df['career_span_years'].mean(),
                    'avg_ufc_fights': wc_df['ufc_fights'].mean(),
                    'avg_win_rate': wc_df['win_rate'].mean(),
                    'median_career_years': wc_df['career_span_years'].median(),
                    'fighters_count': len(wc_df),
                })

        # Sort by career length
        division_careers.sort(key=lambda x: x['avg_career_years'], reverse=True)

        longest_careers = division_careers[:3] if len(division_careers) >= 3 else division_careers
        shortest_careers = division_careers[-3:] if len(division_careers) >= 3 else []

        return {
            'division_career_stats': division_careers,
            'longest_career_divisions': [d['weight_class'] for d in longest_careers],
            'shortest_career_divisions': [d['weight_class'] for d in shortest_careers],
            'overall_avg_career_years': df['career_span_years'].mean(),
            'overall_avg_ufc_fights': df['ufc_fights'].mean(),
            'insights': [
                f"Average UFC career spans {df['career_span_years'].mean():.1f} years and {df['ufc_fights'].mean():.1f} fights",
                f"Longest careers: {', '.join([d['weight_class'] for d in longest_careers[:2]])}",
                f"Shortest careers: {', '.join([d['weight_class'] for d in shortest_careers[:2]])}" if shortest_careers else "",
            ]
        }

    def analyze_division_recovery(self) -> Dict[str, Any]:
        """
        Analyze which divisions are hardest to recover from losing streaks.
        """
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT
                    fi.fight_id, fi.fighter_red_id, fi.fighter_blue_id,
                    fi.winner_id, fi.weight_class, e.date
                FROM fights fi
                JOIN events e ON fi.event_id = e.event_id
                WHERE fi.winner_id IS NOT NULL AND fi.weight_class IS NOT NULL
                ORDER BY e.date
            ''', conn)

        df['date'] = pd.to_datetime(df['date'])

        # Track losing streaks and recovery by division
        division_recovery = {}

        for wc in df['weight_class'].dropna().unique():
            wc_df = df[df['weight_class'] == wc].sort_values('date')

            fighters_in_division = set(wc_df['fighter_red_id'].tolist() + wc_df['fighter_blue_id'].tolist())

            recoveries = []
            for fighter_id in fighters_in_division:
                fighter_fights = wc_df[
                    (wc_df['fighter_red_id'] == fighter_id) |
                    (wc_df['fighter_blue_id'] == fighter_id)
                ].sort_values('date')

                if len(fighter_fights) < 4:
                    continue

                # Track streaks
                current_streak = 0
                in_losing_streak = False
                streak_start_idx = 0

                for idx, (_, fight) in enumerate(fighter_fights.iterrows()):
                    won = fight['winner_id'] == fighter_id

                    if not won:
                        if not in_losing_streak:
                            in_losing_streak = True
                            streak_start_idx = idx
                        current_streak += 1
                    else:
                        if in_losing_streak and current_streak >= 2:
                            # Check if they recovered (won after losing streak)
                            recoveries.append({
                                'streak_length': current_streak,
                                'recovered': True
                            })
                        in_losing_streak = False
                        current_streak = 0

                # If still in losing streak at end of career
                if in_losing_streak and current_streak >= 2:
                    recoveries.append({
                        'streak_length': current_streak,
                        'recovered': False
                    })

            if recoveries:
                recovery_rate = sum(1 for r in recoveries if r['recovered']) / len(recoveries) * 100
                avg_streak = np.mean([r['streak_length'] for r in recoveries])
                division_recovery[wc] = {
                    'recovery_rate': recovery_rate,
                    'avg_losing_streak': avg_streak,
                    'total_streaks': len(recoveries),
                }

        # Sort by recovery rate
        sorted_divisions = sorted(division_recovery.items(), key=lambda x: x[1]['recovery_rate'])

        hardest_divisions = sorted_divisions[:3]
        easiest_divisions = sorted_divisions[-3:]

        return {
            'division_recovery_stats': division_recovery,
            'hardest_to_recover': [d[0] for d in hardest_divisions],
            'easiest_to_recover': [d[0] for d in easiest_divisions],
            'insight': f"Hardest divisions to recover from losing streaks: {', '.join([d[0] for d in hardest_divisions[:2]])}"
        }

    def analyze_division_newcomers(self) -> Dict[str, Any]:
        """
        Analyze which divisions have the most newcomers and debut success rates.
        """
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT
                    fi.fighter_red_id, fi.fighter_blue_id, fi.winner_id,
                    fi.weight_class, e.date, e.name as event_name
                FROM fights fi
                JOIN events e ON fi.event_id = e.event_id
                WHERE fi.weight_class IS NOT NULL AND fi.winner_id IS NOT NULL
                ORDER BY e.date
            ''', conn)

        df['date'] = pd.to_datetime(df['date'])
        recent_df = df[df['date'] >= '2020-01-01']  # Last 4+ years

        # Track first fights
        fighter_first_fights = {}

        for _, row in df.iterrows():
            for fighter_id in [row['fighter_red_id'], row['fighter_blue_id']]:
                if fighter_id not in fighter_first_fights:
                    fighter_first_fights[fighter_id] = {
                        'date': row['date'],
                        'weight_class': row['weight_class'],
                        'won': row['winner_id'] == fighter_id,
                    }

        # Analyze recent newcomers by division
        recent_debuts = [f for f in fighter_first_fights.values() if f['date'] >= pd.Timestamp('2020-01-01')]

        division_newcomers = {}
        for debut in recent_debuts:
            wc = debut['weight_class']
            if wc not in division_newcomers:
                division_newcomers[wc] = {'total': 0, 'wins': 0}
            division_newcomers[wc]['total'] += 1
            if debut['won']:
                division_newcomers[wc]['wins'] += 1

        division_stats = []
        for wc, stats in division_newcomers.items():
            if stats['total'] >= 5:
                division_stats.append({
                    'weight_class': wc,
                    'newcomer_count': stats['total'],
                    'debut_win_rate': stats['wins'] / stats['total'] * 100,
                })

        division_stats.sort(key=lambda x: x['newcomer_count'], reverse=True)

        return {
            'division_newcomer_stats': division_stats,
            'most_newcomers_division': division_stats[0]['weight_class'] if division_stats else None,
            'total_recent_debuts': len(recent_debuts),
            'overall_debut_win_rate': sum(1 for d in recent_debuts if d['won']) / len(recent_debuts) * 100 if recent_debuts else 0,
            'insight': f"Most newcomer-heavy division: {division_stats[0]['weight_class']} with {division_stats[0]['newcomer_count']} debuts since 2020" if division_stats else "Insufficient data"
        }

    # =========================================================================
    # FIGHTING STYLE EVOLUTION
    # =========================================================================

    def analyze_style_evolution(self) -> Dict[str, Any]:
        """
        Analyze how fighting styles have evolved over different UFC eras.
        """
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT
                    f.fight_id, f.method, f.round, f.weight_class,
                    e.date,
                    fsr.sig_strikes_landed_per_min as red_strikes,
                    fsr.takedowns_avg_per_15min as red_td,
                    fsr.submissions_avg_per_15min as red_sub,
                    fsb.sig_strikes_landed_per_min as blue_strikes,
                    fsb.takedowns_avg_per_15min as blue_td,
                    fsb.submissions_avg_per_15min as blue_sub
                FROM fights f
                JOIN events e ON f.event_id = e.event_id
                LEFT JOIN fighter_stats fsr ON f.fighter_red_id = fsr.fighter_id
                LEFT JOIN fighter_stats fsb ON f.fighter_blue_id = fsb.fighter_id
                WHERE f.winner_id IS NOT NULL
            ''', conn)

        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year

        # Calculate era averages
        eras = {
            'Early (Pre-2010)': df[df['year'] < 2010],
            'Growth (2010-2015)': df[(df['year'] >= 2010) & (df['year'] < 2016)],
            'Modern (2016-2020)': df[(df['year'] >= 2016) & (df['year'] < 2021)],
            'Current (2021+)': df[df['year'] >= 2021],
        }

        era_stats = []
        for era_name, era_df in eras.items():
            if len(era_df) < 50:
                continue

            # Calculate style metrics
            ko_rate = (era_df['method'].str.contains('KO|TKO', na=False, case=False)).mean() * 100
            sub_rate = (era_df['method'].str.contains('Sub', na=False, case=False)).mean() * 100
            dec_rate = (era_df['method'].str.contains('Dec', na=False, case=False)).mean() * 100

            avg_strikes = era_df[['red_strikes', 'blue_strikes']].mean().mean()
            avg_td = era_df[['red_td', 'blue_td']].mean().mean()
            avg_rounds = era_df['round'].mean()

            era_stats.append({
                'era': era_name,
                'total_fights': len(era_df),
                'ko_rate': ko_rate,
                'sub_rate': sub_rate,
                'dec_rate': dec_rate,
                'avg_strikes_per_min': avg_strikes,
                'avg_takedowns': avg_td,
                'avg_fight_length_rounds': avg_rounds,
            })

        # Style trend analysis
        yearly_strikes = df.groupby('year')[['red_strikes', 'blue_strikes']].mean().mean(axis=1)
        yearly_td = df.groupby('year')[['red_td', 'blue_td']].mean().mean(axis=1)

        striking_trend = 'increasing' if yearly_strikes.iloc[-5:].mean() > yearly_strikes.iloc[:5].mean() else 'decreasing'
        wrestling_trend = 'increasing' if yearly_td.iloc[-5:].mean() > yearly_td.iloc[:5].mean() else 'decreasing'

        return {
            'era_stats': era_stats,
            'striking_volume_trend': striking_trend,
            'wrestling_trend': wrestling_trend,
            'yearly_striking_avg': yearly_strikes.to_dict(),
            'yearly_takedown_avg': yearly_td.to_dict(),
            'insights': [
                f"Striking volume has been {striking_trend} over the years",
                f"Wrestling/takedown attempts have been {wrestling_trend}",
                "Modern UFC shows more well-rounded fighters with improved striking technique",
            ]
        }

    # =========================================================================
    # CHAMPION ANALYSIS
    # =========================================================================

    def analyze_champion_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in title fights and champion performances.
        """
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT
                    f.fight_id, f.fighter_red_id, f.fighter_blue_id,
                    f.winner_id, f.method, f.round, f.time, f.weight_class,
                    f.is_title_fight, e.date,
                    fr.wins as red_wins, fr.losses as red_losses,
                    fb.wins as blue_wins, fb.losses as blue_losses
                FROM fights f
                JOIN events e ON f.event_id = e.event_id
                JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
                JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
                WHERE f.is_title_fight = 1 AND f.winner_id IS NOT NULL
                ORDER BY e.date
            ''', conn)

        df['date'] = pd.to_datetime(df['date'])

        if len(df) == 0:
            return {'error': 'No title fight data'}

        # Title fight methods
        ko_rate = (df['method'].str.contains('KO|TKO', na=False, case=False)).mean() * 100
        sub_rate = (df['method'].str.contains('Sub', na=False, case=False)).mean() * 100
        dec_rate = (df['method'].str.contains('Dec', na=False, case=False)).mean() * 100

        # Average title fight duration
        avg_rounds = df['round'].mean()

        # Parse time to calculate average fight time
        def parse_time(t):
            if pd.isna(t) or t is None:
                return 0
            try:
                parts = str(t).split(':')
                return int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else 0
            except:
                return 0

        df['time_seconds'] = df['time'].apply(parse_time)
        df['total_time'] = (df['round'] - 1) * 300 + df['time_seconds']
        avg_fight_time_minutes = df['total_time'].mean() / 60

        # Challenger vs Champion (assume red corner is usually champion)
        # This is a simplification - ideally we'd track actual title holders
        df['underdog_wins'] = (
            (df['red_wins'] > df['blue_wins']) & (df['winner_id'] == df['fighter_blue_id']) |
            (df['blue_wins'] > df['red_wins']) & (df['winner_id'] == df['fighter_red_id'])
        )
        challenger_win_rate = df['underdog_wins'].mean() * 100

        # Title fight by round
        title_by_round = df['round'].value_counts().sort_index().to_dict()

        # Title fights by division
        title_by_division = df.groupby('weight_class').agg({
            'fight_id': 'count',
            'method': lambda x: (x.str.contains('KO|TKO', na=False, case=False)).mean() * 100
        }).rename(columns={'fight_id': 'total_title_fights', 'method': 'ko_rate'})

        return {
            'total_title_fights': len(df),
            'title_ko_rate': ko_rate,
            'title_sub_rate': sub_rate,
            'title_dec_rate': dec_rate,
            'avg_title_fight_rounds': avg_rounds,
            'avg_title_fight_minutes': avg_fight_time_minutes,
            'challenger_win_rate': challenger_win_rate,
            'champion_defense_rate': 100 - challenger_win_rate,
            'title_fights_by_round': title_by_round,
            'title_fights_by_division': title_by_division.to_dict('index'),
            'insights': [
                f"Title fights end by KO/TKO {ko_rate:.1f}% of the time",
                f"Average title fight lasts {avg_rounds:.1f} rounds ({avg_fight_time_minutes:.1f} minutes)",
                f"Challengers win {challenger_win_rate:.1f}% of title fights",
            ]
        }

    def analyze_path_to_title(self) -> Dict[str, Any]:
        """
        Analyze how long it takes fighters to get title shots.
        """
        with self.data_service.get_connection() as conn:
            # Get all fighters who have had title fights
            title_fighters = pd.read_sql('''
                SELECT DISTINCT
                    CASE WHEN f.fighter_red_id IS NOT NULL THEN f.fighter_red_id END as fighter_id
                FROM fights f
                WHERE f.is_title_fight = 1
                UNION
                SELECT DISTINCT f.fighter_blue_id
                FROM fights f
                WHERE f.is_title_fight = 1
            ''', conn)

            title_fighter_ids = title_fighters['fighter_id'].dropna().unique().tolist()

            if not title_fighter_ids:
                return {'error': 'No title fight data'}

            # Get fight history for these fighters
            df = pd.read_sql(f'''
                SELECT
                    f.fight_id, f.fighter_red_id, f.fighter_blue_id,
                    f.is_title_fight, f.weight_class, e.date
                FROM fights f
                JOIN events e ON f.event_id = e.event_id
                WHERE f.fighter_red_id IN ({','.join(map(str, title_fighter_ids))})
                   OR f.fighter_blue_id IN ({','.join(map(str, title_fighter_ids))})
                ORDER BY e.date
            ''', conn)

            # Get total fighters for percentage calculation
            total_fighters = pd.read_sql('SELECT COUNT(*) as cnt FROM fighters WHERE wins + losses >= 1', conn)['cnt'][0]

        df['date'] = pd.to_datetime(df['date'])

        # Calculate fights before title shot for each fighter
        title_paths = []
        for fighter_id in title_fighter_ids:
            fighter_fights = df[
                (df['fighter_red_id'] == fighter_id) | (df['fighter_blue_id'] == fighter_id)
            ].sort_values('date')

            # Find first title fight
            title_fight_idx = fighter_fights[fighter_fights['is_title_fight'] == True].index
            if len(title_fight_idx) == 0:
                continue

            first_title_idx = title_fight_idx[0]
            fights_before_title = len(fighter_fights.loc[:first_title_idx]) - 1

            first_fight_date = fighter_fights['date'].iloc[0]
            title_fight_date = fighter_fights.loc[first_title_idx, 'date']
            time_to_title = (title_fight_date - first_fight_date).days / 365.25

            title_paths.append({
                'fighter_id': fighter_id,
                'fights_before_title': fights_before_title,
                'years_to_title': time_to_title,
            })

        if not title_paths:
            return {'error': 'Could not calculate title paths'}

        avg_fights_to_title = np.mean([p['fights_before_title'] for p in title_paths])
        avg_years_to_title = np.mean([p['years_to_title'] for p in title_paths])
        median_fights_to_title = np.median([p['fights_before_title'] for p in title_paths])

        # Percentage of fighters who got title shots
        pct_got_title_shot = len(title_fighter_ids) / total_fighters * 100

        return {
            'total_fighters_with_title_shots': len(title_fighter_ids),
            'total_ufc_fighters': total_fighters,
            'pct_got_title_shot': pct_got_title_shot,
            'avg_fights_before_title': avg_fights_to_title,
            'median_fights_before_title': median_fights_to_title,
            'avg_years_to_title': avg_years_to_title,
            'insights': [
                f"Only {pct_got_title_shot:.1f}% of UFC fighters ever get a title shot",
                f"Average path to title: {avg_fights_to_title:.1f} fights over {avg_years_to_title:.1f} years",
            ]
        }

    # =========================================================================
    # CAREER STATISTICS
    # =========================================================================

    def analyze_career_statistics(self) -> Dict[str, Any]:
        """
        Comprehensive career statistics for UFC fighters.
        """
        with self.data_service.get_connection() as conn:
            df = pd.read_sql('''
                SELECT
                    f.fighter_id, f.name, f.dob, f.wins, f.losses, f.draws,
                    MIN(e.date) as first_fight,
                    MAX(e.date) as last_fight,
                    COUNT(DISTINCT fi.fight_id) as ufc_fights
                FROM fighters f
                JOIN fights fi ON (f.fighter_id = fi.fighter_red_id OR f.fighter_id = fi.fighter_blue_id)
                JOIN events e ON fi.event_id = e.event_id
                GROUP BY f.fighter_id
                HAVING ufc_fights >= 1
            ''', conn)

        df['first_fight'] = pd.to_datetime(df['first_fight'])
        df['last_fight'] = pd.to_datetime(df['last_fight'])
        df['career_span_years'] = (df['last_fight'] - df['first_fight']).dt.days / 365.25
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['current_age'] = (datetime.now() - df['dob']).dt.days / 365.25

        # Filter active fighters (fought in last 2 years)
        recent_cutoff = datetime.now() - timedelta(days=730)
        active_fighters = df[df['last_fight'] >= recent_cutoff]

        return {
            'total_fighters': len(df),
            'avg_career_duration_years': df['career_span_years'].mean(),
            'median_career_duration_years': df['career_span_years'].median(),
            'avg_ufc_fights': df['ufc_fights'].mean(),
            'median_ufc_fights': df['ufc_fights'].median(),
            'avg_fighter_age': df['current_age'].dropna().mean(),
            'median_fighter_age': df['current_age'].dropna().median(),
            'active_fighters_count': len(active_fighters),
            'avg_wins': df['wins'].mean(),
            'avg_losses': df['losses'].mean(),
            'longest_careers': df.nlargest(5, 'career_span_years')[['name', 'career_span_years', 'ufc_fights']].to_dict('records'),
            'most_fights': df.nlargest(5, 'ufc_fights')[['name', 'ufc_fights', 'wins', 'losses']].to_dict('records'),
            'insights': [
                f"Average UFC career: {df['career_span_years'].mean():.1f} years, {df['ufc_fights'].mean():.1f} fights",
                f"Average UFC fighter age: {df['current_age'].dropna().mean():.1f} years",
                f"Currently active fighters: {len(active_fighters):,}",
            ]
        }

    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================

    def _generate_cluster_insights(self, clusters: List[Dict]) -> List[str]:
        """Generate insights from cluster analysis."""
        insights = []

        # Find dominant style
        styles = [c['style_label'] for c in clusters]
        style_counts = Counter(styles)

        if style_counts:
            dominant = style_counts.most_common(1)[0]
            insights.append(f"The most common fighting archetype is '{dominant[0]}' representing the plurality of UFC fighters.")

        # Compare win rates
        best_cluster = max(clusters, key=lambda x: x['avg_win_rate'])
        insights.append(f"'{best_cluster['style_label']}' fighters have the highest average win rate at {best_cluster['avg_win_rate']*100:.1f}%.")

        # Volume analysis
        striker_clusters = [c for c in clusters if 'Striker' in c['style_label']]
        grappler_clusters = [c for c in clusters if 'Grappler' in c['style_label'] or 'Wrestler' in c['style_label']]

        if striker_clusters and grappler_clusters:
            avg_striker_wr = np.mean([c['avg_win_rate'] for c in striker_clusters])
            avg_grappler_wr = np.mean([c['avg_win_rate'] for c in grappler_clusters])

            if avg_striker_wr > avg_grappler_wr:
                insights.append(f"Strikers have a slight edge with {avg_striker_wr*100:.1f}% avg win rate vs {avg_grappler_wr*100:.1f}% for grapplers.")
            else:
                insights.append(f"Grapplers have a slight edge with {avg_grappler_wr*100:.1f}% avg win rate vs {avg_striker_wr*100:.1f}% for strikers.")

        return insights

    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive unsupervised learning report.
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'sections': {},
        }

        # 1. Fighter Style Clustering
        try:
            report['sections']['fighter_clustering'] = {
                'title': 'Fighter Style Analysis',
                'kmeans': self.cluster_fighter_styles(),
                'dbscan': self.cluster_fighters_dbscan(),
            }
        except Exception as e:
            logger.error(f"Fighter clustering failed: {e}")
            report['sections']['fighter_clustering'] = {'error': str(e)}

        # 2. Dimensionality Reduction
        try:
            report['sections']['dimensionality_reduction'] = {
                'title': 'Fighting Dimensions',
                'pca': self.pca_analysis(),
                'tsne': self.tsne_visualization(),
            }
        except Exception as e:
            logger.error(f"Dimensionality reduction failed: {e}")
            report['sections']['dimensionality_reduction'] = {'error': str(e)}

        # 3. Anomaly Detection
        try:
            report['sections']['anomaly_detection'] = {
                'title': 'Unique Fighters',
                'isolation_forest': self.detect_anomalous_fighters(),
                'local_outliers': self.detect_local_outliers(),
            }
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            report['sections']['anomaly_detection'] = {'error': str(e)}

        # 4. Division Patterns
        try:
            report['sections']['division_patterns'] = {
                'title': 'Division Analysis',
                'clustering': self.analyze_division_patterns(),
                'recovery': self.analyze_division_recovery(),
                'newcomers': self.analyze_division_newcomers(),
            }
        except Exception as e:
            logger.error(f"Division analysis failed: {e}")
            report['sections']['division_patterns'] = {'error': str(e)}

        # 5. Temporal Patterns
        try:
            report['sections']['temporal_patterns'] = {
                'title': 'Historical Trends',
                'yearly': self.analyze_temporal_patterns(),
                'monthly': self.analyze_monthly_patterns(),
                'style_evolution': self.analyze_style_evolution(),
            }
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            report['sections']['temporal_patterns'] = {'error': str(e)}

        # 6. Fight Outcome Patterns
        try:
            report['sections']['outcome_patterns'] = {
                'title': 'Fight Outcomes',
                'finishes': self.analyze_finish_patterns(),
                'upsets': self.analyze_upset_patterns(),
            }
        except Exception as e:
            logger.error(f"Outcome analysis failed: {e}")
            report['sections']['outcome_patterns'] = {'error': str(e)}

        # 7. Career Analysis
        try:
            report['sections']['career_analysis'] = {
                'title': 'Career Patterns',
                'trajectories': self.analyze_career_trajectories(),
                'statistics': self.analyze_career_statistics(),
                'weight_class_careers': self.analyze_weight_class_careers(),
            }
        except Exception as e:
            logger.error(f"Career analysis failed: {e}")
            report['sections']['career_analysis'] = {'error': str(e)}

        # 8. Age Analysis
        try:
            report['sections']['age_analysis'] = {
                'title': 'Age & Performance',
                'performance': self.analyze_age_performance(),
                'by_division': self.analyze_age_by_division(),
            }
        except Exception as e:
            logger.error(f"Age analysis failed: {e}")
            report['sections']['age_analysis'] = {'error': str(e)}

        # 9. Physical Attributes Analysis
        try:
            report['sections']['physical_analysis'] = {
                'title': 'Physical Advantages',
                'height_reach': self.analyze_height_reach_advantage(),
                'physical_clusters': self.cluster_by_physical_attributes(),
            }
        except Exception as e:
            logger.error(f"Physical analysis failed: {e}")
            report['sections']['physical_analysis'] = {'error': str(e)}

        # 10. Champion Analysis
        try:
            report['sections']['champion_analysis'] = {
                'title': 'Title Fight Insights',
                'patterns': self.analyze_champion_patterns(),
                'path_to_title': self.analyze_path_to_title(),
            }
        except Exception as e:
            logger.error(f"Champion analysis failed: {e}")
            report['sections']['champion_analysis'] = {'error': str(e)}

        # Generate executive summary
        report['executive_summary'] = self._generate_executive_summary(report)

        # Cache the report
        self._save_report_cache(report)

        return report

    def _generate_executive_summary(self, report: Dict) -> Dict[str, Any]:
        """Generate executive summary of findings."""
        key_findings = []

        # Extract key insights from each section
        sections = report.get('sections', {})

        if 'fighter_clustering' in sections and 'kmeans' in sections['fighter_clustering']:
            kmeans = sections['fighter_clustering']['kmeans']
            if 'insights' in kmeans:
                key_findings.extend(kmeans['insights'][:2])

        if 'anomaly_detection' in sections and 'isolation_forest' in sections['anomaly_detection']:
            iso = sections['anomaly_detection']['isolation_forest']
            if 'insight' in iso:
                key_findings.append(iso['insight'])

        if 'temporal_patterns' in sections and 'yearly' in sections['temporal_patterns']:
            temporal = sections['temporal_patterns']['yearly']
            if 'insight' in temporal:
                key_findings.append(temporal['insight'])

        if 'outcome_patterns' in sections and 'upsets' in sections['outcome_patterns']:
            upsets = sections['outcome_patterns']['upsets']
            if 'insight' in upsets:
                key_findings.append(upsets['insight'])

        return {
            'key_findings': key_findings,
            'total_analyses': len(sections),
            'recommendation': "These unsupervised learning insights reveal hidden patterns in UFC data that can inform prediction models and provide unique fan perspectives."
        }

    def _save_report_cache(self, report: Dict) -> None:
        """Save report to cache."""
        try:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                return obj

            clean_report = convert_numpy(report)

            with open(ANALYSIS_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(clean_report, f, indent=2, default=str)

            logger.info("Saved analysis report to cache")
        except Exception as e:
            logger.error(f"Failed to save report cache: {e}")

    def load_cached_report(self) -> Optional[Dict[str, Any]]:
        """Load cached report if available and fresh."""
        try:
            if ANALYSIS_CACHE_FILE.exists():
                with open(ANALYSIS_CACHE_FILE, 'r', encoding='utf-8') as f:
                    report = json.load(f)

                # Check freshness
                generated_at = datetime.fromisoformat(report.get('generated_at', '2000-01-01'))
                if datetime.now() - generated_at < timedelta(hours=ANALYSIS_CACHE_TTL_HOURS):
                    return report
        except Exception as e:
            logger.warning(f"Failed to load cached report: {e}")

        return None
