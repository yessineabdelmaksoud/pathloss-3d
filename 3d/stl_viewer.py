import streamlit as st
import plotly.graph_objects as go
import numpy as np
from stl import mesh
import math
from scipy.spatial.distance import cdist
import pandas as pd
import pickle
import os
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Nouvelle classe pour la prédiction de propagation radio
class RadioPropagationPredictor:
    def __init__(self):
        self.model = None
        self.model_info = None
        self.feature_names = None
    
    @property
    def is_loaded(self):
        """Vérifie si le modèle est chargé"""
        return self.model is not None
        
    def load_model(self, model_path="xgboost_radio_propagation_model.pkl"):
        """Charge le modèle XGBoost de prédiction de path loss"""
        try:
            with open(model_path, 'rb') as f:
                self.model_info = pickle.load(f)
            
            self.model = self.model_info['model']
            self.feature_names = self.model_info['feature_names']
            return True
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle: {e}")
            return False
    
    def predict_pathloss(self, distance, numwall, etage, frequence):
        """Prédit le path loss pour un point donné"""
        if self.model is None:
            return None
        
        try:
            sample = pd.DataFrame({
                'distance': [distance],
                'numwall': [numwall],
                'etage': [etage],
                'frequence': [frequence]
            })
            
            prediction = self.model.predict(sample)[0]
            return round(prediction, 2)
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {e}")
            return None
    
    def predict_grid(self, wifi_point, grid_bounds, resolution, frequency, analyzer):
        """Génère une grille et prédit le path loss pour chaque point"""
        if self.model is None:
            return None
        
        try:
            # Générer la grille
            x_range = np.arange(grid_bounds['x_min'], grid_bounds['x_max'] + resolution, resolution)
            y_range = np.arange(grid_bounds['y_min'], grid_bounds['y_max'] + resolution, resolution)
            z_range = np.arange(grid_bounds['z_min'], grid_bounds['z_max'] + resolution, resolution)
            
            X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            
            grid_points = pd.DataFrame({
                'x': X.flatten(),
                'y': Y.flatten(),
                'z': Z.flatten()
            })
            
            # Calculer les features pour chaque point
            st.write("🧮 Calcul des features...")
            progress_bar = st.progress(0)
            
            distances = []
            numwalls = []
            etages = []
            
            for idx, point in grid_points.iterrows():
                # Distance 3D
                distance = np.sqrt(
                    (point['x'] - wifi_point['x'])**2 + 
                    (point['y'] - wifi_point['y'])**2 + 
                    (point['z'] - wifi_point['z'])**2
                )
                distances.append(distance)
                
                # Nombre de murs (ray tracing avec l'analyzer)
                if analyzer.triangles is not None:
                    num_walls, _ = analyzer.count_wall_intersections(
                        [wifi_point['x'], wifi_point['y'], wifi_point['z']],
                        [point['x'], point['y'], point['z']]
                    )
                else:
                    # Estimation simple si pas d'analyzer
                    distance_2d = np.sqrt((point['x'] - wifi_point['x'])**2 + (point['y'] - wifi_point['y'])**2)
                    num_walls = max(0, min(int(distance_2d / 10.0 * 1.5), 6))
                
                numwalls.append(num_walls)
                
                # Différence d'étages
                floor_height = 3.0
                floor_wifi = int(wifi_point['z'] // floor_height)
                floor_point = int(point['z'] // floor_height)
                etages.append(abs(floor_point - floor_wifi))
                
                # Mise à jour de la barre de progression
                progress_bar.progress((idx + 1) / len(grid_points))
            
            progress_bar.empty()
            
            # Ajouter les features calculées
            grid_points['distance'] = distances
            grid_points['numwall'] = numwalls
            grid_points['etage'] = etages
            grid_points['frequence'] = frequency
            
            # Prédiction vectorisée
            st.write("🤖 Prédiction du path loss...")
            features_for_prediction = grid_points[self.feature_names]
            predictions = self.model.predict(features_for_prediction)
            grid_points['pathloss_predicted'] = predictions
            
            return grid_points
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération de la grille: {e}")
            return None

# Configuration de la page
st.set_page_config(
    page_title="Analyseur STL 3D Avancé",
    page_icon="🏗️",
    layout="wide"
)

st.title("Analyseur STL 3D avec Prediction de Propagation Radio")
st.markdown("Analysez votre plan de bâtiment STL et prédisez la propagation WiFi avec l'IA")

class BuildingAnalyzer:
    def __init__(self):
        self.mesh_data = None
        self.triangles = None
        self.vertices = None
        self.normals = None
        self.tree = None
    
    def load_stl(self, file_path):
        """Charge le fichier STL et prépare les données pour l'analyse"""
        try:
            stl_mesh = mesh.Mesh.from_file(file_path)
            
            # Extraire les données du mesh
            self.triangles = stl_mesh.vectors
            self.vertices = self.triangles.reshape(-1, 3)
            self.normals = stl_mesh.normals
            
            # Créer un index spatial pour les recherches rapides
            self._build_spatial_index()
            
            return True
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def _build_spatial_index(self):
        """Construit un index spatial pour les triangles"""
        try:
            if self.triangles is not None and len(self.triangles) > 0:
                # Calculer les centres des triangles
                triangle_centers = np.mean(self.triangles, axis=1)
                
                # Pour une recherche plus efficace, on pourrait utiliser KDTree ou BVH
                # Ici on utilise une approche simplifiée
                self.triangle_centers = triangle_centers
            else:
                self.triangle_centers = np.array([])
        except Exception as e:
            st.warning(f"⚠️ Erreur lors de la construction de l'index spatial: {e}")
            self.triangle_centers = np.array([])
    
    def ray_triangle_intersection(self, ray_origin, ray_direction, triangle):
        """
        Calcule l'intersection entre un rayon et un triangle
        Utilise l'algorithme de Möller-Trumbore
        """
        epsilon = 1e-8
        
        # Extraire les vertices du triangle
        v0, v1, v2 = triangle[0], triangle[1], triangle[2]
        
        # Calculer les vecteurs du triangle
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Calculer le produit vectoriel
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)
        
        if abs(a) < epsilon:
            return False, 0  # Le rayon est parallèle au triangle
        
        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            return False, 0
        
        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)
        
        if v < 0.0 or u + v > 1.0:
            return False, 0
        
        # Calculer t pour trouver le point d'intersection
        t = f * np.dot(edge2, q)
        
        if t > epsilon:  # Intersection valide
            return True, t
        
        return False, 0
    
    def count_wall_intersections(self, start_point, end_point):
        """
        Compte le nombre de murs/obstacles traversés entre deux points
        """
        if self.triangles is None:
            return 0, []
        
        # Calculer la direction du rayon
        ray_direction = np.array(end_point) - np.array(start_point)
        ray_length = np.linalg.norm(ray_direction)
        
        # Éviter les divisions par zéro
        if ray_length == 0:
            return 0, []
            
        ray_direction = ray_direction / ray_length
        
        intersections = []
        
        # Tester l'intersection avec chaque triangle
        for i, triangle in enumerate(self.triangles):
            hit, t = self.ray_triangle_intersection(
                np.array(start_point), ray_direction, triangle
            )
            
            if hit and 0 < t < ray_length:
                intersection_point = np.array(start_point) + t * ray_direction
                intersections.append({
                    'triangle_id': i,
                    'distance': t,
                    'point': intersection_point
                })
        
        # Trier par distance et éliminer les doublons proches
        intersections.sort(key=lambda x: x['distance'])
        
        # Éliminer les intersections très proches (même mur)
        filtered_intersections = []
        min_distance = 0.1  # 10cm minimum entre intersections
        
        for intersection in intersections:
            if not filtered_intersections or \
               intersection['distance'] - filtered_intersections[-1]['distance'] > min_distance:
                filtered_intersections.append(intersection)
        
        return len(filtered_intersections), filtered_intersections
    
    def get_floor_level(self, z_coord, floor_height, ground_level=0):
        """Détermine l'étage basé sur la coordonnée Z"""
        if z_coord < ground_level:
            return 0  # Sous-sol ou niveau inférieur
        
        floor_number = int((z_coord - ground_level) / floor_height) + 1
        return max(1, floor_number)

# Initialiser l'analyseur
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = BuildingAnalyzer()

# Fonction pour charger le modèle STL
@st.cache_data
def charger_modele_stl(fichier_plan, scale_factor_x=1.0, scale_factor_y=1.0, scale_factor_z=1.0, _fichier_temporaire=None):
    """Charge et prépare le modèle STL pour Plotly avec mise à l'échelle"""
    try:
        # Vérifier si le fichier existe
        import os
        if not os.path.exists(fichier_plan):
            raise FileNotFoundError(f"Le fichier {fichier_plan} n'existe pas")
        
        # Charger le fichier STL
        stl_mesh = mesh.Mesh.from_file(fichier_plan)
        
        # Vérifier que le mesh est valide
        if stl_mesh.vectors is None or len(stl_mesh.vectors) == 0:
            raise ValueError("Le fichier STL ne contient pas de données valides")
        
        # Extraire les vertices
        vertices = stl_mesh.vectors.reshape(-1, 3)
        
        # Appliquer la mise à l'échelle
        vertices[:, 0] *= scale_factor_x  # X
        vertices[:, 1] *= scale_factor_y  # Y
        vertices[:, 2] *= scale_factor_z  # Z
        
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        
        # Créer les indices des triangles
        n_triangles = len(stl_mesh.vectors)
        i = np.arange(0, n_triangles * 3, 3)
        j = np.arange(1, n_triangles * 3, 3)
        k = np.arange(2, n_triangles * 3, 3)
        
        # Calculer les limites après mise à l'échelle
        limites = {
            'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
            'y_min': float(np.min(y)), 'y_max': float(np.max(y)),
            'z_min': float(np.min(z)), 'z_max': float(np.max(z))
        }
        
        # Charger dans l'analyseur seulement si on n'est pas en mode cache
        try:
            # Créer une copie manuelle du mesh avec mise à l'échelle
            scaled_vectors = stl_mesh.vectors.copy()
            scaled_vectors[:, :, 0] *= scale_factor_x
            scaled_vectors[:, :, 1] *= scale_factor_y
            scaled_vectors[:, :, 2] *= scale_factor_z
            
            # Mise à jour de l'analyseur si possible
            if 'analyzer' in st.session_state:
                st.session_state.analyzer.triangles = scaled_vectors
                st.session_state.analyzer.vertices = vertices
                st.session_state.analyzer.normals = stl_mesh.normals
                st.session_state.analyzer._build_spatial_index()
        except Exception as analyzer_error:
            st.warning(f"⚠️ Erreur lors de la mise à jour de l'analyseur: {analyzer_error}")
        
        return x, y, z, i, j, k, n_triangles, limites, True
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du STL: {e}")
        return None, None, None, None, None, None, 0, None, False

# Interface principal
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("🔧 Configuration du Plan")
    
    # Chargement du fichier STL
    st.subheader("📁 Fichier STL")
    
    # Options de chargement
    option_fichier = st.radio(
        "Choisir le mode de chargement :",
        ["📤 Télécharger un fichier", "📂 Fichier local"],
        index=0
    )
    
    fichier_stl = None
    fichier_temporaire = None
    
    if option_fichier == "📤 Télécharger un fichier":
        uploaded_file = st.file_uploader(
            "Sélectionnez votre fichier STL",
            type=['stl'],
            help="Glissez-déposez votre fichier .stl ou cliquez pour parcourir"
        )
        
        if uploaded_file is not None:
            # Sauvegarder temporairement le fichier
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                fichier_temporaire = tmp_file.name
                fichier_stl = fichier_temporaire
            
            st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès!")
            st.info(f"📊 Taille: {len(uploaded_file.getvalue()) / 1024:.1f} KB")
            
    else:  # Fichier local
        fichier_local = st.text_input(
            "Nom du fichier STL local", 
            value="modell_stl.stl",
            help="Entrez le nom du fichier STL présent dans le dossier de l'application"
        )
        
        if fichier_local:
            import os
            if os.path.exists(fichier_local):
                fichier_stl = fichier_local
                st.success(f"✅ Fichier '{fichier_local}' trouvé!")
                taille = os.path.getsize(fichier_local) / 1024
                st.info(f"📊 Taille: {taille:.1f} KB")
            else:
                st.error(f"❌ Fichier '{fichier_local}' introuvable!")
                st.write("💡 Assurez-vous que le fichier est dans le même dossier que l'application.")
    
    # Dimensions réelles du plan
    st.subheader("📏 Dimensions Réelles")
    col_dim1, col_dim2 = st.columns(2)
    
    with col_dim1:
        longueur_reelle = st.number_input("Longueur réelle (m)", value=50.0, min_value=1.0, step=1.0)
        largeur_reelle = st.number_input("Largeur réelle (m)", value=30.0, min_value=1.0, step=1.0)
    
    with col_dim2:
        hauteur_etage = st.number_input("Hauteur par étage (m)", value=3.0, min_value=2.0, max_value=6.0, step=0.1)
        nombre_etages = st.number_input("Nombre d'étages", value=3, min_value=1, max_value=20, step=1)
    
    # Options d'affichage
    st.subheader("🎨 Mode d'Analyse")
    
    mode_analyse = st.radio(
        "Choisissez le mode d'analyse",
        options=["Analyse de Trajectoire"],
        help="Mode trajectoire: analyse entre 2 points | Mode WiFi: prédiction de couverture"
    )
    couleur_stl = st.color_picker("Couleur du modèle", "#00BFFF")
    opacite = st.slider("Opacité", 0.1, 1.0, 0.8, 0.1)
    afficher_wireframe = st.checkbox("Affichage filaire", False)
    
    # Configuration selon le mode choisi
    if mode_analyse == "Analyse de Trajectoire":
        # Analyse de trajet avancée
        st.subheader("Analyse de Trajet Avancee")
        mode_analyse_trajet = st.checkbox("Mode analyse avancée", False)
    else:
        st.warning("⚠️ Modèle XGBoost non chargé")
        generate_heatmap = False
        mode_analyse_trajet = False
    
    # Analyse de trajet avancée (mode trajectoire uniquement)
    if mode_analyse == "📏 Analyse de Trajectoire":
        st.subheader("📍 Configuration des Points")
    
    # Initialiser les variables d'analyse
    analyser = False
    
    if mode_analyse == "📏 Analyse de Trajectoire" and mode_analyse_trajet:
        st.write("**Point de Départ:**")
        x1 = st.number_input("X1 (m)", value=0.0, step=0.1, key="x1")
        y1 = st.number_input("Y1 (m)", value=0.0, step=0.1, key="y1")
        z1 = st.number_input("Z1 (m)", value=1.5, step=0.1, key="z1")
        
        st.write("**Point d'Arrivée:**")
        x2 = st.number_input("X2 (m)", value=10.0, step=0.1, key="x2")
        y2 = st.number_input("Y2 (m)", value=10.0, step=0.1, key="y2")
        z2 = st.number_input("Z2 (m)", value=4.5, step=0.1, key="z2")
        
        # Bouton d'analyse
        analyser = st.button("🔍 Analyser le Trajet", type="primary")
    else:
        # Valeurs par défaut quand le mode analyse n'est pas activé
        x1, y1, z1 = 0.0, 0.0, 1.5
        x2, y2, z2 = 10.0, 10.0, 4.5

with col1:
    st.subheader("🏗️ Plan STL 3D - Analyse Avancée")
    
    # Vérifier qu'un fichier STL est disponible
    if fichier_stl is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier STL dans le panneau de configuration")
        st.info("""
        **Pour commencer :**
        1. 📤 Utilisez le téléchargeur de fichier pour charger votre plan STL
        2. 📂 Ou spécifiez le nom d'un fichier local existant
        3. ⚙️ Configurez les dimensions réelles de votre bâtiment
        4. 🎯 Activez l'analyse de trajet pour des mesures précises
        """)
    else:
        # Calculer les facteurs d'échelle si on a les dimensions réelles
        with st.spinner("Chargement du modèle STL..."):
            # Charger le modèle avec échelle 1:1 pour obtenir les dimensions originales
            stl_data_original = charger_modele_stl(fichier_stl, 1.0, 1.0, 1.0, _fichier_temporaire=fichier_temporaire)
        
        if stl_data_original[8]:  # Succès du chargement
            _, _, _, _, _, _, _, limites_orig, _ = stl_data_original
            
            # Calculer les facteurs d'échelle
            dim_x_orig = limites_orig['x_max'] - limites_orig['x_min']
            dim_y_orig = limites_orig['y_max'] - limites_orig['y_min']
            dim_z_orig = limites_orig['z_max'] - limites_orig['z_min']
            
            # Éviter la division par zéro
            if dim_x_orig == 0: dim_x_orig = 1
            if dim_y_orig == 0: dim_y_orig = 1
            if dim_z_orig == 0: dim_z_orig = 1
            
            scale_x = longueur_reelle / dim_x_orig
            scale_y = largeur_reelle / dim_y_orig
            scale_z = (hauteur_etage * nombre_etages) / dim_z_orig
            
            # Recharger avec la bonne échelle
            stl_data = charger_modele_stl(fichier_stl, scale_x, scale_y, scale_z, _fichier_temporaire=fichier_temporaire)
            
            # Mise à jour manuelle de l'analyseur si nécessaire
            if stl_data[8] and st.session_state.analyzer.triangles is None:
                try:
                    stl_mesh = mesh.Mesh.from_file(fichier_stl)
                    scaled_vectors = stl_mesh.vectors.copy()
                    scaled_vectors[:, :, 0] *= scale_x
                    scaled_vectors[:, :, 1] *= scale_y
                    scaled_vectors[:, :, 2] *= scale_z
                    
                    st.session_state.analyzer.triangles = scaled_vectors
                    st.session_state.analyzer.vertices = stl_data[0], stl_data[1], stl_data[2]
                    st.session_state.analyzer.normals = stl_mesh.normals
                    st.session_state.analyzer._build_spatial_index()
                except Exception as e:
                    st.warning(f"⚠️ Erreur lors de la mise à jour manuelle de l'analyseur: {e}")
        
        if stl_data[8]:
            x_stl, y_stl, z_stl, i_stl, j_stl, k_stl, n_triangles, limites, _ = stl_data
            
            # Afficher les informations du modèle
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Triangles", f"{n_triangles:,}")
            with col_info2:
                st.metric("Points", f"{len(x_stl):,}")
            with col_info3:
                dimensions = f"{longueur_reelle:.0f}×{largeur_reelle:.0f}×{hauteur_etage * nombre_etages:.1f}"
                st.metric("Dimensions (m)", dimensions)
            with col_info4:
                st.metric("Étages", nombre_etages)
            
            # Variables pour l'analyse
            distance = 0
            murs_traverses = 0
            etage_point1 = 1
            etage_point2 = 1
            intersections_details = []
            
            # Initialiser le prédicteur de propagation radio
            if 'radio_predictor' not in st.session_state:
                st.session_state.radio_predictor = RadioPropagationPredictor()
                st.session_state.radio_predictor.load_model()
            
            # Interface selon le mode choisi
            if mode_analyse == "📏 Analyse de Trajectoire":
                # Mode analyse de trajectoire (code existant)
                # Effectuer l'analyse si demandée
                if mode_analyse_trajet:
                    # Calculer toujours les valeurs de base
                    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                    etage_point1 = st.session_state.analyzer.get_floor_level(z1, hauteur_etage)
                    etage_point2 = st.session_state.analyzer.get_floor_level(z2, hauteur_etage)
                    
                    # Si le bouton analyser a été cliqué, faire l'analyse complète
                    if analyser:
                        # Compter les murs traversés
                        murs_traverses, intersections_details = st.session_state.analyzer.count_wall_intersections(
                            [x1, y1, z1], [x2, y2, z2]
                        )
                        
                        # Afficher les résultats d'analyse
                        st.success("✅ Analyse terminée!")
                        
                        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                        with col_res1:
                            st.metric("Distance 3D", f"{distance:.2f} m")
                        with col_res2:
                            st.metric("Murs Traversés", murs_traverses)
                        with col_res3:
                            st.metric("Étage Point 1", etage_point1)
                        with col_res4:
                            st.metric("Étage Point 2", etage_point2)
                        
                        # Afficher les détails des intersections
                        if intersections_details:
                            with st.expander(f"📋 Détails des {len(intersections_details)} intersections"):
                                intersections_df = pd.DataFrame([
                                    {
                                        'Intersection': i+1,
                                        'Distance (m)': f"{inter['distance']:.2f}",
                                        'X': f"{inter['point'][0]:.2f}",
                                        'Y': f"{inter['point'][1]:.2f}",
                                        'Z': f"{inter['point'][2]:.2f}",
                                        'Étage': st.session_state.analyzer.get_floor_level(
                                            inter['point'][2], hauteur_etage
                                        )
                                    }
                                    for i, inter in enumerate(intersections_details)
                                ])
                                st.dataframe(intersections_df, use_container_width=True)
                    else:
                        # Affichage de prévisualisation sans analyse complète
                        col_preview1, col_preview2, col_preview3 = st.columns(3)
                        with col_preview1:
                            st.metric("Distance Préview", f"{distance:.2f} m")
                        with col_preview2:
                            st.metric("Étage Point 1", etage_point1)
                        with col_preview3:
                            st.metric("Étage Point 2", etage_point2)
            
            else:  # Mode prédiction propagation WiFi
                st.subheader("📡 Configuration WiFi")
                
                # Configuration WiFi
                col_wifi1, col_wifi2 = st.columns(2)
                with col_wifi1:
                    frequence_wifi = st.selectbox(
                        "Fréquence WiFi (MHz)",
                        options=[900, 1800, 2400, 5000, 5800],
                        index=2,  # 2400 MHz par défaut
                        help="Fréquence de votre signal WiFi"
                    )
                    
                    resolution_grille = st.slider(
                        "Résolution de la grille",
                        min_value=10, max_value=100, value=30,
                        help="Nombre de points par dimension (plus = plus précis mais plus lent)"
                    )
                
                with col_wifi2:
                    # Position du point WiFi (émetteur)
                    st.write("📍 **Position de l'émetteur WiFi:**")
                    wifi_x = st.number_input("WiFi X (m)", value=x1, step=0.1, key="main_wifi_x")
                    wifi_y = st.number_input("WiFi Y (m)", value=y1, step=0.1, key="main_wifi_y")
                    wifi_z = st.number_input("WiFi Z (m)", value=z1, step=0.1, key="main_wifi_z")
                
                # Bouton pour générer la prédiction
                if st.button("🚀 Générer la Carte de Propagation WiFi", type="primary"):
                    if st.session_state.radio_predictor.is_loaded:
                        with st.spinner("Génération de la carte de propagation..."):
                            # Créer la grille de points
                            x_range = np.linspace(limites['x_min'], limites['x_max'], resolution_grille)
                            y_range = np.linspace(limites['y_min'], limites['y_max'], resolution_grille)
                            z_range = np.linspace(limites['z_min'], limites['z_max'], max(5, resolution_grille//6))
                            
                            # Préparer les données pour la prédiction
                            predictions_data = []
                            total_points = len(x_range) * len(y_range) * len(z_range)
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            point_count = 0
                            
                            for i, x in enumerate(x_range):
                                for j, y in enumerate(y_range):
                                    for k, z in enumerate(z_range):
                                        # Calculer la distance
                                        dist = math.sqrt((x-wifi_x)**2 + (y-wifi_y)**2 + (z-wifi_z)**2)
                                        
                                        # Calculer le nombre de murs (ray tracing)
                                        if dist > 0.1:  # Éviter les points trop proches
                                            num_walls, _ = st.session_state.analyzer.count_wall_intersections(
                                                [wifi_x, wifi_y, wifi_z], [x, y, z]
                                            )
                                        else:
                                            num_walls = 0
                                        
                                        # Calculer la différence d'étages
                                        etage_wifi = st.session_state.analyzer.get_floor_level(wifi_z, hauteur_etage)
                                        etage_point = st.session_state.analyzer.get_floor_level(z, hauteur_etage)
                                        diff_etage = abs(etage_point - etage_wifi)
                                        
                                        # Prédire le path loss
                                        pathloss = st.session_state.radio_predictor.predict_pathloss(
                                            dist, num_walls, diff_etage, frequence_wifi
                                        )
                                        
                                        if pathloss is not None:
                                            predictions_data.append({
                                                'x': x, 'y': y, 'z': z,
                                                'distance': dist,
                                                'numwall': num_walls,
                                                'etage_diff': diff_etage,
                                                'pathloss': pathloss
                                            })
                                        
                                        point_count += 1
                                        if point_count % 100 == 0:
                                            progress = point_count / total_points
                                            progress_bar.progress(progress)
                                            status_text.text(f"Traitement: {point_count}/{total_points} points ({progress*100:.1f}%)")
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Prédiction terminée!")
                            
                            # Stocker les résultats
                            st.session_state.predictions_data = predictions_data
                            
                            # Afficher les statistiques
                            if predictions_data:
                                pathloss_values = [p['pathloss'] for p in predictions_data]
                                
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                with col_stat1:
                                    st.metric("Points Analysés", f"{len(predictions_data):,}")
                                with col_stat2:
                                    st.metric("Path Loss Min", f"{min(pathloss_values):.1f} dB")
                                with col_stat3:
                                    st.metric("Path Loss Max", f"{max(pathloss_values):.1f} dB")
                                with col_stat4:
                                    st.metric("Path Loss Moyen", f"{np.mean(pathloss_values):.1f} dB")
                                
                                # === GÉNÉRER LA HEATMAP 3D ===
                                st.subheader("🌈 Heatmap 3D du Path Loss")
                                
                                # Convertir en DataFrame pour faciliter la manipulation
                                viz_df = pd.DataFrame(predictions_data)
                                
                                # Échantillonner si trop de points
                                max_points_viz = 5000
                                if len(viz_df) > max_points_viz:
                                    viz_data = viz_df.sample(n=max_points_viz, random_state=42)
                                    st.info(f"📊 Affichage de {max_points_viz:,} points échantillonnés sur {len(viz_df):,}")
                                else:
                                    viz_data = viz_df
                                
                                # 1. Heatmap 3D principale superposée au plan STL
                                st.write("**🎨 Heatmap 3D avec Plan STL**")
                                
                                fig_heatmap = go.Figure()
                                
                                # Ajouter le plan STL en arrière-plan (transparence élevée)
                                fig_heatmap.add_trace(go.Mesh3d(
                                    x=x_stl, y=y_stl, z=z_stl,
                                    i=i_stl, j=j_stl, k=k_stl,
                                    color='lightgray',
                                    opacity=0.2,
                                    name='Plan STL',
                                    showscale=False,
                                    lighting=dict(ambient=0.3, diffuse=0.8, specular=0.1),
                                    hovertemplate='<b>Plan STL</b><extra></extra>'
                                ))
                                
                                # Ajouter la heatmap des prédictions
                                fig_heatmap.add_trace(go.Scatter3d(
                                    x=viz_data['x'],
                                    y=viz_data['y'], 
                                    z=viz_data['z'],
                                    mode='markers',
                                    marker=dict(
                                        size=6,
                                        color=viz_data['pathloss'],
                                        colorscale='Viridis',
                                        cmin=viz_data['pathloss'].min(),
                                        cmax=viz_data['pathloss'].max(),
                                        colorbar=dict(
                                            title="Path Loss (dB)",
                                            thickness=20,
                                            len=0.8,
                                            x=1.02
                                        ),
                                        showscale=True,
                                        opacity=0.8,
                                        line=dict(width=0)
                                    ),
                                    text=[f'Position: ({x:.1f}, {y:.1f}, {z:.1f})<br>'
                                          f'Distance: {d:.1f}m<br>'
                                          f'Murs: {w}<br>'
                                          f'Etages: {e}<br>'
                                          f'Path Loss: {p:.1f} dB'
                                          for x, y, z, d, w, e, p in zip(
                                              viz_data['x'], viz_data['y'], viz_data['z'],
                                              viz_data['distance'], viz_data['numwall'], 
                                              viz_data['etage_diff'], viz_data['pathloss']
                                          )],
                                    hovertemplate='%{text}<extra></extra>',
                                    name='Prediction Path Loss'
                                ))
                                
                                # Point WiFi en évidence
                                fig_heatmap.add_trace(go.Scatter3d(
                                    x=[wifi_x],
                                    y=[wifi_y],
                                    z=[wifi_z],
                                    mode='markers+text',
                                    marker=dict(
                                        size=20,
                                        color='red',
                                        symbol='diamond',
                                        line=dict(color='white', width=3)
                                    ),
                                    text=['WiFi'],
                                    textposition='top center',
                                    textfont=dict(size=14, color='red'),
                                    name='Point WiFi',
                                    hovertemplate=f'<b>Point WiFi Emetteur</b><br>' +
                                                  f'Position: ({wifi_x}, {wifi_y}, {wifi_z})<br>' +
                                                  f'Frequence: {frequence_wifi} MHz<extra></extra>'
                                ))
                                
                                # Configuration du layout
                                fig_heatmap.update_layout(
                                    title=dict(
                                        text=f'Heatmap 3D du Path Loss WiFi - {frequence_wifi} MHz<br>' +
                                             f'<sub>Points: {len(viz_data):,} | Range: {viz_data["pathloss"].min():.1f}-{viz_data["pathloss"].max():.1f} dB</sub>',
                                        x=0.5,
                                        font=dict(size=16)
                                    ),
                                    scene=dict(
                                        xaxis=dict(
                                            title='X (metres)',
                                            range=[limites['x_min'], limites['x_max']],
                                            showgrid=True,
                                            gridcolor='lightgray'
                                        ),
                                        yaxis=dict(
                                            title='Y (metres)',
                                            range=[limites['y_min'], limites['y_max']],
                                            showgrid=True,
                                            gridcolor='lightgray'
                                        ),
                                        zaxis=dict(
                                            title='Z (metres)',
                                            range=[limites['z_min'], limites['z_max']],
                                            showgrid=True,
                                            gridcolor='lightgray'
                                        ),
                                        # Corriger la compression en utilisant un aspect ratio adaptatif
                                        aspectratio=dict(
                                            x=(limites['x_max'] - limites['x_min']),
                                            y=(limites['y_max'] - limites['y_min']),
                                            z=(limites['z_max'] - limites['z_min']) * 2  # Augmenter Z pour éviter la compression
                                        ),
                                        aspectmode='manual',
                                        camera=dict(
                                            eye=dict(x=1.5, y=1.5, z=1.2),
                                            center=dict(x=0, y=0, z=0)
                                        ),
                                        bgcolor='white'
                                    ),
                                    width=1000,
                                    height=700,
                                    showlegend=True,
                                    legend=dict(
                                        x=0.02,
                                        y=0.98,
                                        bgcolor='rgba(255,255,255,0.8)',
                                        bordercolor='black',
                                        borderwidth=1
                                    )
                                )
                                
                                # Afficher la heatmap 3D
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                                
                                # 2. Vues par ÉTAGES (au lieu de tranches Z arbitraires)
                                st.write("**� Vues par Étages**")
                                
                                # Calculer les étages pour chaque point
                                viz_data['etage_point'] = viz_data['z'].apply(
                                    lambda z: st.session_state.analyzer.get_floor_level(z, hauteur_etage)
                                )
                                
                                # Obtenir les étages uniques présents dans les données
                                etages_disponibles = sorted(viz_data['etage_point'].unique())
                                nb_etages_dispo = len(etages_disponibles)
                                
                                # Afficher les informations sur les étages
                                st.info(f"📊 Étages détectés: {etages_disponibles} (hauteur d'étage: {hauteur_etage}m)")
                                
                                if nb_etages_dispo == 0:
                                    st.warning("⚠️ Aucun étage détecté dans les données")
                                else:
                                    # Préparer la configuration des subplots selon le nombre d'étages
                                    if nb_etages_dispo == 1:
                                        fig_slices = make_subplots(
                                            rows=1, cols=1,
                                            subplot_titles=[f'Étage {etages_disponibles[0]}'],
                                            specs=[[{'type': 'scatter'}]]
                                        )
                                    elif nb_etages_dispo == 2:
                                        fig_slices = make_subplots(
                                            rows=1, cols=2,
                                            subplot_titles=[f'Étage {etage}' for etage in etages_disponibles],
                                            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
                                        )
                                    elif nb_etages_dispo <= 4:
                                        fig_slices = make_subplots(
                                            rows=2, cols=2,
                                            subplot_titles=[f'Étage {etage}' for etage in etages_disponibles[:4]],
                                            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                                                   [{'type': 'scatter'}, {'type': 'scatter'}]]
                                        )
                                    else:
                                        # Pour plus de 4 étages, on prend les 4 premiers
                                        etages_selectionnes = etages_disponibles[:4]
                                        fig_slices = make_subplots(
                                            rows=2, cols=2,
                                            subplot_titles=[f'Étage {etage}' for etage in etages_selectionnes],
                                            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                                                   [{'type': 'scatter'}, {'type': 'scatter'}]]
                                        )
                                    
                                    # Traiter chaque étage disponible
                                    etages_a_afficher = etages_disponibles[:4]  # Maximum 4 étages
                                    
                                    for i, etage_num in enumerate(etages_a_afficher):
                                        if nb_etages_dispo == 1:
                                            row, col = 1, 1
                                        elif nb_etages_dispo == 2:
                                            row, col = 1, i + 1
                                        else:
                                            row = (i // 2) + 1
                                            col = (i % 2) + 1
                                        
                                        # Données pour cet étage
                                        etage_data = viz_data[viz_data['etage_point'] == etage_num]
                                        
                                        if len(etage_data) > 0:
                                            fig_slices.add_trace(
                                                go.Scatter(
                                                    x=etage_data['x'],
                                                    y=etage_data['y'],
                                                    mode='markers',
                                                    marker=dict(
                                                        size=10,
                                                        color=etage_data['pathloss'],
                                                        colorscale='Viridis',
                                                        showscale=(i == 0),
                                                        colorbar=dict(
                                                            title="Path Loss (dB)", 
                                                            x=1.02,
                                                            len=0.8
                                                        ) if i == 0 else None,
                                                        line=dict(width=0.5, color='white'),
                                                        cmin=viz_data['pathloss'].min(),
                                                        cmax=viz_data['pathloss'].max()
                                                    ),
                                                    text=[f'Etage {etage_num}<br>Path Loss: {p:.1f} dB<br>Distance: {d:.1f}m<br>Murs: {w}' 
                                                          for p, d, w in zip(etage_data['pathloss'], etage_data['distance'], etage_data['numwall'])],
                                                    hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>%{text}<extra></extra>',
                                                    name=f'Etage {etage_num}',
                                                    showlegend=False
                                                ),
                                                row=row, col=col
                                            )
                                            
                                            # Point WiFi s'il est dans cet étage
                                            etage_wifi = st.session_state.analyzer.get_floor_level(wifi_z, hauteur_etage)
                                            if etage_wifi == etage_num:
                                                fig_slices.add_trace(
                                                    go.Scatter(
                                                        x=[wifi_x],
                                                        y=[wifi_y],
                                                        mode='markers+text',
                                                        marker=dict(
                                                            size=25, 
                                                            color='red', 
                                                            symbol='diamond',
                                                            line=dict(color='white', width=3)
                                                        ),
                                                        text=['WiFi'],
                                                        textfont=dict(size=12, color='white'),
                                                        name='Point WiFi' if i == 0 else '',
                                                        showlegend=(i == 0),
                                                        hovertemplate=f'Point WiFi - Etage {etage_num}<extra></extra>'
                                                    ),
                                                    row=row, col=col
                                                )
                                    
                                    fig_slices.update_layout(
                                        title=f'🏢 Analyse du Path Loss par Etages ({len(etages_a_afficher)} etages affiches)',
                                        height=700,
                                        showlegend=True,
                                        # Améliorer l'aspect ratio pour éviter la compression
                                        autosize=False
                                    )
                                    
                                    # Ajuster l'aspect ratio selon le nombre d'étages
                                    if nb_etages_dispo == 1:
                                        fig_slices.update_xaxes(scaleanchor="y", scaleratio=1)
                                    elif nb_etages_dispo == 2:
                                        fig_slices.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
                                        fig_slices.update_xaxes(scaleanchor="y2", scaleratio=1, row=1, col=2)
                                    
                                    st.plotly_chart(fig_slices, use_container_width=True)
                                
                                # 3. Graphiques statistiques
                                st.write("**📊 Analyses Statistiques**")
                                
                                col_graph1, col_graph2 = st.columns(2)
                                
                                with col_graph1:
                                    # Histogramme du path loss
                                    fig_hist = go.Figure()
                                    fig_hist.add_trace(go.Histogram(
                                        x=viz_data['pathloss'],
                                        nbinsx=30,
                                        name='Distribution Path Loss',
                                        marker_color='lightblue',
                                        opacity=0.7
                                    ))
                                    fig_hist.update_layout(
                                        title='📊 Distribution du Path Loss',
                                        xaxis_title='Path Loss (dB)',
                                        yaxis_title='Fréquence',
                                        height=400
                                    )
                                    st.plotly_chart(fig_hist, use_container_width=True)
                                
                                with col_graph2:
                                    # Diagramme de couverture
                                    good_signal = len(viz_data[viz_data['pathloss'] < 70])
                                    medium_signal = len(viz_data[(viz_data['pathloss'] >= 70) & (viz_data['pathloss'] < 90)])
                                    poor_signal = len(viz_data[viz_data['pathloss'] >= 90])
                                    
                                    fig_pie = go.Figure()
                                    fig_pie.add_trace(go.Pie(
                                        values=[good_signal, medium_signal, poor_signal],
                                        labels=['Bon Signal (<70dB)', 'Signal Moyen (70-90dB)', 'Signal Faible (>90dB)'],
                                        marker_colors=['green', 'orange', 'red'],
                                        textinfo='label+percent',
                                        hovertemplate='%{label}<br>Points: %{value}<br>Pourcentage: %{percent}<extra></extra>'
                                    ))
                                    fig_pie.update_layout(
                                        title='📶 Répartition de la Qualité du Signal',
                                        height=400
                                    )
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                
                                # Affichage du résumé de couverture
                                st.success("✅ Heatmap 3D générée avec succès!")
                                
                                total_points = len(viz_data)
                                good_pct = (good_signal / total_points) * 100
                                medium_pct = (medium_signal / total_points) * 100  
                                poor_pct = (poor_signal / total_points) * 100
                                
                                st.write("**📶 Analyse de couverture WiFi:**")
                                st.write(f"🟢 **Bon signal** (<70 dB): {good_signal:,} points ({good_pct:.1f}%)")
                                st.write(f"🟡 **Signal moyen** (70-90 dB): {medium_signal:,} points ({medium_pct:.1f}%)")
                                st.write(f"🔴 **Signal faible** (>90 dB): {poor_signal:,} points ({poor_pct:.1f}%)")
                                
                    else:
                        st.error("❌ Modèle de prédiction non disponible!")
                
                # Si on a des prédictions, permettre l'affichage
                if hasattr(st.session_state, 'predictions_data') and st.session_state.predictions_data:
                    st.subheader("🎨 Visualisation de la Propagation")
                    
                    col_viz1, col_viz2 = st.columns(2)
                    with col_viz1:
                        seuil_pathloss = st.slider(
                            "Seuil Path Loss (dB)",
                            min_value=50, max_value=200, value=120,
                            help="Afficher seulement les points avec path loss < seuil"
                        )
                    
                    with col_viz2:
                        taille_points = st.slider(
                            "Taille des points",
                            min_value=1, max_value=10, value=4
                        )
            
            # Créer la figure Plotly selon le mode
            if mode_analyse == "📡 Prédiction Propagation WiFi" and 'predictions_data' in st.session_state:
                # Mode WiFi - affichage déjà géré dans la section précédente
                pass
            else:
                # Mode trajectoire ou affichage STL simple
                fig = go.Figure()
                
                # Ajouter le modèle STL
                if afficher_wireframe:
                    # Mode filaire
                    fig.add_trace(go.Mesh3d(
                        x=x_stl, y=y_stl, z=z_stl,
                        i=i_stl, j=j_stl, k=k_stl,
                        color=couleur_stl,
                        opacity=opacite,
                        name=f'Plan STL ({n_triangles:,} triangles)',
                        showscale=False,
                        flatshading=True,
                        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.1),
                        hovertemplate='<b>Plan STL</b><br>' +
                                     'X: %{x:.2f}m<br>' +
                                     'Y: %{y:.2f}m<br>' +
                                     'Z: %{z:.2f}m<extra></extra>'
                    ))
                else:
                    # Mode solide
                    fig.add_trace(go.Mesh3d(
                        x=x_stl, y=y_stl, z=z_stl,
                        i=i_stl, j=j_stl, k=k_stl,
                        color=couleur_stl,
                        opacity=opacite,
                        name=f'Plan STL ({n_triangles:,} triangles)',
                        showscale=False,
                        lighting=dict(ambient=0.18, diffuse=1, specular=0.1),
                        hovertemplate='<b>Plan STL</b><br>' +
                                     'X: %{x:.2f}m<br>' +
                                     'Y: %{y:.2f}m<br>' +
                                     'Z: %{z:.2f}m<extra></extra>'
                    ))
            
                # Ajouter les indicateurs d'étages
                for etage in range(1, nombre_etages + 1):
                    z_etage = (etage - 1) * hauteur_etage + hauteur_etage / 2
                    fig.add_trace(go.Scatter3d(
                        x=[limites['x_min']], 
                        y=[limites['y_min']], 
                        z=[z_etage],
                        mode='markers+text',
                        marker=dict(size=1, color='rgba(0,0,0,0)'),
                        text=f'Étage {etage}',
                        textposition='middle right',
                        name=f'Étage {etage}',
                        showlegend=False,
                        hovertemplate=f'<b>Étage {etage}</b><br>Hauteur: {z_etage:.1f}m<extra></extra>'
                    ))
                
                # Ajouter les points d'analyse si le mode trajectoire est activé
                if mode_analyse == "📏 Analyse de Trajectoire" and mode_analyse_trajet:
                    # Point 1
                    fig.add_trace(go.Scatter3d(
                        x=[x1], y=[y1], z=[z1],
                        mode='markers+text',
                        marker=dict(size=12, color='green'),
                        text=f'Départ (É{etage_point1})',
                        textposition='top center',
                        name=f'Point Départ - Étage {etage_point1}',
                        hovertemplate=f'<b>Point de Départ</b><br>' +
                                     f'Coordonnées: ({x1:.1f}, {y1:.1f}, {z1:.1f})<br>' +
                                     f'Étage: {etage_point1}<extra></extra>'
                    ))
                    
                    # Point 2
                    fig.add_trace(go.Scatter3d(
                        x=[x2], y=[y2], z=[z2],
                        mode='markers+text',
                        marker=dict(size=12, color='red'),
                        text=f'Arrivée (É{etage_point2})',
                        textposition='top center',
                        name=f'Point Arrivée - Étage {etage_point2}',
                        hovertemplate=f'<b>Point d\'Arrivée</b><br>' +
                                     f'Coordonnées: ({x2:.1f}, {y2:.1f}, {z2:.1f})<br>' +
                                     f'Étage: {etage_point2}<extra></extra>'
                    ))
                    
                    # Ligne de trajet
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2],
                        mode='lines',                    line=dict(width=8, color='blue'),
                    name=f'Trajet ({distance:.2f}m, {murs_traverses} murs)',
                    hovertemplate=f'<b>Trajet</b><br>' +
                                 f'Distance: {distance:.2f}m<br>' +
                                 f'Murs traversés: {murs_traverses}<extra></extra>'
                    ))
                    
                    # Ajouter les points d'intersection
                    if intersections_details:
                        intersection_x = [inter['point'][0] for inter in intersections_details]
                        intersection_y = [inter['point'][1] for inter in intersections_details]
                        intersection_z = [inter['point'][2] for inter in intersections_details]
                        
                        fig.add_trace(go.Scatter3d(
                            x=intersection_x, y=intersection_y, z=intersection_z,
                            mode='markers',
                            marker=dict(size=8, color='orange', symbol='diamond'),
                            name=f'Intersections ({len(intersections_details)})',
                            hovertemplate='<b>Intersection avec mur</b><br>' +
                                         'X: %{x:.2f}m<br>' +
                                         'Y: %{y:.2f}m<br>' +
                                         'Z: %{z:.2f}m<extra></extra>'
                        ))
            
                # Configuration de la mise en page
                title_text = f"Plan STL - {n_triangles:,} triangles - {longueur_reelle:.0f}×{largeur_reelle:.0f}×{hauteur_etage * nombre_etages:.1f}m"
                if mode_analyse == "📏 Analyse de Trajectoire" and mode_analyse_trajet and distance > 0:
                    title_text += f" | Trajet: {distance:.2f}m, {murs_traverses} murs"
                
                fig.update_layout(
                    title=title_text,
                    scene=dict(
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)", 
                        zaxis_title="Z (m)",
                        xaxis=dict(range=[limites['x_min'], limites['x_max']]),
                        yaxis=dict(range=[limites['y_min'], limites['y_max']]),
                        zaxis=dict(range=[limites['z_min'], limites['z_max']]),
                        aspectmode='data',
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    height=700,
                    showlegend=True,
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                # Afficher le graphique
                st.plotly_chart(fig, use_container_width=True)
            
            # Informations détaillées
            with st.expander("ℹ️ Informations détaillées du modèle"):
                col_det1, col_det2, col_det3 = st.columns(3)
                
                with col_det1:
                    st.write("**Dimensions réelles:**")
                    st.write(f"• Longueur: {longueur_reelle:.1f} m")
                    st.write(f"• Largeur: {largeur_reelle:.1f} m")
                    st.write(f"• Hauteur totale: {hauteur_etage * nombre_etages:.1f} m")
                    st.write(f"• Hauteur par étage: {hauteur_etage:.1f} m")
                
                with col_det2:
                    st.write("**Limites spatiales (mises à l'échelle):**")
                    st.write(f"• X: {limites['x_min']:.2f} → {limites['x_max']:.2f} m")
                    st.write(f"• Y: {limites['y_min']:.2f} → {limites['y_max']:.2f} m")
                    st.write(f"• Z: {limites['z_min']:.2f} → {limites['z_max']:.2f} m")
                
                with col_det3:
                    st.write("**Statistiques du mesh:**")
                    st.write(f"• Triangles: {n_triangles:,}")
                    st.write(f"• Vertices: {len(x_stl):,}")
                    surface_plan = longueur_reelle * largeur_reelle
                    densite = n_triangles / surface_plan if surface_plan > 0 else 0
                    st.write(f"• Densité: {densite:.1f} tri/m²")
                    
                # Facteurs d'échelle appliqués
                st.write("**Facteurs d'échelle appliqués:**")
                scale_info = f"X: {scale_x:.3f} | Y: {scale_y:.3f} | Z: {scale_z:.3f}"
                st.code(scale_info)
        
            # Suggestions d'optimisation
            if n_triangles > 50000:
                st.warning(f"⚠️ Modèle très complexe ({n_triangles:,} triangles) - Le calcul des intersections peut être lent")
            elif n_triangles > 10000:
                st.info(f"ℹ️ Modèle complexe ({n_triangles:,} triangles) - Analyses détaillées possibles")
            elif n_triangles < 100:
                st.info(f"ℹ️ Modèle simple ({n_triangles} triangles) - Idéal pour les tests rapides")
            else:
                st.success(f"✅ Modèle optimisé ({n_triangles:,} triangles) - Performance excellente")
        
        else:
            st.error("❌ Impossible de charger le modèle STL")
            st.write("Vérifiez que le fichier existe et est un STL valide")

# Nettoyage des fichiers temporaires
if fichier_temporaire:
    try:
        import os
        if os.path.exists(fichier_temporaire):
            os.unlink(fichier_temporaire)
    except:
        pass  # Ignorer les erreurs de nettoyage

# Section d'aide et méthodologie
st.markdown("---")
col_help1, col_help2 = st.columns(2)

with col_help1:
    with st.expander("🎮 Instructions d'utilisation"):
        st.write("""
        **Chargement du fichier STL:**
        1. **Téléchargement**: Glissez-déposez votre fichier .stl ou cliquez pour parcourir
        2. **Fichier local**: Entrez le nom d'un fichier présent dans le dossier de l'application
        
        **Configuration:**
        1. **Dimensions réelles**: Entrez les vraies dimensions de votre bâtiment
        2. **Hauteur d'étage**: Hauteur standard de vos étages (ex: 3m)
        3. **Nombre d'étages**: Nombre total d'étages du bâtiment
        
        **Analyse de trajet:**
        1. Activez le "Mode analyse avancée"
        2. Entrez les coordonnées des points de départ et d'arrivée
        3. Cliquez sur "Analyser le Trajet"
        4. Consultez les résultats: distance, murs traversés, étages
        
        **Navigation 3D:**
        • **Rotation**: Cliquer-glisser avec le bouton gauche
        • **Zoom**: Molette de la souris
        • **Pan**: Shift + cliquer-glisser
        • **Reset**: Double-clic sur le graphique
        """)

with col_help2:
    with st.expander("🔬 Méthodologie de calcul"):
        st.write("""
        **Chargement des fichiers:**
        • Support des fichiers STL téléchargés (jusqu'à 200MB)
        • Validation automatique de la structure STL
        • Nettoyage automatique des fichiers temporaires
        
        **Mise à l'échelle automatique:**
        • Le modèle STL est automatiquement mis à l'échelle selon vos dimensions réelles
        • Les facteurs d'échelle sont calculés pour X, Y et Z indépendamment
        
        **Détection des intersections:**
        • Utilise l'algorithme de Möller-Trumbore pour les intersections rayon-triangle
        • Filtre les intersections multiples du même mur (seuil: 10cm)
        • Calcule la position exacte de chaque intersection
        
        **Calcul des étages:**
        • Étage = ⌊(Z - niveau_sol) / hauteur_étage⌋ + 1
        • Le niveau sol est défini à Z = 0
        • Les coordonnées négatives correspondent au sous-sol
        
        **Limitations:**
        • La précision dépend de la qualité du modèle STL
        • Les murs très fins peuvent ne pas être détectés
        • Performance dégradée avec plus de 50k triangles
        """)

# Informations techniques
with st.expander("⚙️ Informations techniques"):
    st.write("""
    **Bibliothèques utilisées:**
    • `streamlit` - Interface web interactive
    • `plotly` - Visualisation 3D
    • `numpy-stl` - Lecture des fichiers STL
    • `scipy` - Calculs scientifiques avancés
    • `pandas` - Gestion des données tabulaires
    
    **Formats supportés:**
    • Fichiers STL (ASCII ou binaire)
    • Téléchargement direct via l'interface web
    • Fichiers locaux dans le dossier de l'application
    • Taille maximale recommandée: 50MB
    
    **Fonctionnalités de chargement:**
    • Validation automatique des fichiers STL
    • Affichage de la taille et des statistiques
    • Gestion sécurisée des fichiers temporaires
    • Support du glisser-déposer
    
    **Algorithmes:**
    • Intersection rayon-triangle: Möller-Trumbore
    • Index spatial: Centres de triangles (optimisable avec KD-Tree)
    • Filtrage des doublons: Distance euclidienne minimum
    """)

# Footer
st.markdown("---")
st.markdown("**🏗️ Analyseur STL 3D Avancé** - Analyse architecturale précise avec détection d'obstacles")
st.markdown("*Développé pour l'analyse de plans de bâtiments et le calcul de trajectoires optimales*")
