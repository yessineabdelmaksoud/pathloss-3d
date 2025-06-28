import streamlit as st
import plotly.graph_objects as go
import numpy as np
from stl import mesh
import math
from scipy.spatial.distance import cdist
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Analyseur STL 3D Avancé",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Analyseur de Plan STL 3D Avancé")
st.markdown("Analysez votre plan de bâtiment STL avec calcul de trajectoires et détection d'obstacles")

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
def charger_modele_stl(fichier_plan, scale_factor_x=1.0, scale_factor_y=1.0, scale_factor_z=1.0):
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
    
    # Fichier STL
    fichier_stl = st.text_input("Fichier STL", value="modell_stl.stl")
    
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
    st.subheader("🎨 Affichage")
    couleur_stl = st.color_picker("Couleur du modèle", "#00BFFF")
    opacite = st.slider("Opacité", 0.1, 1.0, 0.8, 0.1)
    afficher_wireframe = st.checkbox("Affichage filaire", False)
    
    # Analyse de trajet avancée
    st.subheader("📍 Analyse de Trajet Avancée")
    mode_analyse = st.checkbox("Mode analyse avancée", False)
    
    # Initialiser la variable analyser
    analyser = False
    
    if mode_analyse:
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
    
    # Calculer les facteurs d'échelle si on a les dimensions réelles
    with st.spinner("Chargement du modèle STL..."):
        # Charger le modèle avec échelle 1:1 pour obtenir les dimensions originales
        stl_data_original = charger_modele_stl(fichier_stl, 1.0, 1.0, 1.0)
    
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
        stl_data = charger_modele_stl(fichier_stl, scale_x, scale_y, scale_z)
        
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
            
            # Effectuer l'analyse si demandée
            if mode_analyse:
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
            
            # Créer la figure Plotly
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
            
            # Ajouter les points d'analyse si le mode est activé
            if mode_analyse:
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
                    mode='lines',
                    line=dict(width=8, color='blue'),
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
            if mode_analyse and distance > 0:
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

# Section d'aide et méthodologie
st.markdown("---")
col_help1, col_help2 = st.columns(2)

with col_help1:
    with st.expander("🎮 Instructions d'utilisation"):
        st.write("""
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
    • Le fichier doit être dans le même dossier que l'application
    • Recommandé: modèles avec 1k-10k triangles pour une performance optimale
    
    **Algorithmes:**
    • Intersection rayon-triangle: Möller-Trumbore
    • Index spatial: Centres de triangles (optimisable avec KD-Tree)
    • Filtrage des doublons: Distance euclidienne minimum
    """)

# Footer
st.markdown("---")
st.markdown("**🏗️ Analyseur STL 3D Avancé** - Analyse architecturale précise avec détection d'obstacles")
st.markdown("*Développé pour l'analyse de plans de bâtiments et le calcul de trajectoires optimales*")
