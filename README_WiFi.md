# 📡 STL Viewer avec Prédiction de Propagation WiFi

## 🎯 Fonctionnalités

### Mode Analyse de Trajectoire
- Analyse du trajet entre deux points
- Calcul de la distance 3D
- Comptage des murs traversés (ray tracing)
- Détection des changements d'étages

### Mode Prédiction Propagation WiFi
- **Point WiFi configurable** : Choisissez la position de votre émetteur WiFi
- **Fréquence ajustable** : Support 2.4 GHz et 5 GHz
- **Grille automatique** : Génération d'une grille de points sur tout le plan
- **Calcul intelligent** : Distance, nombre de murs (ray tracing), différence d'étage
- **Prédiction IA** : Modèle XGBoost pré-entraîné pour le path loss
- **Heatmap 3D** : Visualisation interactive avec dégradé de couleurs
- **Analyse de couverture** : Statistiques détaillées des zones de couverture

## 🚀 Utilisation

### 1. Lancement de l'application
```bash
# Option 1: Script automatique
run_wifi_analyzer.bat

# Option 2: Commande directe
streamlit run stl_viewer.py
```

### 2. Mode Prédiction WiFi

#### Étape 1: Chargement du modèle
1. Sélectionnez "📡 Prédiction Propagation WiFi"
2. Cliquez sur "🔄 Charger le Modèle XGBoost"
3. Attendez la confirmation de chargement

#### Étape 2: Configuration du point WiFi
- **Position X, Y, Z** : Coordonnées du point d'émission WiFi
- **Fréquence** : Choisissez entre 2400 MHz (2.4 GHz) ou 5000 MHz (5 GHz)

#### Étape 3: Paramètres de la grille
- **Résolution** : Espacement entre les points de calcul (0.5 à 3.0 m)
- **Distance max** : Zone d'analyse autour du point WiFi
- **Niveaux Z** : Nombre de niveaux verticaux à analyser
- **Z max** : Hauteur maximum d'analyse

#### Étape 4: Génération de la heatmap
1. Cliquez sur "🌈 Générer la Heatmap 3D"
2. Attendez le calcul des features (distance, murs, étages)
3. Le modèle IA prédit le path loss pour chaque point
4. La heatmap 3D s'affiche automatiquement

#### Étape 5: Analyse des résultats
- **Statistiques de couverture** : Min/Max/Moyen du path loss
- **Zones de qualité** :
  - 🟢 Bon signal : < 70 dB
  - 🟡 Signal moyen : 70-90 dB  
  - 🔴 Signal faible : > 90 dB
- **Paramètres de visualisation** : Seuil, taille des points, échelle de couleurs

## 📊 Interprétation des Résultats

### Path Loss (Perte de propagation)
- **Faible path loss** (30-60 dB) : Excellent signal, proche de l'émetteur
- **Path loss moyen** (60-80 dB) : Bon signal, utilisation normale
- **Path loss élevé** (80-100 dB) : Signal dégradé, vitesse réduite
- **Path loss très élevé** (>100 dB) : Signal faible, connexion difficile

### Facteurs d'influence
1. **Distance** : Plus la distance augmente, plus le path loss augmente
2. **Nombre de murs** : Chaque mur traversé ajoute ~10-20 dB de path loss
3. **Différence d'étages** : Les planchers causent une atténuation importante
4. **Fréquence** : Les hautes fréquences (5 GHz) s'atténuent plus rapidement

## 🔧 Configuration Avancée

### Modèle de Machine Learning
- **Fichier** : `model/xgboost_radio_propagation_model.pkl`
- **Features** : distance, numwall, etage, frequence
- **Performance** : RMSE ~3-5 dB, R² > 0.85

### Paramètres de Ray Tracing
- **Résolution spatiale** : Configurable via la grille
- **Algorithme** : Möller-Trumbore pour intersection rayon-triangle
- **Élimination doublons** : Distance minimum 10cm entre intersections

## 📁 Structure des Fichiers

```
3d/
├── stl_viewer.py              # Application principale
├── run_wifi_analyzer.bat      # Script de lancement
└── modell_stl.stl            # Plan STL exemple

model/
├── xgboost_radio_propagation_model.pkl  # Modèle pré-entraîné
├── model.ipynb               # Notebook d'entraînement
└── requirements.txt          # Dépendances ML

data/
└── radio_dataset_notebook.csv  # Dataset d'entraînement
```

## ⚡ Conseils d'Optimisation

### Performance
- **Résolution** : Utilisez 1-2m pour un bon compromis vitesse/précision
- **Zone d'analyse** : Limitez la distance max selon vos besoins
- **Points affichés** : Utilisez le seuil pour réduire la charge graphique

### Précision
- **Modèle 3D détaillé** : Plus le STL est précis, meilleur est le ray tracing
- **Dimensions réelles** : Configurez correctement les dimensions du bâtiment
- **Hauteur des étages** : Ajustez selon votre bâtiment (2.5-4m typique)

## 🛠️ Dépendances

```
streamlit
plotly
numpy
pandas
scipy
stl
xgboost
scikit-learn
pickle
```

## 📞 Support

Pour toute question ou amélioration, consultez le code source dans `stl_viewer.py` ou le notebook de ML dans `model/model.ipynb`.
