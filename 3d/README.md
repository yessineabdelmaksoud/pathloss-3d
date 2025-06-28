# 🏗️ Analyseur STL 3D Avancé

Une application web interactive pour l'analyse avancée de plans de bâtiments au format STL avec calcul de trajectoires et détection d'obstacles.

## 🚀 Fonctionnalités

### ✨ Nouvelles fonctionnalités avancées
- **Dimensions réelles** : Entrez les vraies dimensions de votre bâtiment
- **Gestion des étages** : Configuration de la hauteur et du nombre d'étages
- **Détection d'obstacles** : Calcul précis du nombre de murs traversés
- **Analyse de trajectoires** : Distance 3D et intersections détaillées
- **Mise à l'échelle automatique** : Le modèle STL s'adapte aux dimensions réelles

### 📊 Analyses disponibles
- Distance 3D entre deux points
- Nombre de murs/obstacles traversés
- Étage de chaque point (départ et arrivée)
- Position exacte des intersections
- Visualisation 3D interactive

## 🛠️ Installation

1. **Cloner ou télécharger** ce projet
2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
3. **Lancer l'application** :
   ```bash
   streamlit run stl_viewer.py
   ```

## 📋 Prérequis

- Python 3.8+
- Fichier STL du plan de bâtiment
- Navigateur web moderne

## 🎯 Comment utiliser

### 1. Configuration du plan
- Chargez votre fichier STL (par défaut : `modell_stl.stl`)
- Entrez les **dimensions réelles** de votre bâtiment :
  - Longueur (m)
  - Largeur (m)
  - Hauteur par étage (m)
  - Nombre d'étages total

### 2. Analyse de trajet
1. Activez le **"Mode analyse avancée"**
2. Entrez les coordonnées du **point de départ** (X1, Y1, Z1)
3. Entrez les coordonnées du **point d'arrivée** (X2, Y2, Z2)
4. Cliquez sur **"🔍 Analyser le Trajet"**

### 3. Résultats obtenus
- **Distance 3D** exacte entre les points
- **Nombre de murs traversés** avec positions précises
- **Étage** de chaque point
- **Visualisation 3D** avec trajectoire et intersections

## 🔬 Méthodologie technique

### Algorithmes utilisés
- **Intersection rayon-triangle** : Möller-Trumbore
- **Filtrage des doublons** : Seuil de distance minimum (10cm)
- **Calcul d'étages** : Basé sur la hauteur Z et la hauteur d'étage

### Mise à l'échelle
Le modèle STL est automatiquement mis à l'échelle selon vos dimensions réelles :
```
Facteur X = Longueur_réelle / Longueur_STL
Facteur Y = Largeur_réelle / Largeur_STL  
Facteur Z = Hauteur_totale / Hauteur_STL
```

### Performance
- **Optimal** : 1k-10k triangles
- **Acceptable** : 10k-50k triangles
- **Lent** : >50k triangles

## 📁 Structure du projet

```
3d/
├── stl_viewer.py       # Application principale
├── requirements.txt    # Dépendances Python
├── modell_stl.stl     # Fichier STL exemple
└── README.md          # Ce fichier
```

## 🔧 Bibliothèques utilisées

- **streamlit** (≥1.28.0) : Interface web
- **plotly** (≥5.15.0) : Visualisation 3D
- **numpy-stl** (≥3.0.0) : Lecture STL
- **scipy** (≥1.9.0) : Calculs scientifiques
- **pandas** (≥1.5.0) : Gestion des données
- **numpy** (≥1.21.0) : Calculs numériques
- **scikit-learn** (≥1.1.0) : Outils d'analyse

## 🎨 Interface utilisateur

### Panneau de configuration (droite)
- Configuration des dimensions réelles
- Options d'affichage (couleur, opacité, wireframe)
- Paramètres d'analyse (points de départ/arrivée)

### Visualisation principale (gauche)
- Modèle 3D interactif
- Indicateurs d'étages
- Points d'analyse et trajectoire
- Intersections avec obstacles

## 📊 Exemple d'analyse

Pour un bâtiment de **50m × 30m × 9m** (3 étages) :

**Points analysés :**
- Départ : (5.0, 5.0, 1.5) → Étage 1
- Arrivée : (45.0, 25.0, 7.5) → Étage 3

**Résultats :**
- Distance 3D : 42.43m
- Murs traversés : 3
- Différence d'étages : 2

## 🐛 Dépannage

### Erreurs courantes
1. **"Impossible de charger le modèle STL"**
   - Vérifiez que le fichier `.stl` existe
   - Contrôlez le format (ASCII ou binaire)

2. **Application lente**
   - Réduisez le nombre de triangles du modèle
   - Utilisez un fichier STL optimisé

3. **Résultats incohérents**
   - Vérifiez les dimensions réelles saisies
   - Contrôlez les coordonnées des points

## 🔄 Améliorations futures

- Support de formats 3D supplémentaires (OBJ, PLY)
- Algorithmes d'optimisation de chemin
- Export des résultats (PDF, CSV)
- API REST pour intégration
- Support multi-fichiers

## 📝 Licence

Ce projet est développé pour l'analyse architecturale et l'optimisation de trajectoires en milieu bâti.

---

**Développé avec ❤️ pour l'analyse architecturale avancée**
