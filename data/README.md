# Générateur de Dataset Propagation Radio

Ce dossier contient un générateur de dataset réaliste pour l'analyse de la propagation radio en environnement indoor.

## 📋 Contenu

- `generation.py` : Script principal de génération du dataset
- `requirements.txt` : Dépendances Python requises
- `run_generation.bat` : Script d'exécution automatique (Windows)

## 🎯 Dataset Généré

Le script génère un dataset avec les colonnes suivantes :

| Colonne | Description | Unité | Plage typique |
|---------|-------------|-------|---------------|
| `distance` | Distance entre émetteur et récepteur | mètres | 0.5 - 100 m |
| `numwall` | Nombre de murs traversés | entier | 0 - 10 |
| `etage` | Différence d'étages | entier | 0 - 4 |
| `frequence` | Fréquence du signal | MHz | 800 - 6000 MHz |
| `pathloss` | Perte de propagation (variable cible) | dB | 30 - 150 dB |

## 🔬 Modèles Mathématiques Utilisés

### 1. Perte en Espace Libre (Friis)
```
PL_free = 20*log10(d) + 20*log10(f) + 32.45
```

### 2. Modèle One-Slope Indoor
```
PL = PL_ref + 10*n*log10(d/d_ref)
```
avec n = 2.8 (exposant de propagation indoor)

### 3. Pertes d'Obstacles
- **Murs** : 3.5 dB par mur traversé
- **Étages** : 15 dB par étage traversé

### 4. Effets Environnementaux
- **Effet de densité** : 0.5 * num_walls * log10(distance)
- **Multi-trajet** : 2.0 * sin²(2π * f * d / c)
- **Résonance** : 0.3 * sin(π * f / 100)

### 5. Variabilité Réaliste
- **Shadowing** : Log-normal (σ = 4.0 dB)
- **Bruit de mesure** : Gaussien (σ = 2.5 dB)

## 🚀 Utilisation

### Méthode 1 : Script automatique (Windows)
```bash
run_generation.bat
```

### Méthode 2 : Manuel
```bash
# Installation des dépendances
pip install -r requirements.txt

# Génération du dataset
python generation.py
```

### Méthode 3 : Personnalisé
```python
from generation import RadioPropagationDataGenerator

# Créer le générateur
generator = RadioPropagationDataGenerator(random_seed=42)

# Générer 5000 échantillons
df = generator.generate_dataset(n_samples=5000, save_path='mon_dataset.csv')

# Analyser le dataset
generator.analyze_dataset(df)
```

## 📊 Sorties Générées

1. **radio_propagation_dataset.csv** : Dataset en format CSV
2. **radio_propagation_dataset.xlsx** : Dataset en format Excel
3. **dataset_analysis.png** : Visualisations et analyses

## 📈 Analyses Incluses

Le script génère automatiquement :

- **Statistiques descriptives** complètes
- **Matrice de corrélation** entre variables
- **Validation physique** des relations
- **12 visualisations** différentes :
  - Distributions des variables
  - Relations bivariées
  - Heatmaps
  - Graphiques 3D
  - Boxplots

## 🎛️ Paramètres Configurables

Dans la classe `RadioPropagationDataGenerator` :

```python
# Paramètres physiques
self.wall_loss_db = 3.5          # Perte par mur (dB)
self.floor_loss_db = 15.0        # Perte par étage (dB)
self.reference_freq = 2400       # Fréquence de référence (MHz)

# Paramètres de bruit
self.noise_std = 2.5             # Bruit de mesure (dB)
self.shadow_fading_std = 4.0     # Shadowing (dB)
```

## 🔍 Validation du Dataset

Le script valide automatiquement :

- ✅ Corrélation distance-pathloss > 0.7
- ✅ Corrélation murs-pathloss > 0.3
- ✅ Corrélation étages-pathloss > 0.2
- ✅ Plage de pathloss réaliste (50-120 dB)

## 🎯 Applications

Ce dataset est idéal pour :

- **Machine Learning** : Prédiction du pathloss
- **Optimisation réseau** : Placement d'antennes
- **Simulation radio** : Validation de modèles
- **Recherche** : Analyse de propagation indoor

## 📚 Références

- Modèle de Friis pour l'espace libre
- ITU-R P.1238 : Propagation indoor
- IEEE 802.11 : Standards WiFi
- Rappaport, T.S. : "Wireless Communications"

## ⚙️ Personnalisation

Pour modifier les scénarios de génération, éditez la méthode `generate_realistic_scenarios()` dans `generation.py`.

Pour ajuster les modèles physiques, modifiez les méthodes de calcul du pathloss.

## 🐛 Dépannage

**Erreur d'import** : Vérifiez l'installation des dépendances
```bash
pip install -r requirements.txt
```

**Erreur de visualisation** : Installez matplotlib avec backend
```bash
pip install matplotlib[tk]
```

**Mémoire insuffisante** : Réduisez `n_samples` dans le script
