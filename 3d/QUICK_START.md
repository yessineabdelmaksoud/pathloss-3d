# 🚀 Installation et Démarrage Rapide

## Installation des dépendances

Ouvrez un terminal dans le dossier du projet et exécutez :

```bash
pip install -r requirements.txt
```

## Lancement de l'application

```bash
streamlit run stl_viewer.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : http://localhost:8501

## Test rapide

1. L'application charge automatiquement le fichier `modell_stl.stl`
2. Ajustez les dimensions réelles de votre bâtiment
3. Activez le "Mode analyse avancée"
4. Entrez deux points et cliquez sur "Analyser le Trajet"
5. Consultez les résultats !

## Dépannage

Si vous rencontrez des erreurs :
1. Vérifiez que Python 3.8+ est installé : `python --version`
2. Vérifiez que pip est à jour : `pip install --upgrade pip`
3. Réinstallez les dépendances : `pip install -r requirements.txt --force-reinstall`

## Structure finale du projet

```
3d/
├── stl_viewer.py       # ✅ Application principale corrigée
├── requirements.txt    # ✅ Dépendances mises à jour
├── modell_stl.stl     # ✅ Fichier STL d'exemple
├── README.md          # ✅ Documentation complète
└── QUICK_START.md     # ✅ Ce guide de démarrage
```

## Fonctionnalités corrigées

✅ **Configuration des dimensions réelles**
✅ **Calcul automatique des étages**  
✅ **Détection des murs traversés**
✅ **Analyse de trajectoires 3D**
✅ **Visualisation interactive**
✅ **Interface utilisateur intuitive**

## Commande complète

```bash
# Installation
pip install streamlit plotly numpy-stl numpy scipy pandas scikit-learn

# Lancement
streamlit run stl_viewer.py
```

---

**L'application est maintenant entièrement fonctionnelle !** 🎉
