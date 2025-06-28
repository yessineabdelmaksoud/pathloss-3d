#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple du chargement STL
"""

try:
    from stl import mesh
    import numpy as np
    import os
    
    print("✅ Imports réussis")
    
    # Tester le chargement du fichier
    filename = "modell_stl.stl"
    if os.path.exists(filename):
        print(f"✅ Fichier {filename} trouvé")
        
        # Charger le mesh
        stl_mesh = mesh.Mesh.from_file(filename)
        print(f"✅ Fichier STL chargé avec succès")
        print(f"   - Nombre de triangles: {len(stl_mesh.vectors)}")
        print(f"   - Forme des vecteurs: {stl_mesh.vectors.shape}")
        
        # Tester la copie
        vectors_copy = stl_mesh.vectors.copy()
        print(f"✅ Copie des vecteurs réussie")
        
        # Tester les calculs de base
        vertices = stl_mesh.vectors.reshape(-1, 3)
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        
        limites = {
            'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
            'y_min': float(np.min(y)), 'y_max': float(np.max(y)),
            'z_min': float(np.min(z)), 'z_max': float(np.max(z))
        }
        
        print(f"✅ Calculs de limites réussis:")
        print(f"   - X: {limites['x_min']:.2f} → {limites['x_max']:.2f}")
        print(f"   - Y: {limites['y_min']:.2f} → {limites['y_max']:.2f}")
        print(f"   - Z: {limites['z_min']:.2f} → {limites['z_max']:.2f}")
        
        print("\n🎉 Tous les tests sont passés ! Le fichier STL peut être chargé correctement.")
        
    else:
        print(f"❌ Fichier {filename} non trouvé")
        print("   Vérifiez que le fichier STL est dans le bon répertoire")
        
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("   Installez les dépendances avec: pip install numpy-stl")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("   Type d'erreur:", type(e).__name__)
