# Test de syntaxe Python
import ast

def check_syntax(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Vérifier la syntaxe en compilant en AST
        ast.parse(source)
        print(f"✅ {filename} - Syntaxe Python correcte!")
        return True
        
    except SyntaxError as e:
        print(f"❌ {filename} - Erreur de syntaxe:")
        print(f"   Ligne {e.lineno}: {e.text.strip() if e.text else 'N/A'}")
        print(f"   Erreur: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ {filename} - Erreur: {e}")
        return False

if __name__ == "__main__":
    check_syntax("stl_viewer.py")
