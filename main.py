"""LEBI Project - Complete Pipeline Runner

Runs all phases of the LEBI project sequentially.

Usage:
    python main.py              # Run all phases
    python main.py --skip-scraping  # Skip scraping phase
"""
import os
import sys
import argparse

def run_step(step_name, command):
    """Execute a pipeline step and handle errors."""
    print(f"\n{'='*60}")
    print(f">>> LANCEMENT : {step_name}")
    print(f"{'='*60}\n")
    exit_code = os.system(command)
    if exit_code != 0:
        print(f"\n❌ ERREUR lors de l'étape : {step_name}")
        print(f"   Commande: {command}")
        print(f"   Code de sortie: {exit_code}")
        sys.exit(exit_code)
    print(f"\n✅ SUCCÈS : {step_name} terminé.")

def main():
    parser = argparse.ArgumentParser(description='Run LEBI PROJECT pipeline')
    parser.add_argument('--skip-scraping', action='store_true', 
                       help='Skip Phase 1 (scraping) if data already exists')
    args = parser.parse_args()
    
    print("🚀 Démarrage du pipeline complet LEBI PROJECT...")
    print("   Architecture modulaire avec src/")
    
    # Phase 1: Scraping (optional)
    if not args.skip_scraping:
        run_step("Phase 1 : Web Scraping", "python run_scraping.py")
    else:
        print("\n⏭️  Phase 1 : Scraping ignorée (données existantes utilisées)")
    
    # Phase 2: ETL
    run_step("Phase 2 : ETL (Nettoyage)", "python run_etl.py")
    
    # Phase 3: Machine Learning
    run_step("Phase 3 : Machine Learning", "python run_ml.py")
    
    # Phase 4: Dashboard
    print(f"\n{'='*60}")
    print(">>> LANCEMENT : Phase 4 : Dashboard Interactif")
    print("    Le serveur Dash va démarrer.")
    print("    Accédez à : http://127.0.0.1:8050/")
    print("    Appuyez sur Ctrl+C pour arrêter le serveur")
    print(f"{'='*60}\n")
    os.system("python run_dashboard.py")

if __name__ == "__main__":
    main()
