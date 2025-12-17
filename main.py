import os
import sys
import time

def run_step(step_name, command):
    print(f"\n{'='*50}")
    print(f">>> LANCEMENT : {step_name}")
    print(f"{'='*50}\n")
    exit_code = os.system(command)
    if exit_code != 0:
        print(f"\n❌ ERREUR lors de l'étape : {step_name}")
        sys.exit(exit_code)
    print(f"\n✅ SUCCÈS : {step_name} terminé.")

def main():
    print("🚀 Démarrage du pipeline complet LEBI PROJECT...")
    
    # 1. Scraping
    # Note: Cela peut prendre du temps (+10 mins)
    run_step("Phase 1 : Extraction (Scraping)", "python LEBI_Scrapping.py")
    
    # 2. ETL
    run_step("Phase 2 : Préparation (ETL)", "python LEBI_ETL.py")
    
    # 3. Modelisation
    run_step("Phase 3 : Modélisation (ML)", "python Phase3_Modelisation.py")
    
    # 4. Dashboard
    print(f"\n{'='*50}")
    print(">>> LANCEMENT : Phase 4 : Dashboard")
    print("    Le serveur va démarrer. Accédez à http://127.0.0.1:8050/")
    print(f"{'='*50}\n")
    os.system("python Phase4_Dashboard.py")

if __name__ == "__main__":
    main()
