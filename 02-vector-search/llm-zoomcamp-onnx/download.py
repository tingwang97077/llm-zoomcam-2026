import os  # accès aux variables d'environnement
import shutil  # opérations sur les fichiers (copie)
import logging  # gestion des messages de log
from pathlib import Path  # manipulation des chemins de façon orientée objet
from huggingface_hub import hf_hub_download, list_repo_files  # télécharger un fichier / lister les fichiers d'un repo HF

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # désactive l'envoi de télémétrie par huggingface_hub
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # ne garde que les logs de niveau ERROR (silencieux)

ONNX_CANDIDATES = [  # noms de fichiers ONNX possibles, testés dans l'ordre de priorité
    "onnx/model.onnx",  # emplacement le plus courant
    "onnx/encoder_model.onnx",  # variante encodeur (modèles seq2seq)
    "model.onnx",  # modèle à la racine du repo
]

def download(repo, dest="models"):  # télécharge le modèle ONNX du repo HF dans le dossier dest
    dest = Path(dest) / repo  # construit le chemin de destination : models/<repo>
    dest.mkdir(parents=True, exist_ok=True)  # crée le dossier (et parents), sans erreur s'il existe déjà

    files = list_repo_files(repo_id=repo)  # liste tous les fichiers disponibles dans le repo HF
    onnx_file = next((c for c in ONNX_CANDIDATES if c in files), None)  # 1er candidat ONNX présent, sinon None
    if not onnx_file:  # aucun fichier ONNX trouvé dans le repo
        raise FileNotFoundError(f"No ONNX model found in {repo}")  # on arrête avec une erreur explicite

    for remote, local in [  # paires (nom distant sur HF, nom local à enregistrer)
        ("tokenizer.json", "tokenizer.json"),  # le tokenizer, gardé sous le même nom
        (onnx_file, "model.onnx"),  # le modèle ONNX, renommé uniformément en model.onnx
    ]:
        src = hf_hub_download(repo_id=repo, filename=remote)  # télécharge le fichier dans le cache HF, renvoie son chemin
        dst = dest / local  # chemin de destination final
        if not dst.exists():  # ne copie que si le fichier n'est pas déjà présent
            shutil.copy2(src, dst)  # copie le fichier (en conservant les métadonnées) du cache vers dest
            print(f"  saved {dst}")  # confirme l'enregistrement
        else:
            print(f"  exists {dst}")  # signale que le fichier existait déjà

    onnx_ext = onnx_file + "_data"  # nom du fichier de poids externe éventuel (gros modèles ONNX)
    if onnx_ext in files:  # ce fichier de données externe existe dans le repo
        src = hf_hub_download(repo_id=repo, filename=onnx_ext)  # le télécharge
        dst = dest / "model.onnx_data"  # nom local attendu à côté de model.onnx
        if not dst.exists():  # idem : on évite de recopier inutilement
            shutil.copy2(src, dst)  # copie le fichier de poids
            print(f"  saved {dst}")  # confirme l'enregistrement
        else:
            print(f"  exists {dst}")  # signale qu'il existait déjà

if __name__ == "__main__":  # exécuté seulement si le script est lancé directement
    download("Xenova/all-MiniLM-L6-v2")  # télécharge le modèle d'embedding all-MiniLM-L6-v2 au format ONNX
