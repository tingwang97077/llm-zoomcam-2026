import numpy as np  
import onnxruntime as ort  # moteur d'inférence pour exécuter le modèle ONNX
from tokenizers import Tokenizer  # tokenizer rapide de Hugging Face
from pathlib import Path  # manipulation des chemins de façon orientée objet


class Embedder:  # encapsule le tokenizer + le modèle ONNX pour produire des embeddings
    def __init__(self, path="models/Xenova/all-MiniLM-L6-v2"):  # dossier contenant le modèle téléchargé
        path = Path(path)  # convertit la chaîne en objet Path
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))  # charge le tokenizer depuis tokenizer.json
        self.session = ort.InferenceSession(  # crée la session d'inférence ONNX
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]  # charge model.onnx, exécution sur CPU
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}  # noms des entrées attendues par le modèle

    def encode(self, text, normalize=True):  # encode un seul texte en vecteur
        return self.encode_batch([text], normalize=normalize)[0]  # délègue au batch puis retourne le 1er résultat

    def encode_batch(self, texts, normalize=True):  # encode une liste de textes en une matrice de vecteurs
        self.tokenizer.enable_padding()  # active le padding pour aligner les séquences à la même longueur
        encoded = self.tokenizer.encode_batch(texts)  # tokenise tous les textes d'un coup
        feed = {}  # dictionnaire des entrées à fournir au modèle ONNX
        if "input_ids" in self.input_names:  # si le modèle attend les identifiants de tokens
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)  # matrice des IDs de tokens
        if "attention_mask" in self.input_names:  # si le modèle attend le masque d'attention
            feed["attention_mask"] = np.array(  # 1 pour les vrais tokens, 0 pour le padding
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:  # si le modèle attend les types de tokens (segments)
            feed["token_type_ids"] = np.array(  # généralement 0 (phrase unique)
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self.session.run(None, feed)[0]  # exécute le modèle ; récupère les états cachés (token embeddings)
        mask = feed["attention_mask"][..., None]  # ajoute une dimension pour le broadcast sur la dimension cachée
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)  # mean pooling : moyenne des tokens réels (hors padding)
        if normalize:  # si normalisation demandée
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)  # normalise chaque vecteur en norme L2 (norme 1)
        return pooled  # retourne la matrice des embeddings (un vecteur par texte)
