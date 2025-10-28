# Dossier Model

## Description

Ce dossier est destiné à accueillir les versions sauvegardées du modèle d'intelligence artificielle du projet PokeBrAIn. Les fichiers de modèle sont généralement au format `.pth` (PyTorch).

## Structure

Les fichiers de modèle sauvegardés incluent :
- Les poids du réseau de neurones
- L'architecture du modèle
- Les paramètres d'entraînement

## Instructions

### Si les fichiers .pth ne sont pas inclus dans le dépôt GitHub

En raison de leur taille importante, les fichiers de modèle (.pth) ne sont généralement pas inclus directement dans le dépôt GitHub. Voici comment obtenir ou générer le modèle :

#### Option 1 : Télécharger un modèle pré-entraîné

Si un modèle pré-entraîné est disponible :
1. Téléchargez le fichier depuis [insérer le lien de téléchargement]
2. Placez le fichier `.pth` dans ce dossier `model/`
3. Vérifiez que le nom du fichier correspond à celui attendu par le code

#### Option 2 : Entraîner votre propre modèle

Pour générer un nouveau modèle :
1. Suivez les instructions d'installation du projet dans le README principal
2. Préparez les données d'entraînement (voir documentation)
3. Lancez le script d'entraînement :
   ```bash
   python train.py
   ```
4. Le modèle entraîné sera automatiquement sauvegardé dans ce dossier

### Utilisation du modèle

Pour charger et utiliser un modèle sauvegardé :
```python
import torch
from model import YourModelClass

# Charger le modèle
model = YourModelClass()
model.load_state_dict(torch.load('model/votre_modele.pth'))
model.eval()
```

## Notes importantes

- **Taille des fichiers** : Les fichiers .pth peuvent être volumineux (plusieurs Mo à plusieurs Go)
- **Git LFS** : Pour versionner de gros fichiers de modèle, considérez l'utilisation de Git Large File Storage (LFS)
- **Compatibilité** : Assurez-vous d'utiliser la même version de PyTorch pour charger le modèle que celle utilisée pour le sauvegarder
- **Sauvegarde** : Conservez toujours une copie de sauvegarde de vos modèles entraînés

## Contact

Pour toute question concernant les modèles ou leur utilisation, veuillez consulter la documentation principale ou ouvrir une issue sur le dépôt GitHub.
