# Dossier Model

## Description

Ce dossier contient les informations et instructions pour obtenir et utiliser le modèle d'intelligence artificielle du projet PokéBrAIn.

### Informations sur le Modèle

- **Architecture** : ResNet-18 (pré-entraîné sur ImageNet)
- **Type** : Classification d'images (reconnaissance de cartes Pokémon)
- **Nombre de classes** : ~18 831 cartes Pokémon distinctes
- **Accuracy de validation** : 97,96%
- **Framework** : PyTorch
- **Format** : `.pth` (TorchScript pour optimisation)
- **Taille du fichier** : 240 Mo

---

## 📥 Téléchargement du Modèle

⚠️ **Important** : En raison de sa taille (240 Mo), le modèle n'est **pas inclus** dans le dépôt GitHub.

### 🔗 Lien de Téléchargement

**Télécharger le modèle depuis Google Drive** :

👉 **[Cliquez ici pour télécharger best_pokemon_model.pth](https://drive.google.com/drive/folders/1BUG8RTBIyn6XU4KDBwZ46T7fK8_i4Rmd?usp=drive_link)**

### 📍 Instructions d'Installation

1. **Téléchargez** le fichier `best_pokemon_model.pth` depuis le lien ci-dessus
2. **Placez** le fichier dans ce dossier `model/` à la racine du projet
3. **Vérifiez** que le chemin est : `PokeBrAIn/model/best_pokemon_model.pth`

---

## 🛠️ Utilisation du Modèle

### Chargement du Modèle en Python

```python
import torch
from torchvision import models

# Charger l'architecture ResNet-18
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 18831)  # Adapter pour 18831 classes

# Charger les poids entraînés
model.load_state_dict(torch.load('model/best_pokemon_model.pth'))
model.eval()  # Mode évaluation

print("Modèle chargé avec succès !")
```

### Utilisation pour la Prédiction

```python
import torch
from PIL import Image
from torchvision import transforms

# Préparation de l'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Charger et préparer l'image
image = Image.open('path/to/pokemon_card.jpg')
image_tensor = transform(image).unsqueeze(0)

# Prédiction
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    print(f"Carte prédite : classe {predicted.item()}")
```

---

## 📊 Structure du Modèle

Le modèle sauvegardé contient :

- **Les poids du réseau de neurones** : Paramètres entraînés sur ~370 000 images
- **L'architecture ResNet-18 modifiée** : Couche finale adaptée pour 18 831 classes
- **Les paramètres d'entraînement** : Optimisation pour Jetson Nano

---

## 📝 Dataset d'Entraînement

- **Source** : API pokemontcg.io
- **Images originales** : ~18 831 cartes Pokémon
- **Après data augmentation** : ~370 000 images
- **Augmentations appliquées** :
  - Variations de luminosité
  - Rotations
  - Changements de couleur
  - Modifications de fond
  - Zoom et recadrage

---

## ⚙️ Configuration et Optimisation

### Pour Jetson Nano

Le modèle a été converti en TorchScript pour optimiser les performances sur Jetson Nano :

```python
# Conversion en TorchScript
model.eval()
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model/best_pokemon_model.pth')
```

### Prérequis

- **Python** : 3.7+
- **PyTorch** : 1.8+ (compatible avec la version utilisée sur Jetson)
- **torchvision** : 0.9+
- **PIL/Pillow** : Pour le traitement d'images

---

## 🚨 Notes Importantes

### Compatibilité

⚠️ Assurez-vous d'utiliser la **même version de PyTorch** pour charger le modèle que celle utilisée pour l'entraînement.

### Sauvegarde

💾 Conservez toujours une **copie de sauvegarde** du modèle entraîné.

### Git et Fichiers Volumineux

🚫 Le fichier `.pth` est **ignoré par Git** (voir `.gitignore`) car il dépasse les limites de GitHub (100 Mo max).

---

## 🔄 Ré-entraînement

Si vous souhaitez ré-entraîner le modèle :

1. Préparez votre dataset (voir `scripts/telechargement_pokemon.py` et `scripts/data_aug_v2.py`)
2. Utilisez le script d'entraînement approprié
3. Sauvegardez le nouveau modèle dans ce dossier
4. Mettez à jour ce README avec les nouvelles performances

---

## 📞 Contact

Pour toute question concernant le modèle ou son utilisation :

- **Issues GitHub** : [Ouvrir une issue](https://github.com/AlexisXueref/PokéBrAIn/issues)
- **Documentation complète** : Voir `docs/Rapport-PokéBrAIn.pdf`

---

**Modèle PokeBrAIn** - ResNet-18 pour reconnaissance de cartes Pokémon 🎴🧠
