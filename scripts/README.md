# Scripts

Ce dossier regroupe tous les scripts Python nécessaires au fonctionnement du projet PokéBrAIn.

## Contenu

Ce répertoire contient les scripts suivants :

---

### Scripts de préparation des données

#### **`telechargement_pokemon.py`**
- **Description** : Programme de téléchargement de toutes les cartes Pokémon depuis l'API pokemontcg.io
- **Fonction** : Récupère automatiquement les images et métadonnées de toutes les cartes Pokémon disponibles dans l'API
- **Sortie** : Dataset de base contenant ~18 831 images de cartes Pokémon distinctes
- **Utilisation** : À exécuter en premier pour créer le dataset initial

#### **`data_aug_v2.py`**
- **Description** : Programme de data augmentation du dataset de base
- **Fonction** : Applique des transformations aux images (rotation, zoom, changements de luminosité, modifications de couleur, etc.) pour augmenter la taille du dataset
- **Sortie** : Dataset augmenté de ~18 831 images originales → ~370 000 images (facteur d'augmentation de 25x)
- **Utilisation** : À exécuter après `telechargement_pokemon.py` pour enrichir le dataset d'entraînement

---

### Scripts de traitement et classification

#### **`classification_v2.py`**
- **Description** : Programme qui scanne et classe les images de cartes Pokémon enregistrées dans le même dossier
- **Fonction** : 
  - Lit les images présentes dans le dossier d'exécution
  - Utilise le modèle ResNet-18 entraîné pour identifier chaque carte
  - Classe les cartes détectées et affiche les résultats
- **Modèle utilisé** : `model/best_pokemon_model.pth`
- **Sortie** : Résultats de classification pour chaque image analysée
- **Utilisation** : Pour analyser des images de cartes sauvegardées en mode batch

#### **`test_camera_v2.py`**
- **Description** : Programme utilisant la caméra pour détecter et classer les cartes Pokémon en temps réel
- **Fonction** :
  - Capture le flux vidéo depuis la caméra (Logitech C505e)
  - Détecte les cartes Pokémon présentes dans le champ de vision
  - Utilise le modèle ResNet-18 pour identifier la carte en temps réel
  - Classe automatiquement la carte dans l'inventaire
- **Modèle utilisé** : `model/best_pokemon_model.pth`
- **Matériel** : Compatible avec webcams standard, optimisé pour Logitech C505e
- **Déploiement** : Fonctionne sur NVIDIA Jetson Nano avec Docker
- **Utilisation** : Pour la reconnaissance en temps réel et la gestion d'inventaire

---

## Utilisation

### Ordre d'exécution recommandé

1. **Préparation du dataset** :
   ```bash
   python telechargement_pokemon.py  # Télécharge les ~18 831 cartes depuis l'API
   python data_aug_v2.py             # Augmente le dataset → ~370 000 images
   ```

2. **Classification d'images statiques** :
   ```bash
   python classification_v2.py       # Analyse les images dans le dossier courant
   ```

3. **Reconnaissance en temps réel** :
   ```bash
   python test_camera_v2.py          # Lance la détection avec la caméra
   ```

---

## Prérequis

- Python 3.7+
- PyTorch 1.8+ (compatible avec la version utilisée sur Jetson Nano)
- torchvision 0.9+
- PIL/Pillow (traitement d'images)
- OpenCV (pour `test_camera_v2.py`)
- API pokemontcg.io accessible (pour `telechargement_pokemon.py`)

---

## Notes importantes

- Les scripts de classification nécessitent le modèle entraîné : `model/best_pokemon_model.pth` (voir [model/README.md](../model/README.md) pour le téléchargement)
- Pour l'utilisation sur Jetson Nano, assurez-vous d'avoir Docker installé et configuré
- Les scripts utilisent le GPU si disponible, sinon fonctionnent en mode CPU
- Consultez les commentaires dans chaque fichier pour plus de détails sur les paramètres et dépendances spécifiques

---

## Contact

Pour toute question concernant les scripts ou leur utilisation :

- **Issues GitHub** : [Ouvrir une issue](https://github.com/AlexisXueref/PokéBrAIn/issues)
- **Documentation complète** : Voir `docs/Rapport-PokéBrAIn.pdf`
