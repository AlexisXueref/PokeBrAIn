# Scripts

Ce dossier regroupe tous les scripts Python nécessaires au fonctionnement du projet PokeBrAIn.

## Contenu

Ce répertoire contient les scripts suivants :

### Scripts de préparation des données
- **data_augmentation.py** : Script pour augmenter le dataset en appliquant des transformations aux images (rotation, flip, zoom, etc.) afin d'améliorer la robustesse du modèle.
- **download.py** : Script pour télécharger automatiquement les images de Pokémon depuis des sources en ligne et organiser le dataset.

### Scripts de traitement et classification
- **classification.py** : Script principal pour entraîner et évaluer les modèles de classification de Pokémon à partir des images.

### Scripts d'utilisation en temps réel
- **camera_recognition.py** : Script pour effectuer la reconnaissance de Pokémon en temps réel via la caméra de l'appareil, en utilisant le modèle entraîné.

## Utilisation

Chaque script peut être exécuté indépendamment selon les besoins du projet. Consultez les commentaires dans chaque fichier pour plus de détails sur les paramètres et dépendances nécessaires.
