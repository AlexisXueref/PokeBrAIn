# PokéBrAIn

**PokéBrAIn** : L'intelligence embarquée au service des cartes rares.

Un système embarqué de reconnaissance automatique de cartes Pokémon en temps réel, utilisant l'intelligence artificielle sur NVIDIA Jetson Nano.

---

## Objectif du Projet

Ce projet en collaboration avec Tiago Guéhéneux vise à construire un système embarqué, autonome et transportable, capable d'identifier automatiquement des cartes Pokémon à partir d'images capturées en temps réel par une webcam. Le système fournit pour chaque carte des informations clés comme le nom, la rareté et la valeur estimée, tout en respectant les contraintes de performance et de robustesse face aux conditions réelles d'utilisation (brocantes, collections privées, etc.).

---

## Technologies Utilisées

### Matériel

- **NVIDIA Jetson Nano** : Plateforme embarquée pour l'exécution du modèle IA en temps réel
- **Webcam Logitech C505e** : Capture des images des cartes
- **Ordinateur Asus TUF A17** : Entraînement du modèle (carte graphique dédiée)

### Logiciel

- **Modèle IA** : ResNet-18 pré-entraîné sur ImageNet, puis fine-tuné sur notre dataset
- **Framework** : PyTorch + TorchScript (optimisation pour l'embarqué)
- **Vision par ordinateur** : OpenCV
- **Conteneurisation** : Docker
- **Dataset** : API pokemontcg.io (~18 831 cartes) + data augmentation (~370 000 images)

---

## Performances

- **Accuracy de validation** : 97,96%
- **Nombre de classes** : ~18 831 cartes Pokémon distinctes
- **Dataset entraînement** : ~370 000 images (après data augmentation)
- **Optimisation** : Conversion TorchScript pour inférence rapide sur Jetson Nano

---

## Structure du Projet

```
PokéBrAIn/
├── docs/                    # Documentation (rapport, présentation)
├── examples/                # Exemples de résultats (matrices, courbes, images de test)
├── model/                   # Instructions pour le modèle (README, config)
├── scripts/                 # Scripts Python (data augmentation, entraînement, classification, caméra)
├── .gitignore               # Fichiers ignorés par Git
└── README.md                # Ce fichier
```

### Rôle des Dossiers

#### **`docs/`** - Documentation Complète

- Rapport technique détaillé du projet
- Présentation orale (slides)
- Méthodologie, résultats et analyses

#### **`examples/`** - Exemples et Résultats

- Images de test de cartes Pokémon
- Cartes data-augmentées (exemples)
- Matrice de confusion
- Courbes de perte (loss evolution)

#### **`model/`** - Modèle IA

- README avec instructions pour obtenir/utiliser le modèle
- Configuration et architecture
- **Note** : Les fichiers `.pth` (poids du modèle) ne sont PAS versionnés sur GitHub (trop volumineux)

#### **`scripts/`** - Scripts Python

- **data_aug_v2.py** : Data augmentation des cartes (lumière, rotation, couleur...)
- **telechargement_pokemon.py** : Téléchargement du dataset via l'API
- **classification_v2.py** : Classification des cartes avec le modèle entraîné
- **test_camera_v2.py** : Reconnaissance en temps réel via webcam (script principal)

---

## Démarrage Rapide

### 1. Cloner le Dépôt

```bash
git clone https://github.com/AlexisXueref/PokéBrAIn.git
cd PokéBrAIn
```

### 2. Installer les Dépendances

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 3. Télécharger le Modèle

**Important** : Les poids du modèle (.pth) ne sont pas inclus dans le dépôt Git.

Consultez `model/README.md` pour les instructions détaillées sur comment obtenir ou entraîner le modèle.

### 4. Tester le Système

```bash
# Classification d'une image
python scripts/classification_v2.py --image examples/test_card.jpg

# Reconnaissance en temps réel avec webcam
python scripts/test_camera_v2.py
```

---

## Documentation

Pour plus de détails, consultez :

- **[Rapport technique](docs/Rapport-PokéBrAIn.pdf)** : Méthodologie complète, résultats, analyses
- **[Présentation](docs/Présentation%20PokéBrAIn%20XUEREF%20Alexis%20GUEH...)** : Vue d'ensemble du projet
- **[README modèle](model/README.md)** : Instructions pour le modèle IA

---

## Fichiers Ignorés (.gitignore)

Pour maintenir un dépôt léger, les éléments suivants sont automatiquement ignorés :

- **Dataset** : `pokemon_dataset_advanced/`, `dataset/`, `*.zip`
- **Environnements virtuels** : `venv/`, `env/`, `__pycache__/`
- **Modèles entraînés** : `*.pth`, `*.pt`, `model/weights/`
- **Outputs** : `outputs/`, `logs/`, `*.log`

---

## Workflow du Projet

1. **Téléchargement du dataset** via l'API pokemontcg.io (~18 831 cartes)
2. **Data augmentation** pour générer ~370 000 images variées
3. **Entraînement** du modèle ResNet-18 (fine-tuning)
4. **Conversion** en TorchScript pour optimisation
5. **Déploiement** sur Jetson Nano avec Docker
6. **Test en temps réel** avec webcam

---

## Problèmes Connus et Solutions

### Limitations GPU sur Google Colab
- **Solution** : Entraînement sur machine locale (Asus TUF A17)

### Incompatibilités Docker/CUDA sur Jetson
- **Solution** : Configuration spécifique des images Docker et versions CUDA

### Problèmes de versions Python/OpenCV
- **Solution** : Synchronisation des versions entre Jetson et environnement de développement

### Qualité caméra insuffisante
- **Solution** : Impression de cartes en taille agrandie pour les tests
- **Amélioration future** : Ajout d'images floues et reflets dans le dataset

---

## Prochaines Étapes

### Application Mobile
- Développer une app mobile type "Google Lens" pour les cartes Pokémon
- Reconnaissance instantanée avec historique de cartes scannées
- Intégration avec marchés de vente de cartes

### Extensions Possibles
- Autres jeux de cartes : Magic, Yu-Gi-Oh, cartes de sport (Panini)
- Amélioration de la robustesse face aux conditions d'éclairage variables
- Optimisation supplémentaire pour d'autres plateformes embarquées

---

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir des issues ou des pull requests.

---

## Contact

- **GitHub** : [AlexisXueref/PokéBrAIn](https://github.com/AlexisXueref/PokéBrAIn)
- **Issues** : [Ouvrir une issue](https://github.com/AlexisXueref/PokéBrAIn/issues)

---

## Remerciements

- **API Pokémon TCG** : [pokemontcg.io](https://pokemontcg.io) - Fourniture du dataset
- **PyTorch** : Framework deep learning
- **NVIDIA** : Jetson Nano et support CUDA
- **Communauté Pokémon TCG** : Inspiration et passion

---

**PokéBrAIn** - Détection intelligente de cartes Pokémon 
