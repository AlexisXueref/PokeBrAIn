# PokéBrAIn

**PokéBrAIn** : L'intelligence embarquée au service des cartes rares.

Un système d'intelligence artificielle basé sur YOLO pour la détection et la classification des cartes Pokémon, avec une interface embarquée sur Raspberry Pi.

---

## 📁 Structure du Projet

Le projet suit une architecture claire et organisée pour faciliter la navigation et la maintenance :

```
PokeBrAIn/
├── examples/          # Exemples d'utilisation et démonstrations
├── scripts/           # Scripts d'automatisation et utilitaires
├── docs/              # Documentation complète du projet
├── model/             # Architecture et définitions des modèles
├── .gitignore         # Fichiers et dossiers ignorés par Git
└── README.md          # Ce fichier
```

### 🗂️ Rôle des Dossiers

#### **`examples/`** - Exemples et Démonstrations
- **Contenu** : Images de test, exemples de prédictions, notebooks de démonstration
- **Usage** : 
  - Placez vos images d'exemple dans `examples/images/`
  - Les résultats de détection seront sauvegardés dans `examples/outputs/`
  - Consultez les notebooks pour comprendre l'utilisation du modèle
- **Cas d'usage** : Tester rapidement le modèle, valider les performances, créer des démonstrations

#### **`scripts/`** - Scripts d'Automatisation
- **Contenu** : Scripts Python pour l'entraînement, l'évaluation, le déploiement
- **Usage** :
  - `train.py` : Entraînement du modèle YOLO
  - `evaluate.py` : Évaluation des performances
  - `deploy_raspberry.py` : Scripts de déploiement sur Raspberry Pi
  - Scripts utilitaires pour le traitement des données
- **Cas d'usage** : Automatiser les workflows, faciliter les expérimentations

#### **`docs/`** - Documentation Complète
- **Contenu** : Présentation, rapport technique, guides d'installation
- **Usage** :
  - Documentation architecturale du projet
  - Guides d'utilisation détaillés
  - Méthodologie et résultats expérimentaux
  - Présentation du projet (slides, PDF)
- **Cas d'usage** : Comprendre le projet en profondeur, référence technique

#### **`model/`** - Définitions et Architecture
- **Contenu** : Configuration YOLO, architecture du réseau, fichiers de définition
- **Usage** :
  - Fichiers de configuration `.yaml` pour YOLO
  - Architecture des couches du réseau
  - Fichiers de classe et de métadonnées
- **⚠️ Important** : Les fichiers de poids (`.pth`, `.pt`) ne sont **pas versionnés** (voir .gitignore)

---

## 📂 Organisation des Fichiers Spécifiques

### Images et Exemples
- **Emplacement** : `examples/images/`
- **Format accepté** : JPG, PNG
- **Utilisation** : Images de test pour la détection

### Outputs et Résultats
- **Emplacement** : `examples/outputs/`
- **Contenu** : Images annotées, logs de prédiction, métriques
- **⚠️ Statut Git** : Non versionné (généré automatiquement)

### Modèles Entraînés
- **Emplacement** : `model/weights/` (local uniquement)
- **Format** : `.pth`, `.pt`, `.onnx`
- **⚠️ Statut Git** : **Non versionné** - Les poids sont trop volumineux pour Git
- **Alternative** : Utiliser Git LFS ou un service de stockage externe (Google Drive, Hugging Face Hub)

### Documentation
- **Présentation** : `docs/presentation.pdf`
- **Rapport technique** : `docs/rapport_technique.pdf`
- **Guides** : `docs/guides/`

---

## 🚫 Fichiers et Dossiers Ignorés (.gitignore)

Pour maintenir un dépôt propre et léger, les éléments suivants sont **automatiquement ignorés par Git** :

### 🗃️ Datasets
```
dataset/
data/
*.zip
```
**Pourquoi ?** Les datasets peuvent être très volumineux (plusieurs Go). Partagez-les via des liens externes.

### 🐍 Environnements Virtuels Python
```
venv/
env/
.venv/
__pycache__/
*.pyc
*.pyo
```
**Pourquoi ?** Les environnements virtuels sont spécifiques à chaque machine. Utilisez `requirements.txt` pour partager les dépendances.

### 🧠 Fichiers de Poids de Modèles
```
*.pth
*.pt
*.onnx
model/weights/
```
**Pourquoi ?** Les fichiers de poids peuvent atteindre plusieurs centaines de Mo. Utilisez des services spécialisés pour les partager.

### 📊 Fichiers Temporaires et Outputs
```
examples/outputs/
logs/
*.log
.DS_Store
```
**Pourquoi ?** Ces fichiers sont générés automatiquement et varient selon les exécutions.

---

## 📚 Documentation

### Présentation du Projet
📄 **[Présentation](docs/presentation.pdf)** - Vue d'ensemble, objectifs, architecture

### Rapport Technique
📄 **[Rapport Technique](docs/rapport_technique.pdf)** - Méthodologie détaillée, résultats, analyses

### Guides d'Utilisation
- **Installation** : `docs/guides/installation.md`
- **Entraînement** : `docs/guides/training.md`
- **Déploiement Raspberry Pi** : `docs/guides/raspberry_deployment.md`

---

## 🚀 Démarrage Rapide

### 1. Cloner le Dépôt
```bash
git clone https://github.com/AlexisXueref/PokeBrAIn.git
cd PokeBrAIn
```

### 2. Installer les Dépendances
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Télécharger les Poids du Modèle
**⚠️ Important** : Les poids ne sont pas inclus dans le dépôt Git.

- **Option 1** : Télécharger depuis [lien vers stockage externe]
- **Option 2** : Entraîner votre propre modèle avec `scripts/train.py`

Placez les fichiers `.pth` dans `model/weights/`

### 4. Tester le Modèle
```bash
python scripts/predict.py --image examples/images/test_card.jpg
```

Les résultats seront sauvegardés dans `examples/outputs/`

---

## 🎯 Bonnes Pratiques

### ✅ À Faire
- Placer vos images de test dans `examples/images/`
- Documenter vos scripts dans `scripts/`
- Mettre à jour la documentation dans `docs/` après modifications majeures
- Utiliser des branches Git pour les nouvelles fonctionnalités
- Tester vos modifications avec les exemples fournis

### ❌ À Éviter
- **Ne pas** commiter de fichiers `.pth` ou `.pt` (trop volumineux)
- **Ne pas** versionner les datasets (utiliser des liens)
- **Ne pas** commiter les environnements virtuels (`venv/`, `env/`)
- **Ne pas** inclure les outputs générés automatiquement
- **Ne pas** modifier `.gitignore` sans consultation de l'équipe

---

## 💡 Workflow Recommandé

### Pour les Développeurs
1. Créer une branche pour votre fonctionnalité
2. Développer et tester localement
3. Mettre à jour la documentation si nécessaire
4. Soumettre une Pull Request avec description claire

### Pour les Contributeurs Documentation
1. Ajouter/modifier les fichiers dans `docs/`
2. Vérifier les liens et références
3. Maintenir la cohérence avec le code

### Pour l'Entraînement de Modèles
1. Préparer votre dataset (hors Git)
2. Configurer les paramètres dans `model/config.yaml`
3. Lancer l'entraînement avec `scripts/train.py`
4. Sauvegarder les poids localement (`model/weights/`)
5. Partager via lien externe (Drive, HF Hub)

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Consultez `docs/CONTRIBUTING.md` pour les guidelines.

### Structure de Commit
```
type(scope): description courte

Détails additionnels si nécessaire
```

**Types** : `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

---

## 📞 Contact et Support

Pour toute question ou suggestion :
- **Issues** : [GitHub Issues](https://github.com/AlexisXueref/PokeBrAIn/issues)
- **Discussions** : [GitHub Discussions](https://github.com/AlexisXueref/PokeBrAIn/discussions)

---

## 📜 Licence

*[À définir selon votre choix de licence]*

---

## 🏆 Remerciements

- **YOLO** : Framework de détection d'objets
- **Ultralytics** : Implémentation YOLOv8
- **Raspberry Pi Foundation** : Plateforme embarquée
- **Communauté Pokémon TCG** : Passion et inspiration

---

**PokéBrAIn** - Détection intelligente de cartes Pokémon 🎴🧠✨
