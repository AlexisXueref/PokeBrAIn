import requests
import os
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# --- Configuration ---
API_KEY = "d741bfe9-5443-477a-8190-27951ab74604"
HEADERS = {"X-Api-Key": API_KEY}
SAVE_DIR = "pokemon_dataset/train"
PROGRESS_FILE = "download_progress.json"
ALL_METADATA_FILE = "all_cards_metadata.json"

# --- Préparation session requests avec retry ---
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def clean_name(name):
    return name.replace(" ", "_").replace("/", "_").replace(":", "_")

def fetch_all_cards():
    print("Téléchargement de toutes les cartes depuis l'API...")
    all_cards = []
    page = 1
    while True:
        url = f"https://api.pokemontcg.io/v2/cards?page={page}&pageSize=250"
        try:
            response = session.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
            if not data:
                break
            all_cards.extend(data)
            print(f"Page {page} récupérée - Total cartes : {len(all_cards)}")
            page += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"Erreur à la page {page}: {e} - Nouvelle tentative dans 5s...")
            time.sleep(5)
    print(f"Total cartes récupérées : {len(all_cards)}")
    return all_cards

def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def download_dataset(cards, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    done_ids = set(load_progress())

    print("Téléchargement des images et métadonnées (reprise activée)...")
    for card in tqdm(cards):
        card_id = card["id"]
        card_name = clean_name(card["name"]) + "_" + card_id
        card_dir = os.path.join(save_dir, card_name)
        img_path = os.path.join(card_dir, "img.jpg")
        meta_path = os.path.join(card_dir, "metadata.json")

        # Vérifie que les deux fichiers existent
        if card_id in done_ids and os.path.exists(img_path) and os.path.exists(meta_path):
            continue

        try:
            os.makedirs(card_dir, exist_ok=True)

            # Téléchargement de l'image
            img_url = card["images"]["large"]
            resp_img = session.get(img_url, timeout=10)
            resp_img.raise_for_status()
            img = Image.open(BytesIO(resp_img.content)).convert("RGB")
            img.save(img_path)

            # Sauvegarde métadonnées individuelles
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(card, f, indent=2, ensure_ascii=False)

            # Mise à jour de la progression
            done_ids.add(card_id)
            save_progress(list(done_ids))

            time.sleep(0.1)
        except Exception as e:
            print(f"Erreur carte {card_id} : {e}")
            time.sleep(1)

# --- Exécution principale ---
cards = fetch_all_cards()

# Sauvegarde de toutes les métadonnées globales
with open(ALL_METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(cards, f, indent=2, ensure_ascii=False)

# Téléchargement des images + métadonnées
download_dataset(cards)

import requests
import os
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# --- Configuration ---
API_KEY = "d741bfe9-5443-477a-8190-27951ab74604"
HEADERS = {"X-Api-Key": API_KEY}
SAVE_DIR = "pokemon_dataset/train"
PROGRESS_FILE = "download_progress.json"
ALL_METADATA_FILE = "all_cards_metadata.json"

# --- Préparation session requests avec retry ---
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def clean_name(name):
    return name.replace(" ", "_").replace("/", "_").replace(":", "_")

def fetch_all_cards():
    print("Téléchargement de toutes les cartes depuis l'API...")
    all_cards = []
    page = 1
    while True:
        url = f"https://api.pokemontcg.io/v2/cards?page={page}&pageSize=250"
        try:
            response = session.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
            if not data:
                break
            all_cards.extend(data)
            print(f"Page {page} récupérée - Total cartes : {len(all_cards)}")
            page += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"Erreur à la page {page}: {e} - Nouvelle tentative dans 5s...")
            time.sleep(5)
    print(f"Total cartes récupérées : {len(all_cards)}")
    return all_cards

def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def download_dataset(cards, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    done_ids = set(load_progress())

    print("Téléchargement des images et métadonnées (reprise activée)...")
    for card in tqdm(cards):
        card_id = card["id"]
        card_name = clean_name(card["name"]) + "_" + card_id
        card_dir = os.path.join(save_dir, card_name)
        img_path = os.path.join(card_dir, "img.jpg")
        meta_path = os.path.join(card_dir, "metadata.json")

        # Vérifie que les deux fichiers existent
        if card_id in done_ids and os.path.exists(img_path) and os.path.exists(meta_path):
            continue

        try:
            os.makedirs(card_dir, exist_ok=True)

            # Téléchargement de l'image
            img_url = card["images"]["large"]
            resp_img = session.get(img_url, timeout=10)
            resp_img.raise_for_status()
            img = Image.open(BytesIO(resp_img.content)).convert("RGB")
            img.save(img_path)

            # Sauvegarde métadonnées individuelles
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(card, f, indent=2, ensure_ascii=False)

            # Mise à jour de la progression
            done_ids.add(card_id)
            save_progress(list(done_ids))

            time.sleep(0.1)
        except Exception as e:
            print(f"Erreur carte {card_id} : {e}")
            time.sleep(1)

# --- Exécution principale ---
cards = fetch_all_cards()

# Sauvegarde de toutes les métadonnées globales
with open(ALL_METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(cards, f, indent=2, ensure_ascii=False)

# Téléchargement des images + métadonnées
download_dataset(cards)