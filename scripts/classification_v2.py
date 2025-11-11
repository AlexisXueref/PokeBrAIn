import json
from typing import List, Dict, Optional, Tuple
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import re
import csv
import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class PokemonBatchClassifier:
    def __init__(self, model_path="best_pokemon_model.pth", metadata_path="metadata.json"):
        """Initialiseur avec le nouveau modèle entraîné"""
        
        # === Chargement du modèle ===
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Utilisation de : {self.device}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.classes = checkpoint['class_names']
            
            # Recréer l'architecture
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.fc.in_features, len(self.classes))
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            self.model = model
            
            print(f"✅ Modèle chargé avec {len(self.classes)} classes")
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            raise
        
        # === Prétraitement ===
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # === Test Time Augmentation ===
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation((5, 5)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation((-5, -5)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]
        
        # === Métadonnées ===
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print("✅ Métadonnées chargées")
        except:
            print("⚠️ Métadonnées non trouvées")
            self.metadata = []
        
        # === Statistiques ===
        self.classification_times = []
        self.confidence_scores = []
        
    def extract_card_id(self, label: str) -> Optional[str]:
        """Extrait l'ID de la carte depuis le label"""
        match = re.search(r'[a-z0-9]+-\d+[a-zA-Z_]*$', label)
        return match.group(0) if match else None
    
    def get_card_value(self, card_label: str) -> Optional[float]:
        """Récupère la valeur de la carte"""
        card_id = self.extract_card_id(card_label)
        if not card_id or not self.metadata:
            return None
            
        for card in self.metadata:
            if card.get("id", "").lower() == card_id.lower():
                prices = card.get("tcgplayer", {}).get("prices", {})
                holo_price = prices.get("holofoil", {}).get("mid")
                if holo_price is not None:
                    return holo_price
                return card.get("cardmarket", {}).get("prices", {}).get("averageSellPrice")
        return None
    
    def get_card_info(self, card_label: str) -> Dict:
        """Récupère toutes les infos d'une carte"""
        card_id = self.extract_card_id(card_label)
        if not card_id or not self.metadata:
            return {"name": card_label, "set": "Unknown", "rarity": "Unknown", "value": None}
            
        for card in self.metadata:
            if card.get("id", "").lower() == card_id.lower():
                return {
                    "name": card.get("name", card_label),
                    "set": card.get("set", {}).get("name", "Unknown"),
                    "rarity": card.get("rarity", "Unknown"),
                    "value": self.get_card_value(card_label),
                    "type": card.get("types", ["Unknown"])[0] if card.get("types") else "Unknown"
                }
        return {"name": card_label, "set": "Unknown", "rarity": "Unknown", "value": None, "type": "Unknown"}
    
    def classify_single_image(self, image_path: str, use_tta: bool = False) -> Tuple[str, float]:
        """Classification d'une seule image avec option TTA"""
        start_time = time.time()
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if use_tta:
                # Test Time Augmentation
                predictions = []
                confidences = []
                
                for transform in self.tta_transforms:
                    input_tensor = transform(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                        
                        predictions.append(predicted_idx.item())
                        confidences.append(confidence.item())
                
                # Moyenne des prédictions
                pred_counts = defaultdict(int)
                conf_sums = defaultdict(float)
                
                for pred, conf in zip(predictions, confidences):
                    pred_counts[pred] += 1
                    conf_sums[pred] += conf
                
                # Prédiction finale : la plus fréquente ou la plus confiante
                best_pred = max(pred_counts.keys(), key=lambda x: (pred_counts[x], conf_sums[x]))
                avg_confidence = conf_sums[best_pred] / pred_counts[best_pred]
                
                predicted_class = self.classes[best_pred]
                final_confidence = avg_confidence
                
            else:
                # Classification simple
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    predicted_class = self.classes[predicted_idx.item()]
                    final_confidence = confidence.item()
            
            classification_time = time.time() - start_time
            self.classification_times.append(classification_time)
            self.confidence_scores.append(final_confidence)
            
            return predicted_class, final_confidence
            
        except Exception as e:
            print(f"❌ Erreur lors de la classification de {image_path}: {e}")
            return "ERREUR", 0.0
    
    def classify_batch_images(self, directory: str = ".", use_tta: bool = False, 
                            confidence_threshold: float = 0.5) -> Dict:
        """Classification par lot avec filtrage par confiance"""
        
        # Recherche des images
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(directory) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print("❌ Aucune image trouvée dans le répertoire")
            return {}
        
        print(f"📸 {len(image_files)} images trouvées")
        print(f"🎯 Seuil de confiance: {confidence_threshold*100:.1f}%")
        print(f"🔄 TTA activé: {'Oui' if use_tta else 'Non'}")
        
        inventory = {}
        low_confidence_items = []
        
        # Barre de progression
        for filename in tqdm(image_files, desc="Classification en cours"):
            image_path = os.path.join(directory, filename)
            card_name, confidence = self.classify_single_image(image_path, use_tta)
            
            if confidence >= confidence_threshold:
                # Confiance suffisante
                card_info = self.get_card_info(card_name)
                
                if card_name not in inventory:
                    inventory[card_name] = {
                        "count": 1,
                        "info": card_info,
                        "confidence_scores": [confidence],
                        "images": [filename]
                    }
                else:
                    inventory[card_name]["count"] += 1
                    inventory[card_name]["confidence_scores"].append(confidence)
                    inventory[card_name]["images"].append(filename)
                
                value_str = f"{card_info['value']:.2f}€" if card_info['value'] else "N/A"
                print(f"✅ {filename}: {card_name} ({confidence*100:.1f}%) - {value_str}")
                
            else:
                # Confiance trop faible
                low_confidence_items.append({
                    "filename": filename,
                    "prediction": card_name,
                    "confidence": confidence
                })
                print(f"⚠️ {filename}: {card_name} ({confidence*100:.1f}%) - CONFIANCE FAIBLE")
        
        # Rapport final
        print(f"\n📊 Résultats:")
        print(f"   Cartes identifiées avec confiance: {len(inventory)}")
        print(f"   Images à confiance faible: {len(low_confidence_items)}")
        print(f"   Temps moyen par image: {np.mean(self.classification_times):.3f}s")
        print(f"   Confiance moyenne: {np.mean(self.confidence_scores)*100:.1f}%")
        
        return {
            "inventory": inventory,
            "low_confidence": low_confidence_items,
            "stats": {
                "total_images": len(image_files),
                "high_confidence": len(inventory),
                "low_confidence": len(low_confidence_items),
                "avg_time": np.mean(self.classification_times),
                "avg_confidence": np.mean(self.confidence_scores)
            }
        }
    
    def save_detailed_inventory(self, results: Dict, output_dir: str = "outputs"):
        """Sauvegarde détaillée avec plusieurs formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        # === CSV Principal ===
        csv_path = os.path.join(output_dir, "inventaire_detaille.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Nom de carte', 'Quantité', 'Set', 'Rareté', 'Type',
                'Valeur unitaire (€)', 'Valeur totale (€)', 
                'Confiance moyenne (%)', 'Images'
            ])
            
            total_value = 0
            for card, data in results["inventory"].items():
                info = data["info"]
                count = data["count"]
                value = info["value"]
                avg_conf = np.mean(data["confidence_scores"]) * 100
                images = "; ".join(data["images"])
                
                total_val = count * value if value else 0
                total_value += total_val
                
                writer.writerow([
                    info["name"], count, info["set"], info["rarity"], info["type"],
                    f"{value:.2f}" if value else "N/A",
                    f"{total_val:.2f}" if value else "N/A",
                    f"{avg_conf:.1f}",
                    images
                ])
            
            # Ligne totale
            writer.writerow([
                "TOTAL", "", "", "", "", "", f"{total_value:.2f}€", "", ""
            ])
        
        print(f"💾 Inventaire détaillé sauvegardé: {csv_path}")
        
        # === CSV Confiance Faible ===
        if results["low_confidence"]:
            low_conf_path = os.path.join(output_dir, "confiance_faible.csv")
            with open(low_conf_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Fichier', 'Prédiction', 'Confiance (%)'])
                
                for item in results["low_confidence"]:
                    writer.writerow([
                        item["filename"], 
                        item["prediction"], 
                        f"{item['confidence']*100:.1f}"
                    ])
            
            print(f"⚠️ Items à faible confiance: {low_conf_path}")
        
        # === Rapport JSON ===
        json_path = os.path.join(output_dir, "rapport_complet.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Rapport JSON complet: {json_path}")
        
        # === Résumé console ===
        inventory = results["inventory"]
        if inventory:
            print(f"\n💰 **RÉSUMÉ FINANCIER**")
            total_value = sum(
                data["count"] * (data["info"]["value"] or 0) 
                for data in inventory.values()
            )
            print(f"   Valeur totale estimée: {total_value:.2f}€")
            
            # Top 5 des cartes les plus chères
            valuable_cards = [
                (name, data["info"]["value"] or 0, data["count"]) 
                for name, data in inventory.items()
            ]
            valuable_cards.sort(key=lambda x: x[1] * x[2], reverse=True)
            
            print(f"\n🏆 **TOP CARTES LES PLUS PRÉCIEUSES**")
            for i, (name, value, count) in enumerate(valuable_cards[:5], 1):
                if value > 0:  # Ne montrer que les cartes avec valeur
                    total_card_value = value * count
                    print(f"   {i}. {name}: {value:.2f}€ x{count} = {total_card_value:.2f}€")


# === SCRIPT PRINCIPAL ===
if __name__ == "__main__":
    try:
        print("🚀 Démarrage du classificateur Pokemon...")
        classifier = PokemonBatchClassifier()
        
        # Lancement de la classification
        results = classifier.classify_batch_images(
            directory=".",  # Dossier courant
            use_tta=False,  # Mettre True pour plus de précision
            confidence_threshold=0.5  # 50% de confiance minimum
        )
        
        # Sauvegarde des résultats
        classifier.save_detailed_inventory(results)
        
        print("\n✅ Classification terminée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()