import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
from collections import deque, Counter
import json
import re
from typing import Optional, Dict, List

class PokemonCardRecognizer:
    def __init__(self, model_path="best_pokemon_model.pth", metadata_path="metadata.json"):
        # === Chargement du nouveau modèle ===
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Utilisation de : {self.device}")
        
        # Charger le modèle PyTorch (pas TorchScript)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['class_names']
        
        # Recréer l'architecture (doit correspondre à votre modèle entraîné)
        import torchvision.models as models
        import torch.nn as nn
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
        
        # === Prétraitement identique à l'entraînement ===
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # === Chargement des métadonnées pour les prix ===
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print("✅ Métadonnées chargées")
        except:
            print("⚠️ Métadonnées non trouvées, prix non disponibles")
            self.metadata = []
        
        # === Système de lissage temporel ===
        self.prediction_history = deque(maxlen=10)  # Garde les 10 dernières prédictions
        self.confidence_threshold = 0.6  # Seuil de confiance minimum
        self.stable_prediction_count = 3  # Nombre de prédictions stables nécessaires
        
        # === Statistiques ===
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
    def extract_card_id(self, label: str) -> Optional[str]:
        """Extrait l'ID de la carte depuis le label"""
        match = re.search(r'[a-z0-9]+-\d+[a-zA-Z_]*$', label)
        return match.group(0) if match else None
    
    def get_card_value(self, card_label: str) -> Optional[float]:
        """Récupère la valeur de la carte depuis les métadonnées"""
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
    
    def detect_card_region(self, frame):
        """Détection automatique de la région de la carte (optionnel)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détection de contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Chercher des rectangles
        for contour in contours:
            if cv2.contourArea(contour) > 5000:  # Taille minimum
                # Approximation polygonale
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Rectangle trouvé
                    # Calculer la bounding box
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = h / w
                    
                    # Vérifier si c'est une carte (ratio ~1.4)
                    if 1.2 < aspect_ratio < 1.6:
                        return x, y, x + w, y + h, True
        
        # Si aucune carte détectée, utiliser la région centrale
        return None, None, None, None, False
    
    def predict_frame(self, frame):
        """Prédiction sur une frame avec lissage temporel"""
        # === Région d'intérêt ===
        frame_height, frame_width = frame.shape[:2]
        
        # Essayer la détection automatique
        x1, y1, x2, y2, auto_detected = self.detect_card_region(frame)
        
        if not auto_detected:
            # Fallback : région centrale
            box_width, box_height = min(400, frame_width//2), min(560, frame_height//2)
            x1 = frame_width // 2 - box_width // 2
            y1 = frame_height // 2 - box_height // 2
            x2 = x1 + box_width
            y2 = y1 + box_height
        
        # === Extraction et prétraitement ===
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, 0.0, (x1, y1, x2, y2), auto_detected
            
        image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # === Prédiction ===
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_label = self.classes[predicted_idx.item()]
            confidence_score = confidence.item()
        
        # === Lissage temporel ===
        if confidence_score > self.confidence_threshold:
            self.prediction_history.append(predicted_label)
        
        # Trouver la prédiction la plus fréquente
        if len(self.prediction_history) >= self.stable_prediction_count:
            most_common = Counter(self.prediction_history).most_common(1)[0]
            stable_prediction = most_common[0]
            stability_count = most_common[1]
            
            if stability_count >= self.stable_prediction_count:
                predicted_label = stable_prediction
            
        return predicted_label, confidence_score, (x1, y1, x2, y2), auto_detected
    
    def calculate_fps(self):
        """Calcul du FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = current_time
        return np.mean(self.fps_history) if self.fps_history else 0
    
    def run_camera_test(self):
        """Boucle principale de test caméra"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Caméra non détectée.")
            return
        
        # Configuration caméra
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("📷 Test en cours...")
        print("Contrôles:")
        print("  [q] - Quitter")
        print("  [s] - Capturer et sauvegarder")
        print("  [r] - Reset historique")
        print("  [h] - Afficher/masquer l'aide")
        
        show_help = True
        last_prediction = ""
        last_confidence = 0.0
        last_value = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de capture")
                break
            
            self.frame_count += 1
            
            # === Prédiction ===
            prediction, confidence, bbox, auto_detected = self.predict_frame(frame)
            x1, y1, x2, y2 = bbox
            
            # === Calcul FPS ===
            fps = self.calculate_fps()
            
            # === Récupération de la valeur ===
            if prediction != last_prediction:
                last_value = self.get_card_value(prediction)
                last_prediction = prediction
                last_confidence = confidence
            
            # === Affichage ===
            # Rectangle de détection
            color = (0, 255, 0) if auto_detected else (255, 255, 0)  # Vert si auto, jaune si manuel
            thickness = 3 if confidence > self.confidence_threshold else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Informations principales
            label_text = f"{prediction}"
            conf_text = f"Confiance: {confidence*100:.1f}%"
            value_text = f"Valeur: {last_value:.2f}€" if last_value else "Valeur: N/A"
            
            # Couleur du texte selon confiance
            text_color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 255, 255)
            
            # Affichage des textes
            y_offset = y1 - 15
            cv2.putText(frame, label_text, (x1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            y_offset -= 25
            cv2.putText(frame, conf_text, (x1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            if last_value:
                y_offset -= 25
                cv2.putText(frame, value_text, (x1, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Informations système
            fps_text = f"FPS: {fps:.1f}"
            history_text = f"Historique: {len(self.prediction_history)}/10"
            detection_text = "Auto" if auto_detected else "Manuel"
            
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, history_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Détection: {detection_text}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Aide
            if show_help:
                help_texts = [
                    "Controles: [q]Quitter [s]Capturer [r]Reset [h]Aide",
                    f"Seuil confiance: {self.confidence_threshold*100:.0f}%",
                    f"Predictions stables: {self.stable_prediction_count}"
                ]
                for i, text in enumerate(help_texts):
                    cv2.putText(frame, text, (10, frame.shape[0] - 70 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # === Affichage ===
            cv2.imshow("Détection Pokémon - Modèle Avancé", frame)
            
            # === Gestion des touches ===
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Sauvegarder l'image
                timestamp = int(time.time())
                filename = f"capture_{prediction}_{timestamp}.jpg"
                roi = frame[y1:y2, x1:x2]
                cv2.imwrite(filename, roi)
                print(f"📸 Image sauvegardée: {filename}")
            elif key == ord('r'):
                # Reset de l'historique
                self.prediction_history.clear()
                print("🔄 Historique réinitialisé")
            elif key == ord('h'):
                # Toggle aide
                show_help = not show_help
        
        cap.release()
        cv2.destroyAllWindows()
        
        # === Statistiques finales ===
        print(f"\n📊 Statistiques de session:")
        print(f"   Frames traitées: {self.frame_count}")
        print(f"   FPS moyen: {np.mean(self.fps_history):.1f}")
        print(f"   Prédictions en historique: {len(self.prediction_history)}")

def main():
    # Vérifier les fichiers nécessaires
    model_files = ["best_pokemon_model.pth", "final_pokemon_model.pth"]
    model_path = None
    
    for file in model_files:
        if os.path.exists(file):
            model_path = file
            break
    
    if not model_path:
        print("❌ Aucun modèle trouvé. Vérifiez que best_pokemon_model.pth existe.")
        return
    
    print(f"🔧 Utilisation du modèle: {model_path}")
    
    try:
        recognizer = PokemonCardRecognizer(model_path=model_path)
        recognizer.run_camera_test()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("💡 Vérifiez que le modèle correspond à l'architecture attendue")

if __name__ == "__main__":
    import os
    main()