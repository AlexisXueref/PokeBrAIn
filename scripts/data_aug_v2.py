import os
import random
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from PIL import Image, ImageFilter, ImageEnhance
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
from torchvision.datasets.folder import default_loader
import cv2

# 🔁 Dataset personnalisé
class RGBImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = default_loader(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

# 🎨 Transformations personnalisées pour conditions réelles
class AddGaussianNoise:
    """Ajoute du bruit gaussien pour simuler les conditions de faible éclairage"""
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class AddMotionBlur:
    """Simule le flou de mouvement"""
    def __init__(self, max_kernel_size=15):
        self.max_kernel_size = max_kernel_size
    
    def __call__(self, img):
        if random.random() < 0.3:  # 30% de chance d'appliquer le flou
            # Conversion PIL -> numpy
            img_array = np.array(img)
            
            # Création d'un kernel de flou directionnel aléatoire
            kernel_size = random.randint(3, self.max_kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            angle = random.randint(0, 180)
            kernel = np.zeros((kernel_size, kernel_size))
            
            # Ligne diagonale pour simuler le mouvement
            center = kernel_size // 2
            for i in range(kernel_size):
                kernel[center, i] = 1
            
            # Rotation du kernel
            M = cv2.getRotationMatrix2D((center, center), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            kernel = kernel / kernel.sum()
            
            # Application du flou
            blurred = cv2.filter2D(img_array, -1, kernel)
            img = Image.fromarray(blurred.astype(np.uint8))
        
        return img

class AddGaussianBlur:
    """Ajoute un flou gaussien pour simuler des problèmes de mise au point"""
    def __init__(self, max_radius=3):
        self.max_radius = max_radius
    
    def __call__(self, img):
        if random.random() < 0.4:  # 40% de chance
            radius = random.uniform(0.5, self.max_radius)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

class AddReflection:
    """Simule des reflets en ajoutant des zones plus claires"""
    def __call__(self, img):
        if random.random() < 0.25:  # 25% de chance
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            
            # Création d'un masque de reflet aléatoire
            center_x = random.randint(w//4, 3*w//4)
            center_y = random.randint(h//4, 3*h//4)
            radius = random.randint(20, min(w, h)//3)
            
            # Masque circulaire ou elliptique
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
            
            # Application du reflet (augmentation de la luminosité)
            reflection_strength = random.uniform(0.3, 0.7)
            img_array = img_array.astype(np.float32)
            img_array[mask] = np.clip(img_array[mask] * (1 + reflection_strength), 0, 255)
            
            img = Image.fromarray(img_array.astype(np.uint8))
        return img

class AddBackground:
    """Ajoute un arrière-plan autour de la carte"""
    def __init__(self, bg_colors=None):
        if bg_colors is None:
            # Couleurs typiques d'arrière-plans : blanc, beige, bois, tissu, etc.
            self.bg_colors = [
                (255, 255, 255),  # Blanc
                (240, 240, 235),  # Blanc cassé
                (245, 245, 220),  # Beige
                (210, 180, 140),  # Bois clair
                (139, 69, 19),    # Bois foncé
                (128, 128, 128),  # Gris
                (50, 50, 50),     # Gris foncé
                (0, 100, 0),      # Vert (tapis)
                (0, 0, 139),      # Bleu foncé
            ]
        else:
            self.bg_colors = bg_colors
    
    def __call__(self, img):
        if random.random() < 0.6:  # 60% de chance
            # Taille de l'image originale
            w, h = img.size
            
            # Nouvelle taille avec de l'espace autour
            scale_factor = random.uniform(0.6, 0.9)
            new_w = int(w / scale_factor)
            new_h = int(h / scale_factor)
            
            # Couleur d'arrière-plan aléatoire
            bg_color = random.choice(self.bg_colors)
            
            # Création de la nouvelle image avec arrière-plan
            bg_img = Image.new('RGB', (new_w, new_h), bg_color)
            
            # Position aléatoire pour coller la carte
            x_offset = random.randint(0, new_w - w)
            y_offset = random.randint(0, new_h - h)
            bg_img.paste(img, (x_offset, y_offset))
            
            # Redimensionnement final
            img = bg_img.resize((w, h), Image.Resampling.LANCZOS)
        
        return img

class AddShadow:
    """Ajoute des ombres pour plus de réalisme"""
    def __call__(self, img):
        if random.random() < 0.3:  # 30% de chance
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            
            # Création d'un gradient d'ombre
            shadow_strength = random.uniform(0.1, 0.4)
            direction = random.choice(['top', 'bottom', 'left', 'right', 'corner'])
            
            shadow_mask = np.ones((h, w), dtype=np.float32)
            
            if direction == 'top':
                for i in range(h//3):
                    shadow_mask[i, :] *= (1 - shadow_strength * (1 - i/(h//3)))
            elif direction == 'bottom':
                for i in range(2*h//3, h):
                    shadow_mask[i, :] *= (1 - shadow_strength * ((i - 2*h//3)/(h//3)))
            elif direction == 'left':
                for j in range(w//3):
                    shadow_mask[:, j] *= (1 - shadow_strength * (1 - j/(w//3)))
            elif direction == 'right':
                for j in range(2*w//3, w):
                    shadow_mask[:, j] *= (1 - shadow_strength * ((j - 2*w//3)/(w//3)))
            elif direction == 'corner':
                center_x, center_y = w//4, h//4
                for i in range(h):
                    for j in range(w):
                        dist = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                        max_dist = np.sqrt(center_y**2 + center_x**2)
                        shadow_mask[i, j] *= (1 - shadow_strength * (1 - min(dist/max_dist, 1)))
            
            # Application de l'ombre
            for c in range(3):
                img_array[:, :, c] = (img_array[:, :, c] * shadow_mask).astype(np.uint8)
            
            img = Image.fromarray(img_array)
        return img

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # 🎯 Objectif
    print("🎯 Objectif : entraîner avec data augmentation avancée pour conditions réelles")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # --- Data Augmentation enrichie pour conditions réelles ---
    # Transformations PIL (avant ToTensor)
    pil_augmentations = transforms.Compose([
        transforms.Resize((256, 256)),  # Un peu plus grand pour avoir de la marge
        AddGaussianBlur(max_radius=2),
        AddMotionBlur(max_kernel_size=11),
        AddReflection(),
        AddBackground(),
        AddShadow(),
        transforms.RandomHorizontalFlip(p=0.3),  # Moins fréquent pour les cartes
        transforms.RandomRotation(degrees=15, fill=0),  # Rotation plus douce
        transforms.ColorJitter(
            brightness=0.4,  # Plus de variation pour simuler différents éclairages
            contrast=0.4,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=5,
            fill=0
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Perspective différente
        transforms.Resize((224, 224)),  # Taille finale
    ])
    
    # Transformations tenseur (après ToTensor)
    tensor_augmentations = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(mean=0., std=0.05),  # Bruit après conversion en tenseur
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Occlusion partielle
    ])

    AUGMENTED_PER_CARD = 25  # Plus d'augmentations

    def generate_augmented_images(input_dir, output_dir, existing_dir=None):
        """Génère les images augmentées avec les nouvelles transformations"""
        for card_folder in tqdm(os.listdir(input_dir), desc="Génération d'images augmentées"):
            card_path = os.path.join(input_dir, card_folder)
            img_path = os.path.join(card_path, "img.jpg")
            if not os.path.isfile(img_path):
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                save_path = os.path.join(output_dir, card_folder)
                
                # Vérifier si déjà généré avec le bon nombre d'images
                if os.path.exists(save_path) and len(os.listdir(save_path)) >= AUGMENTED_PER_CARD:
                    continue
                
                os.makedirs(save_path, exist_ok=True)
                
                # 🔄 COPIER les images existantes si elles existent
                existing_path = os.path.join(existing_dir, card_folder) if existing_dir else None
                images_copied = 0
                
                if existing_path and os.path.exists(existing_path):
                    print(f"📋 Copie des images existantes pour {card_folder}")
                    for existing_img in os.listdir(existing_path):
                        if existing_img.endswith(('.jpg', '.jpeg', '.png')):
                            src = os.path.join(existing_path, existing_img)
                            dst = os.path.join(save_path, existing_img)
                            if not os.path.exists(dst):  # Éviter d'écraser
                                try:
                                    import shutil
                                    shutil.copy2(src, dst)
                                    images_copied += 1
                                except Exception as copy_e:
                                    print(f"Erreur copie {existing_img}: {copy_e}")
                
                # Compter les images déjà présentes
                existing_count = len([f for f in os.listdir(save_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                
                # Si on n'a pas assez d'images, en générer de nouvelles
                if existing_count < AUGMENTED_PER_CARD:
                    print(f"🎨 Génération de {AUGMENTED_PER_CARD - existing_count} nouvelles images pour {card_folder}")
                    
                    # S'assurer qu'on a l'image originale
                    if not os.path.exists(os.path.join(save_path, "img_0.jpg")):
                        original_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
                        original_resized.save(os.path.join(save_path, "img_0.jpg"), quality=95)
                    
                    # Générer les images manquantes avec les NOUVELLES augmentations
                    for i in range(existing_count, AUGMENTED_PER_CARD + 1):
                        try:
                            augmented = pil_augmentations(image)  # Nouvelles augmentations avancées
                            augmented.save(os.path.join(save_path, f"img_advanced_{i}.jpg"), quality=90)
                        except Exception as aug_e:
                            print(f"Erreur augmentation {i} pour {card_folder}: {aug_e}")
                else:
                    print(f"✅ {card_folder} a déjà assez d'images ({existing_count})")
                        
            except Exception as e:
                print(f"Erreur pour {card_folder}: {e}")

    INPUT_DIR = "pokemon_dataset/train"
    EXISTING_DIR = "pokemon_dataset_subset_/train"  # Vos images déjà générées
    OUTPUT_DIR = "pokemon_dataset_advanced/train"
    
    # Options de contrôle
    force_regen = False  # Mettre True pour tout regénérer
    use_existing = True  # Mettre False pour ignorer les images existantes

    if force_regen or not os.path.exists(OUTPUT_DIR) or len(os.listdir(OUTPUT_DIR)) == 0:
        print("📸 Génération des images augmentées avec transformations avancées...")
        if use_existing and os.path.exists(EXISTING_DIR):
            print(f"🔄 Réutilisation des images existantes depuis {EXISTING_DIR}")
            generate_augmented_images(INPUT_DIR, OUTPUT_DIR, EXISTING_DIR)
        else:
            print("🆕 Génération complète depuis zéro")
            generate_augmented_images(INPUT_DIR, OUTPUT_DIR)
    else:
        print("✅ Images déjà augmentées, on passe à l'entraînement.")

    # Transformations pour l'entraînement (plus légères car déjà augmentées)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # Augmentations légères supplémentaires pendant l'entraînement
        transforms.RandomErasing(p=0.05, scale=(0.01, 0.05))
    ])

    # Transformations pour la validation (sans augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Chargement des données
    full_dataset = RGBImageFolder(OUTPUT_DIR, transform=train_transform)
    train_size = int(0.85 * len(full_dataset))  # Un peu plus de données d'entraînement
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Appliquer les transformations de validation au dataset de validation
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"🧠 Nombre de classes : {len(full_dataset.classes)}")
    print(f"🖼️ Total d'images : {len(full_dataset)}")
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 🏗️ Création du modèle avec dropout pour éviter l'overfitting
    model = models.resnet18(weights="IMAGENET1K_V1")
    
    # Ajout de dropout avant la couche finale
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(full_dataset.classes))
    )
    model = model.to(device)

    # Fonction de perte avec label smoothing pour plus de robustesse
    class LabelSmoothingLoss(nn.Module):
        def __init__(self, classes, smoothing=0.1):
            super(LabelSmoothingLoss, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.classes = classes

        def forward(self, pred, target):
            pred = pred.log_softmax(dim=-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.classes - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            return torch.mean(torch.sum(-true_dist * pred, dim=-1))

    criterion = LabelSmoothingLoss(len(full_dataset.classes), smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Scheduler plus sophistiqué
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)

    # 🔐 Early stopping amélioré
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    epochs = 15  # Plus d'époques avec early stopping
    train_losses = []
    val_losses = []
    val_accuracies = []

    print(f"🚀 Début de l'entraînement sur {device}")

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        train_pbar = tqdm(train_loader, desc=f"Époque {epoch+1}/{epochs} - Train")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping pour stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Calcul précision
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Mise à jour barre de progression
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct_predictions/total_predictions:.2f}%'
            })

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100. * correct_predictions / total_predictions
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Époque {epoch+1}/{epochs} - Val")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct_val/total_val:.2f}%'
                })

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"📊 Epoch {epoch+1}:")
        print(f"   Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%")
        print(f"   Val   - Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}%")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping et sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'class_names': full_dataset.classes
            }, "best_pokemon_model.pth")
            print("💾 Nouveau meilleur modèle sauvegardé !")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping à l'époque {epoch+1}")
                break

        scheduler.step()
        print("-" * 50)

    # 💾 Sauvegarde finale
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': full_dataset.classes
    }, "final_pokemon_model.pth")
    print("📦 Modèle final sauvegardé")

    # 📊 Visualisation des résultats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Courbes de perte
    ax1.plot(train_losses, label="Train Loss", color='blue')
    ax1.plot(val_losses, label="Validation Loss", color='red')
    ax1.set_xlabel("Époques")
    ax1.set_ylabel("Perte")
    ax1.set_title("Évolution de la fonction de perte")
    ax1.legend()
    ax1.grid(True)

    # Courbe de précision
    ax2.plot(val_accuracies, label="Validation Accuracy", color='green')
    ax2.set_xlabel("Époques")
    ax2.set_ylabel("Précision (%)")
    ax2.set_title("Évolution de la précision de validation")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # 🧪 Évaluation finale avec matrice de confusion
    print("📊 Génération de la matrice de confusion...")
    
    # Chargement du meilleur modèle
    checkpoint = torch.load("best_pokemon_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Évaluation finale"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    
    # Affichage de la matrice (limitée si trop de classes)
    num_classes = len(full_dataset.classes)
    if num_classes <= 20:
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)
        disp.plot(xticks_rotation="vertical", cmap="Blues")
        plt.title(f"Matrice de confusion - Précision finale: {val_accuracies[-1]:.2f}%")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Trop de classes ({num_classes}) pour afficher la matrice de confusion complète")
        
    # Statistiques finales
    final_accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"🎯 Précision finale sur validation: {final_accuracy:.2f}%")
    print(f"🏆 Meilleure précision atteinte: {max(val_accuracies):.2f}%")
    
    print("\n✅ Entraînement terminé ! Le modèle est maintenant prêt pour la reconnaissance en temps réel.")
    print("💡 Conseils pour l'utilisation en temps réel :")
    print("   - Utilisez une bonne illumination pour réduire le bruit")
    print("   - Évitez les mouvements brusques (flou de mouvement)")
    print("   - Le modèle peut maintenant gérer les reflets et arrière-plans variés")