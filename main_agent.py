# ==============================================================================
# PROJECT: PRIVACY-PRESERVING FEDERATED AI NODE
# AUTHOR:  Mohamed Aymen Lahmer (MSc Bioinformatics )
# DATE:    December 2025
# 
# This code sends the AI to the data, not the data to the AI.
# It protects patient privacy better than I protect my own passwords.
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import copy

# --- 1. THE BRAIN (Model Architecture) ---
def initialiser_cerveau():
    print(">>> [LAHMER-AGENT] Loading the neural architecture...")
    
    # Downloading the brain structure.
    # We use 'DEFAULT' weights (ImageNet) because training from scratch takes 84 years.
    try:
        model = models.resnet18(weights='DEFAULT')
    except:
        # Fallback for older PyTorch versions (back in the dinosaurs era)
        model = models.resnet18(pretrained=True)
    
    # Fine-tuning: ResNet thinks it sees 1000 things (dogs, cats, planes...).
    # We force it to focus on just 2 things: Cancer vs Normal.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    return model

# --- 2. THE GYM (Local Training Loop) ---
def entrainement_securise(dossier_donnees):
    # Data preprocessing magic.
    # We resize everything to 224x224 because the AI gets confused if images are too big.
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalization math (don't ask, it just works better this way)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Security Check: Does the folder actually exist?
    if not os.path.exists(dossier_donnees):
        return None, "Folder not found (Are you sure you typed it right?)"

    # Loading the images
    try:
        dataset = datasets.ImageFolder(dossier_donnees, transformations)
        if len(dataset) == 0:
            return None, "Folder is empty. The AI cannot learn from void."
            
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        classes = dataset.classes 
        taille_dataset = len(dataset)
    except Exception as e:
        return None, f"Read Error: {str(e)}"

    print(f"\n>>> [PRIVACY MODE] Scanning {taille_dataset} confidential images locally.")
    print(f">>> [INFO] Classes found: {classes}")
    
    # Hardware check: Are we running on a Ferrari (GPU) or a Potato (CPU)?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">>> [HARDWARE] Running on: {device}")
    
    model = initialiser_cerveau().to(device)
    criterion = nn.CrossEntropyLoss()
    # SGD Optimizer: The teacher that corrects the AI when it makes mistakes
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # STARTING THE WORKOUT
    model.train()
    print(">>> [LAHMER-AGENT] Start auto-training (fingers crossed)...")
    
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # Resetting the brain before new info
        outputs = model(inputs)
        loss = criterion(outputs, labels) # Calculating how wrong the AI is
        loss.backward() # Backpropagation (The math magic)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / taille_dataset
    print(f">>> [SUCCESS] Training done. Final Loss: {epoch_loss:.4f} (Not bad!)")
    
    return model, "Success"

# --- 3. THE SMART INTERFACE (User Experience) ---
if __name__ == "__main__":
    print("===============================================================")
    print("   🏥  FEDERATED AI AGENT - HOSPITAL NODE (SECURE MODE)      ")
    print("   coded by M.A. Lahmer                                      ")
    print("===============================================================")
    
    # The standard folder name
    dossier_racine = "data"
    
    # --- CHECK 1: Does the main folder exist? ---
    if not os.path.exists(dossier_racine):
        print(f" CRITICAL ERROR: The folder '{dossier_racine}' is missing.")
        print(" SOLUTION: Please create a 'data' folder. It's not rocket science! :v ")
    
    else:
        # --- CHECK 2: Are the subfolders correct? ---
        sous_dossiers_requis = ["Normal", "Cancer"]
        dossiers_manquants = [d for d in sous_dossiers_requis if not os.path.exists(os.path.join(dossier_racine, d))]
        
        if dossiers_manquants:
            print(f"❌ STRUCTURE ERROR: Missing folders -> {dossiers_manquants}")
            print("💡 SOLUTION: Inside 'data', you need exactly:")
            print("   📂 data")
            print("    ├── 📂 Normal  (Healthy stuff)")
            print("    └── 📂 Cancer  (Tumor stuff)")
        
        else:
            # --- LAUNCH IF EVERYTHING IS GOOD ---
            print("✅ STRUCTURE VALIDATED. Launching the training protocol...")
            
            # Run the beast
            modele_entraine, statut = entrainement_securise(dossier_racine)
            
            if modele_entraine:
                nom_fichier_export = "poids_ia_lahmer_v1.pth"
                torch.save(modele_entraine.state_dict(), nom_fichier_export)
                
                print("\n" + "="*60)
                print(" SUCCESS: The AI has learned from your local data.")
                print(f" FILE GENERATED: {nom_fichier_export}")
                print(" ACTION REQUIRED: Send ONLY this file to the central server.")
                print(" SECURITY: No images were harmed (or stolen) in this process.")
                print("=")
            else:
                print(f" FAILURE: {statut}")