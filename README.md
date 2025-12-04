# 🏥 Federated Learning Node for Medical Imaging (Privacy-Preserving)

**Author:** Mohamed Aymen Lahmer  
**Stack:** Python, PyTorch, Transfer Learning (ResNet18)

![Badge](https://img.shields.io/badge/AI-Deep_Learning-blue) ![Badge](https://img.shields.io/badge/Privacy-High-green)

## 🧐 What is this?
In healthcare, data is siloed. Hospitals cannot share patient images due to privacy laws (GDPR).
**This project solves the problem.**

Instead of moving the data to the AI, **this agent moves the AI to the data.**
It runs locally on the hospital's machine, trains on confidential images, and **only exports the mathematical weights** (the intelligence), never the pixels.

## 🚀 How it works (The "Lahmer Protocol")
1. **The Brain:** It downloads a pre-trained ResNet18 model (Transfer Learning).
2. **The Training:** It fine-tunes the model locally to distinguish `Normal` vs `Cancer` tissues.
3. **The Privacy:** It ignores internet connection during training. No data leak possible.

## 🛠️ How to use (For Laboratories)

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```
### 2. Prepare your data folder
Create a folder named `data` next to the script with this exact structure:
```text
/data
    /Normal    <-- Put healthy tissue images here (.jpg, .png)
    /Cancer    <-- Put tumor images here
```
### 3. Run the agent
Launch the secure training node:
```bash
python main_agent.py

```
```markdown
### 4. Send the result
The script will generate a file named `poids_ia_lahmer_v1.pth`.
> ⚠️ **Action Required:** Send ONLY this `.pth` file to the central server. It contains the learned patterns, not the patient images.
