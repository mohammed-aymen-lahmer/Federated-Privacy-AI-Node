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
