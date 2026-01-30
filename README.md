# Lung and Colon Cancer Classification: CNN vs. Vision Transformer

This research project evaluates the performance of **ResNet50** and **Vision Transformers (ViT)** for the automated classification of histopathology images into five categories of lung and colon tissue.

## 📊 Key Results
* **Best Model:** Vision Transformer (ViT-B/16)
* **Top Accuracy:** **97.36%**
* **Key Finding:** Stain normalization did not significantly improve accuracy for this dataset, suggesting a high degree of inherent color consistency.

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Framework:** PyTorch
* **Architectures:** ResNet50, ViT-B/16
* **Preprocessing:** Macenko Stain Normalization (`staintools`)

## 📂 Dataset
We used the **LC25000 dataset**, containing 25,000 images of lung and colon tissue. Due to size constraints, the data is not included in this repo. You can download it [here](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).

## 🚀 How to Run
1. **Clone the repo:** `git clone https://github.com/your-username/your-repo-name.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Train Model 3 (ViT):** `python scripts/train.py --model_type vit`
