A Comparative Study of CNN and Vision Transformer for Histopathology Classification📌 Project OverviewThis repository contains the complete experimental pipeline for classifying lung and colon cancer histopathology images. The study performs a head-to-head comparison between Convolutional Neural Networks (ResNet50) and Vision Transformers (ViT), while also evaluating the efficacy of Macenko Stain Normalization in the preprocessing pipeline.Key Research Questions:Does the global attention mechanism of Vision Transformers outperform the local feature extraction of CNNs in medical imaging?Does Macenko Stain Normalization improve model robustness and accuracy on the LC25000 dataset?📊 Performance SummaryAfter 20 epochs of training using transfer learning, the results were as follows:Model IDArchitectureStain NormalizationTest AccuracyMacro F1-ScoreModel 1ResNet50 (Baseline)None97.07%0.9699Model 2ResNet50Macenko96.53%0.9645Model 3Vision TransformerNone97.36%0.9729Model 4Vision TransformerMacenko97.36%0.9729Key Finding: The Vision Transformer achieved the highest accuracy. Interestingly, stain normalization did not provide a performance boost, suggesting the LC25000 dataset possesses significant inherent color consistency.📁 Repository StructurePlaintext.
├── scripts/
│   ├── dataset.py      # Custom Dataset class with Macenko normalization
│   ├── model.py        # Model definitions (ResNet50 & ViT-B/16)
│   ├── train.py        # Training & validation loop logic
│   └── eval.py         # Comprehensive evaluation & plotting
├── results/
│   ├── plots/          # Confusion matrices and training curves
│   ├── models/         # Best saved weights (.pth)
│   └── metrics.json    # Final recorded performance metrics
├── requirements.txt    # Necessary Python packages
└── .gitignore          # Prevents uploading large data/envs
🧬 DatasetThe project utilizes the LC25000 dataset, which consists of 25,000 images across 5 classes:colon_aca: Colon Adenocarcinomacolon_n: Benign Colon Tissuelung_aca: Lung Adenocarcinomalung_n: Benign Lung Tissuelung_scc: Lung Squamous Cell Carcinoma[!IMPORTANT]The dataset is not included in this repository. Download it from Kaggle and place it in a folder named data/.🚀 Getting Started1. InstallationBash# Clone the repository
git clone https://github.com/your-username/lung-colon-cancer-classification.git
cd lung-colon-cancer-classification

# Install dependencies
pip install -r requirements.txt
2. Running ExperimentsTo replicate the study, run the following commands:Bash# Train Model 1 (ResNet Baseline)
python scripts/train.py --model_type resnet

# Train Model 3 (ViT Challenger)
python scripts/train.py --model_type vit

# Train with Stain Normalization (Models 2 & 4)
python scripts/train.py --model_type resnet --use_stain_norm
python scripts/train.py --model_type vit --use_stain_norm
📈 VisualizationsYou can find full Confusion Matrices and Accuracy/Loss curves in the results/plots/ directory.
