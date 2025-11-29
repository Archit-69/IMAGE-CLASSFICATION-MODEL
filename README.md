# CIFAR-10 CNN Classifier

This repository contains a simple Convolutional Neural Network (CNN) implemented in PyTorch
to classify images from the CIFAR-10 dataset.
------------------------------------------------------------------

## Contents
- `train.py` : Python script to train and evaluate the CNN on CIFAR-10.
- `requirements.txt` : Python dependencies.
- `model_checkpoint.pth` : (generated after training) saved model weights.
-------------------------------------------------------------------------------------
## Dataset
CIFAR-10 has 60,000 color images (32x32) in 10 classes. The script will download the dataset automatically.
-------------------------------------------------------------------
## Model
The model is a compact CNN with:
- Two convolutional layers (with BatchNorm and ReLU)
- Max pooling and dropout
- Two fully connected layers
This architecture is intentionally small so it trains reasonably quickly for demonstration.
-----------------------------------------------------------------------------------
## Installation
1. Create a virtual environment (recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
-----------------------------------------------------------------------------------
## Usage
Train with default settings:
```
python train.py
```
------------------------------------------------------------------------------------
Options:
- `--epochs` : Number of epochs (default 10)
- `--batch-size` : Batch size (default 64)
- `--lr` : Learning rate (default 0.001)
- `--data-dir` : Directory to download CIFAR-10 (default ./data)
- `--no-cuda` : Disable GPU even if available
------------------------------------------------------------------------------------------
Example:
```
python train.py --epochs 5 --batch-size 128
```
----------------------------------------------------------------------------------------------------------------
## Output
- Training progress printed to console.
- Final test accuracy and confusion matrix printed.
- Model checkpoint saved to `model_checkpoint.pth`.
- <img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/9e25ccda-e68e-486e-b4c6-ffb5451f3285" />
------------------------------------------------------------------------------------------------------------------------------
## Notes
- Using a GPU (CUDA) is recommended for faster training.
- Feel free to modify the architecture, hyperparameters, or add training enhancements.
------------------------------------------------------------------------------------------------------------
* Company: CODTECH IT SOLUTIONS
* Name: Archit kapre
* Intern ID:CT04DR999
* Domain: Machine learning 
* Duration: 4 weeks
* Mentor: Neela Santosh


