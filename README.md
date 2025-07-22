# LSTM-based price movement prediction

This project implements an LSTM-based deep learning model to predict future price movements in time series data. It incorporates a variety of engineered features—including the Inverse Fast Fourier Transform (IFFT)—to improve prediction accuracy. The model is built using PyTorch and is designed to be modular and easily extensible for financial or signal forecasting tasks.

# Features
- LSTM model for sequence-based price prediction
- Incorporates multiple technical indicators including IFFT
- Custom PyTorch Dataset and DataLoader for flexible data handling
- Training loop with real-time loss tracking and evaluation
- Visualization of model predictions and performance metrics

# Model Overview
- Uses LSTM layers to learn temporal dependencies
- Input features may include:
    - Raw prices
    - IFFT-transformed signals
    - Momentum indicators
    - Moving averages
    - Other custom technical features


# Project Structure
IFFT-LSTM/
├── IFFT-LSTM.ipynb    # Main notebook with model code and training logic
├── data/              # Folder for input CSVs or time series (user-provided)
└── README.md          # This documentation

# Requirments
- Python 3.8+
- PyTorch
- torchvision
- timm
- numpy, pandas, matplotlib
- scikit-learn
- tqdm

pip install torch torchvision timm numpy pandas matplotlib scikit-learn tqdm
jupyter notebook IFFT-LSTM.ipynb

# Training & Evaluation

The notebook includes:
- Feature engineering from raw time series data
- Sequence creation for LSTM input
- Model training and validation
- Loss visualization and prediction plotting
