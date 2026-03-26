# Detection of Chagas Disease from ECG using Machine Learning

This project focuses on detecting Chagas disease from electrocardiogram (ECG) signals using machine learning and deep learning methods. The project is based on the George B. Moody PhysioNet Challenge 2025 dataset.

## Project Goal
The goal of this project is to compare and extend machine learning approaches for detecting Chagas disease from ECG signals, including convolutional neural networks, transformer-based models, and methods for handling weak labels.

## Dataset
The dataset used in this project comes from the PhysioNet Challenge 2025. The dataset contains ECG recordings from multiple sources, including:
- CODE-15% 
- SaMi-Trop
- PTB-XL

Due to the large size of the dataset (~64GB), the data is not included in this repository. The dataset can be downloaded from:
CODE-15%: https://zenodo.org/records/4916206
SaMi-Trop: https://zenodo.org/records/4905618
PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/

The labels for Chagas disease are downloaded separately
 
For CODE-15% you can find them in this webpage: https://moody-challenge.physionet.org/2025/
SaMi-trop Chagas labels are all positive
PTB-XL Chagas labels are all negative


After downloading, place the dataset in the following directory structure:

data/
    CODE-15%
    Sami-trop
    PTB-XL

## Project Structure