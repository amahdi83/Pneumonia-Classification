# Pneumonia Classification

This project involves comparing the performance of various machine learning models on a pneumonia classification task. The models evaluated include a custom Convolutional Neural Network (CNN), transfer learning, partial fine-tuning, and full fine-tuning.

## Models Evaluated

- **Custom CNN**: A lightweight model with minimal parameters designed for fast training. This model is built from scratch and optimized for speed.
- **Transfer Learning**: Utilizes pre-trained models to leverage learned features from other tasks. This approach balances between training time and model performance.
- **Partial Fine-Tuning**: Fine-tunes ten layers of a pre-trained model to enhance performance while being more memory-efficient than full fine-tuning.
- **Full Fine-Tuning**: Involves fine-tuning all layers of a pre-trained model. 

## Dataset

The dataset used for this project is the [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) dataset from Kaggle. It contains chest X-ray images classified into two categories: Pneumonia and Normal. 

## Analysis Approach

The analysis involves the following steps:

1. **Data Preprocessing**: Loading and preprocessing the dataset, including image resizing, normalization, and splitting into training and validation sets.
2. **Model Training**: Training each model (Custom CNN, Transfer Learning, Partial Fine-Tuning, and Full Fine-Tuning) on the preprocessed dataset.
3. **Performance Evaluation**: Evaluating the models based on prediction accuracy and precision.
4. **Comparison**: Comparing the models' performance metrics and analyzing the trade-offs between training time, memory usage, and accuracy.

## Requirements

To run the code and reproduce the results, ensure the following libraries are installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- SciPy
- Matplotlib
- Seaborn

Use the following command to install the dependencies:

```bash
pip install -r requirements.txt
