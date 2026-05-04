# Chest X-Ray Pneumonia Classifier

A deep learning model that detects pneumonia from chest X-rays 
using Python, TensorFlow, and transfer learning (MobileNetV2).

## Results
- 97% accuracy on 624 test images
- Dataset: Kaggle Chest X-Ray Images (5,863 images)

## How it works
1. Loads and augments chest X-ray images
2. Uses pretrained MobileNetV2 as a feature extractor
3. Adds custom classification layers on top
4. Trains for 10 epochs using binary crossentropy loss

## Technologies
- Python 3.11
- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- scikit-learn, matplotlib
