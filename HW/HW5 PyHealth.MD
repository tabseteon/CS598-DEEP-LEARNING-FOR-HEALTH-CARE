# PyHealth
### PyHealth is an open-source Python package that provides an end-to-end pipeline for clinical predictive modeling
### In PyHealth, a machine learning pipeline consists of five stages:
- Dataset Processing 
- Define Healthcare Task 
- Build ML Model 
- Train ML Model 
- Inference & Evaluation

## 1 Dataset Processing (The pyhealth.datasets Module)
### The pyhealth.datasets module provides processing functions for various public EHR datasets, including MIMIC-III, MIMIC-IV, eICU, and all OMOP-CDM-based datasets. It transforms unstructured data into a unified structured object.

## 2 Define Healthcare Task (The pyhealth.tasks Module)
### The pyhealth.tasks module provides functions to define various healthcare ML tasks from EHR data. It transforms structured datasets into samples (features and labels) ready for model training.

## 3 Build ML Model (The pyhealth.models Module)
### The pyhealth.models module provides common deep learning models such as RNN, CNN, and Transformer, along with specialized healthcare models like RETAIN, SafeDrug, and GAMENet. Most models can be applied to a wide range of healthcare prediction tasks, except for a few task-specific models (e.g., GAMENet, SafeDrug, and MICRON, which are designed exclusively for drug recommendation).

## 4 Train Model (The pyhealth.trainer Module)
### The pyhealth.trainer module serves as the training handler for deep learning models. It provides a streamlined workflow for model training, including logging, checkpointing, and performance monitoring.

## 5 Model Evaluation (One-line Evaluation)
### The Trainer class provides the .evaluate(data_loader) method to compute performance metrics on any test dataset. This will output a dictionary containing the evaluation metrics

