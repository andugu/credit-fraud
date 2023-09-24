# 💳 Credit Fraud Detection
This project implements a comprehensive solution for detecting credit card fraud using machine learning. It leverages the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) to build an end-to-end pipeline for fraud detection.

## Key Features
- **Data Version Control (DVC):** We use [DVC](https://github.com/iterative/dvc) to efficiently manage and version control our dataset and model files. This ensures reproducibility and easy collaboration.
- **Serving with BentoML:** Our model is deployed using [BentoML](https://github.com/bentoml/BentoML), a powerful framework for serving machine learning models. This allows for seamless integration into production environments.
- **XGBoost Binary Classifier:** The heart of our fraud detection system is a binary classifier built using the [XGBoost native library](https://xgboost.readthedocs.io/en/latest/python/python_api.html), known for its high performance and accuracy.

## Getting Started
To run this project, follow these simple steps:

1. Clone the repository:
   ```
   git clone https://github.com/andugu/credit-fraud.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Use DVC to reproduce the pipeline:
   ```
   dvc repro
   ```

4. Start serving the model on the default port (8000) of localhost:
   ```
   bentoml serve bentoml/FraudClassifier --port 8000
   ```
   To run the service on a different port, simply replace `8000` with the desired port.

## Project Workflow
The DAG of the project is as follows:

```
+---------+    
| Prepare |    
+---------+    
     |         
     |         
     |         
+----------------+ 
| Feature Engine | 
+----------------+ 
     |         
     |         
     |       
 +---------+ 
 |  Train  |
 +---------+
     |         
     |         
     |       
 +--------+      
 |  Pack  |      
 +--------+      
     |         
     |         
     |         
+---------+      
|  Serve  |      
+---------+      
```
