# credit-fraud

A simple python project based on [Kaagle's Credit Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
This project implements a vertical pipeline over the dataset, from processing to the serving of the model.
[DVC](https://github.com/iterative/dvc) is used for tracking, [BentoML](https://github.com/bentoml/BentoML) for serving,
and the classifier is implemented with [XGBoost natives library](https://xgboost.readthedocs.io/en/latest/python/python_api.html).

To run this project simply use: <br>
`dvc repro` <br>
The serving service is loaded by default on port 8000 of localhost .
In order to run the service on a different port, use the following command once the pack stage has run: <br>
` bentoml serve bentoml/FraudClassifier --port 8000`

The DAG of the project is as follows:

        +---------+    
        | prepare |    
        +---------+    
             *         
             *         
             *         
    +----------------+ 
    | feature_engenc | 
    +----------------+ 
             *         
             *         
             *       
         +-------+ 
         | train |
         +-------+
             *         
             *         
             *       
         +------+      
         | pack |      
         +------+      
             *         
             *         
             *         
        +-------+      
        | serve |      
        +-------+      
