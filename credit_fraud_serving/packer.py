# Packer
import pickle
from credit_fraud_serving.service import FraudClassifier

# Init a service instance
fraud_classifier_service = FraudClassifier()

# Load the model
with open('pickles/model.pkl', 'rb') as file:
    model = pickle.load(file)
# Load encoder
with open('pickles/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Pack the newly trained model artifact
fraud_classifier_service.pack('model', model)
fraud_classifier_service.pack('encoder', encoder)

# Save the prediction service to disk for model serving
saved_path = fraud_classifier_service.save_to_dir('bentoml')
