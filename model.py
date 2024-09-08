import pickle
import numpy as np

# Load the trained model from the pickle file
def load_model():
    with open('gb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Define the prediction function using the loaded model
def predict_susceptibility(model, amt, age, gender):
    # Convert the input into the appropriate format for the model
    input_data = np.array([[amt, age, gender]])
    # Make prediction
    prediction = model.predict(input_data)
    # Return the result as a human-readable string
    return "Highly Susceptible" if prediction == 1 else "Low Susceptibility"
