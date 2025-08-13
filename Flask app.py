import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify


app = Flask(__name__, template_folder="templates") # Flask App Setup

 
MODEL_PATH = os.path.join("model", "model.pkl") #Load Model 

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/") #Routes
def home():
    """Serve the main HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    size_str = request.form.get("size")

    
    try: # Validate input
        size = float(size_str)
    except (TypeError, ValueError):
        return jsonify(error="Invalid input for size."), 400

    
    prediction = model.predict(np.array([[size]]))[0] # Predict
    return jsonify(result=f"Predicted Rent: {prediction:,.2f}")

if __name__ == "__main__": #Run Server

    app.run(debug=True)
