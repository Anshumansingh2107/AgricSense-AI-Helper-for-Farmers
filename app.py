from flask import Flask, request, render_template
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
import os
import json
import markdown
from langchain_ollama import OllamaLLM  # Use this import for Ollama models
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Loading models for crop yield prediction
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Load the plant disease prediction model
plant_disease_model = tf.keras.models.load_model('Trained_Model/plant_disease_prediction_model.h5')

# Load class indices for plant disease model
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Initialize Flask app
app = Flask(__name__)

# Initialize the Ollama LLM
llm = OllamaLLM(model="llama3", timeout=30)  # Specify your Ollama model here

# Define a prompt template for predicting crop yield using LangChain
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
    You are a crop yield prediction assistant. Given the following information, provide a comprehensive response:
    {query}
    Include a short introductory paragraph followed by a bullet-point list of key insights or predictions.
    """,
)

# Create the LangChain LLM chain to process input using LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to generate prediction based on a natural language query
def predict_crop_yield(query):
    response = llm_chain.run({"query": query})  # Pass query as a dictionary
    return response

# Function to preprocess the uploaded image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize the pixel values
    return img_array

# Function to predict plant disease
def predict_plant_disease(image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = plant_disease_model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]  # Map index to class name
    return predicted_class_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data for crop yield prediction
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        # Create feature array and preprocess
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', prediction=prediction)

# Route to handle natural language predictions using LangChain
@app.route("/langchain_predict", methods=['POST'])
def langchain_predict():
    if request.method == 'POST':
        query = request.form['query']
        prediction = predict_crop_yield(query)

        # Format the prediction for display using markdown conversion
        formatted_prediction = format_prediction(prediction)

        return render_template('index.html', langchain_prediction=formatted_prediction)

def format_prediction(prediction):
    # Convert markdown-formatted prediction to HTML
    html_output = markdown.markdown(prediction)
    return html_output

# Route to handle plant disease prediction
@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'image' not in request.files:
        return 'No image file uploaded', 400

    image = request.files['image']

    # Make disease prediction
    predicted_disease = predict_plant_disease(image)

    return render_template('index.html', disease_prediction=predicted_disease)

if __name__ == "__main__":
    app.run(debug=True)