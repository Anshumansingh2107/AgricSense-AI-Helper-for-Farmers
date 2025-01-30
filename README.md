# AgriSense: Data-Driven Crop Yield and Disease Prediction System

Dashboard link: https://app.powerbi.com/groups/me/reports/fbfd226f-e974-4e80-9996-2645d41eb1b1/8c9a3804b63a5e3a961f?experience=power-bi

## Description

AgriSense is an AI-powered platform aimed at revolutionizing agriculture by providing farmers with in-depth data analysis and predictive insights on crop yield and disease detection. This project leverages data visualization, machine learning models, and image analysis to help farmers make informed decisions to improve crop productivity and manage diseases effectively.


## Key Features

### Crop Yield Data Analysis:
- Interactive Power BI dashboard to explore and analyze crop yield data.
- Visualize trends, correlations, and key metrics that influence agricultural outcomes.

### Crop Yield Prediction:
- A machine learning model that predicts optimal crop yields based on historical data.
- Analyzes multiple factors, including soil, weather, and past performance, to provide accurate forecasts.

### Smart AI Assistant:
- A smart assistant built using large language models (LLMs) that answers agriculture-related questions in real-time.
- Provides guidance on best farming practices, crop management, and troubleshooting.

### Disease Detection via Image Analysis:
- A deep learning model that detects crop diseases by analyzing plant images.
- Trained on the PlantVillage dataset, the model identifies common diseases like Apple Rust and Black Rot.
- Farmers can upload images of crops for real-time disease diagnosis.

### Model Performance Tracking:
- Graphical visualizations of model accuracy and loss during training and validation.
- Provides insight into the model's effectiveness and improvements over time.


# Running the Models

![Screenshot 2024-10-15 002417](https://github.com/user-attachments/assets/8063c7df-2066-430f-850b-80b162c88733)

## Power BI Dashboard:
- Open the crop_yield.pbix file in Power BI Desktop.
- Explore and analyze crop yield data using interactive visualizations.
![Screenshot 2024-10-15 141403](https://github.com/user-attachments/assets/2d9c2657-63c5-4145-b5f6-a13789dfbdd1)

## Crop Yield Prediction:
- Open the CropYield-Prediction.ipynb file in Jupyter Notebook.
- Follow the steps to preprocess the data, train the model, and evaluate predictions.
![Screenshot 2024-10-13 223309](https://github.com/user-attachments/assets/d1ba7ec7-9540-46ce-9d41-a702fe3bbd7e)

## Disease Detection Model:
- Ensure the dataset is unzipped and paths are correctly set.
- Train the image classification model for disease detection using TensorFlow and Keras.
- Test the model by uploading an image from the images/ folder.
![Screenshot 2024-10-15 122123](https://github.com/user-attachments/assets/f3683917-0384-4f51-8036-5207461556fe)

## Smart AI Chatbot (AgriSense Assistant):
- Interact with the AI-powered chatbot, built using large language models (LLMs), for real-time support.
- Ask questions related to farming practices, crop management, and disease prevention.
- The chatbot provides expert recommendations, answers, and insights based on the data analysis and disease detection results, helping farmers make well-informed decisions.
![Screenshot 2024-10-15 001701](https://github.com/user-attachments/assets/040511bb-62db-4486-b541-d4cdac21cda4)



# Tech Stack

- Power BI: For data analysis and visualization.
- Python: For data preprocessing, machine learning model development, and API handling.
- TensorFlow/Keras: For building deep learning models for crop disease prediction.
- Pillow/Matplotlib: For image processing and visualization.
- Flask: For web app development (if applicable).
- Large Language Models (LLMs): For the smart AI assistant.
- ImageDataGenerator: For image augmentation in model training.


# Project Structure

AgriSense/

├── CropYield-Prediction.ipynb  
├── app.py                        
├── crop_yield.pbix              
├── class_indices.json            
├── plant_disease_prediction_model.h5 
├── kaggle.json                   
├── README.md                     
├── requirements.txt              
└── images/                       


# Getting Started
## Prerequisites

1. Prerequisites:
- Python 3.7+
- Power BI Desktop (for opening the .pbix file)
- Jupyter Notebook (if running the notebooks locally)
- Kaggle API Key (for downloading the dataset)
Install the required Python libraries by running:
- pip install -r requirements.txt

2. Dataset:
The project uses the PlantVillage Dataset for crop disease prediction. You can download the dataset by setting up your Kaggle credentials:
- Download kaggle.json from your Kaggle account.
- Place the file in your project directory.
- Run the following code in the notebook to download the dataset:
!pip install kaggle
kaggle_credentails = json.load(open("kaggle.json"))
os.environ['KAGGLE_USERNAME'] = kaggle_credentails["username"]
os.environ['KAGGLE_KEY'] = kaggle_credentails["key"]
!kaggle datasets download -d abdallahalidev/plantvillage-dataset

# Model Evaluation

- The crop yield prediction model was trained on historical agricultural data and provides accurate yield forecasts based on a variety of factors.
- The disease detection model achieved high accuracy in classifying crop diseases and can detect early signs of plant infections, helping farmers take preventive actions.
