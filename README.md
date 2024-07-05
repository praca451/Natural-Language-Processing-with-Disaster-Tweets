# Natural-Language-Processing-with-Disaster-Tweets

Overview
This repository contains a machine learning project focused on natural language processing (NLP) to identify disaster-related tweets. The project involves training a model to distinguish between tweets that are about real disasters and those that are not. The dataset used for training is sourced from Kaggle's "Natural Language Processing with Disaster Tweets" competition.

Contents
Natural Language Processing with Disaster Tweets.h5: The trained machine learning model.
notebook.ipynb: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
README.md: This README file providing an overview and instructions for the project.
Project Structure
Data Preprocessing: Cleaning and preparing the tweet text data for model training.
Model Training: Building and training a machine learning model using TensorFlow to classify tweets.
Evaluation: Assessing the performance of the trained model on test data.
Installation
To run the code in this repository, you will need to install the necessary Python libraries. The primary libraries used include TensorFlow, Pandas, and Scikit-learn.

pip install tensorflow pandas scikit-learn

Usage
Clone the repository:
git clone https://github.com/praca451/DisasterTweetsNLP.git
cd DisasterTweetsNLP

Load the Jupyter Notebook:
Open notebook.ipynb in Jupyter Notebook or JupyterLab to view and execute the code cells.

Data Preprocessing:
The notebook includes steps to clean the text data, including removing special characters, stop words, and performing tokenization.

Model Training:
The code trains a machine learning model using the preprocessed data. The model is saved as Natural Language Processing with Disaster Tweets.h5.

Model Evaluation:
Evaluate the model’s performance using accuracy, precision, recall, and F1 score.

Example
Here’s a brief example of how to load and use the trained model to make predictions:

import tensorflow as tf
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model('Natural Language Processing with Disaster Tweets.h5')

# Example tweet
tweet = "There's a forest fire at spot X"

# Preprocess the tweet (follow the preprocessing steps used during training)
# ...

# Make a prediction
prediction = model.predict([preprocessed_tweet])
print("Disaster Tweet" if prediction > 0.5 else "Not a Disaster Tweet")


Contributing
Contributions are welcome! If you have any suggestions or improvements, please open an issue or create a pull request.

License
This project is licensed under the MIT License.


Contact
For any questions or inquiries, please contact reg@tothemoonwithai.com
