# Import Libraries
from flask import Flask, jsonify, request
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pickle
import random
import numpy as np
from tensorflow.keras import models 
import json

# Load Stored Variables
data = open('intents.json').read()
intents = json.loads(data)
with open('words.pkl', 'rb') as infile:
    words_new = pickle.load(infile)
with open('classes.pkl', 'rb') as infile:
    classes = pickle.load(infile)
model = models.load_model('bot_model.h5')
wordnet_lemmatizer = WordNetLemmatizer()
punctuations = ['?', '!', ',']


# Preprocess Text Input (User Question) - Tokenization & Lemmatization
def preprocess_pattern(pattern, show_details=True):
    # return words array: 0 or 1 for each word in the bag that exists in the pattern
    input_array = np.zeros((len(words_new), ))
    # tokenize
    tokenized_words = nltk.word_tokenize(pattern)
    #lemmatize
    lemmatized_pattern_words = []
    for pattern_word in tokenized_words:
        lemma = wordnet_lemmatizer.lemmatize(pattern_word.lower())
        lemmatized_pattern_words.append(lemma)

    for idx, w in enumerate(words_new):
        if w in lemmatized_pattern_words:
            input_array[idx] = 1
            if show_details:
                print('found word in word_list')
    return input_array

# Predict Classes - Categories / Classes
def predict_class(pattern, model):
    # filter out predictions below a threshold
    input_array = preprocess_pattern(pattern, show_details=False)
    y_predict = model.predict(input_array.reshape(1, -1))
    results = []
    for idx, y in enumerate(y_predict.reshape(-1, 1).ravel()):
        results.append([idx, y])
        results.sort(key=lambda x: x[1], reverse=True)

    results_list = []
    for result in results:
        results_list.append({'_tag': classes[result[0]], 'probability' : np.round(result[1], 3)})
    return results_list


# Bot Response
def chatbot_response(text):
    results_ = predict_class(text, model)
    tag = results_[0]['_tag']
    list_of_intents = intents['intents']
    for intent in list_of_intents:
        if intent['tag']== tag:
            responses = random.choice(intent['responses']) # give a random response from the list of responses for that tag
            break
    return responses


# API Call
# Initialize Flask App
app = Flask(__name__)

@app.route("/bot_response",  methods = ["POST"])
def bot_response():
    pred_data = []
    msg = request.json.get("data")
    bot_reply = chatbot_response(msg)
    response = {
        "description" : 'Chat Bot',
        "Bot" : bot_reply
    }
    return jsonify(bot_reply), 200

#  Main Loop or App Running Entry
if __name__ == "__main__":
    app.run(port=5000, debug=True)