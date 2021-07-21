import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pickle
import random
import numpy as np
from tensorflow.keras import models 
import json
import tkinter
from tkinter import *

data = open('intents.json').read()
intents = json.loads(data)
with open('words.pkl', 'rb') as infile:
    words_new = pickle.load(infile)
with open('classes.pkl', 'rb') as infile:
    classes = pickle.load(infile)
model = models.load_model('bot_model.h5')
wordnet_lemmatizer = WordNetLemmatizer()
punctuations = ['?', '!', ',']

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

def chatbot_response(text):
    results_ = predict_class(text, model)
    tag = results_[0]['_tag']
    list_of_intents = intents['intents']
    for intent in list_of_intents:
        if intent['tag']== tag:
            responses = random.choice(intent['responses']) # give a random response from the list of responses for that tag
            break
    return responses

#Creating GUI with tkinter
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)

#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()