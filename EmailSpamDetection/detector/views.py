from django.shortcuts import render
import pandas as pd
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from .forms import MessageForm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = pd.read_csv(os.path.join(BASE_DIR, 'emails.csv'))

# ML setup
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['text'])

X_train, X_test, y_train, y_test = train_test_split(X, dataset['spam'], test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

# function to predict if a message is spam or not
def predictMessage(message):
    messageVector = vectorizer.transform([message])
    prediction = model.predict(messageVector)
    return 'spam' if prediction[0] == 1 else 'Ham'


def Home(request):
    result = None   

    if request.method == 'POST':
        form = MessageForm(request.POST)  

        if form.is_valid():
            message = form.cleaned_data['text']
            result = predictMessage(message)

    else:
        form = MessageForm()

    return render(request, 'index.html', {   
        'form': form,
        'result': result
    })