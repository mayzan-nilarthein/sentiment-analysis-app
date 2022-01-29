from cProfile import label
from distutils.log import debug
from email import message
from operator import imod
from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def main():
    if request.method == 'POST':
        input = request.form.get('input')
        sid = SentimentIntensityAnalyzer()
        pos = "{:.2f}".format(sid.polarity_scores(input)['pos']*100)
        neg = "{:.2f}".format(sid.polarity_scores(input)['neg']*100)
        neu = "{:.2f}".format(sid.polarity_scores(input)['neu']*100)
        data = [
            ('Positive',pos),
            ('Negative',neg),
            ('Neutral',neu),
        ]

        labels = [row[0] for row in data]
        values = [row[1] for row in data]
        score = sid.polarity_scores(input)['compound']
        if(score>0):
            label = "Positive"
        elif(score==0):
            label = "Neutral"
        else:
            label= "Negative"
        return render_template('index.html', message= label, keys = labels, values = values)


if __name__ == "__main__": 
    app.run(port='8082', debug=True)

