import os
from textblob import TextBlob

DIR = "../TextDocData"
stories = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]

def getStorySentiment(story):
    analysis = TextBlob(story)
    if analysis.sentiment.polarity > 0: 
        return 'positive'
    elif analysis.sentiment.polarity == 0: 
        return 'neutral'
    else: 
        return 'negative'

for story in stories:
    result = getStorySentiment(story)
    print(result)
    if (result == 'neutral'):
        print(story)