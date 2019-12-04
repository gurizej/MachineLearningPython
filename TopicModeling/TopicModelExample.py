import os
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import string
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

DIR = "../TextDocData"
stories = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]

#NOTE: more cleaning will yield better results on topics
def tokenizeAndCleanText(text):
    englishStemmer = nltk.stem.SnowballStemmer('english')
    stopWords = stopwords.words('english')
    text = text.lower()

    origTokenizedText = nltk.word_tokenize(text)
    tokenizedText = []

    for token in nltk.word_tokenize(text):
        if (token in string.punctuation or token in stopWords):
            continue
        tokenizedText.append(token)
    return tokenizedText

def split(word): 
    return list(word) 

def replaceIdsWithWords(topic, dictionary):
    inWord = False
    numberToLookUp = []
    toReturn = ""
    for word in topic[1]:
        if (inWord):
            if (word == "\""):
                inWord=False
                lookUpWord = dictionary[int(''.join(numberToLookUp))]
                toReturn = toReturn + lookUpWord + " "
                numberToLookUp = [] #Reset it
            else:
                numberToLookUp.append(word)
        elif (word == "\""):
            inWord=True
        

    #string.replace("what is there", "what is supposed to be")
    print("----------------------------------------")
    return toReturn

tokenizedStories = []
for story in stories:
    tokenizedStories.append(tokenizeAndCleanText(story))

# Holds the actual words
dictionary = Dictionary(tokenizedStories)
# Holds the word ID, and its count
corpus = [dictionary.doc2bow(text) for text in tokenizedStories]

lda = LdaModel(corpus, num_topics=10)

topicToId = [[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]

for t in lda.print_topics():
    printableTopic = replaceIdsWithWords(t, dictionary)
    print(printableTopic)
