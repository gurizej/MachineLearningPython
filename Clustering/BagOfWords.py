from sklearn.feature_extraction.text import CountVectorizer
import os
import scipy as sp
import sys
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer


# Vectorizer with english stemming, meaning it takes into considerations the stemming of words
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

# Vectorizer with term frequency – inverse document frequency (TF-IDF) 
# Basically removed the words that are used in a lot of the document
# The resulting document vectors will not contain counts any more. Instead they will contain the individual TF-IDF values per term.
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', decode_error='ignore')

#This will show how it is splitting raw text into vectors
#print(vectorizer)

#The following content is getting put into vectors
content = ["How to format my hard disk", " Hard disk format problems "]
X = vectorizer.fit_transform(content)
#print(vectorizer.get_feature_names())
#print(X.toarray().transpose())

DIR = "../TextDocData"

posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]
X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
#samples = # of docs, features = # of words
print("#samples: %d, #features: %d" % (num_samples, num_features))
#Vectorize the data
posts_vector = vectorizer.transform(posts)

#Calculating Euclidean distance between vectors
def dist_raw(v1, v2):
    delta = v1-v2
    return sp.linalg.norm(delta.toarray())

#Calculating the Euclidean normalized distance between vectors
def dist_norm(v1, v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

#term frequency – inverse document frequency
def tfidf(term, doc, corpus):
    tf = doc.count(term) / len(doc)
    num_docs_with_term = len([d for d in corpus if term in d])
    idf = sp.log(len(corpus) / num_docs_with_term)
    return tf * idf

best_doc = None
best_dist = float("inf")
best_i = None

for i in range(0,num_samples):
    post_vec_i = posts_vector[i]
    d = float("inf")
    for j in range(0,num_samples):
        if (i==j):
            continue
        post_vec_j = posts_vector[j]
        d = dist_norm(post_vec_i, post_vec_j)
        if (d < best_dist):
            best_dist = d
            best_i = i
        #print("===Post %i with dist=%.2f: %s"%(i,d, posts[i][0:20]))
        #print("===Post %i with dist=%.2f"%(i, d))
print("Best post is %i with dist=%.2f"%(best_i, best_dist))


#Testing out: def tfidf(term, doc, corpus):
a, abb, abc = ["a"], ["a", "b", "b"], ["a", "b", "c"]
D = [a, abb, abc]
print(tfidf("a", a, D))
print(tfidf("b", abb, D))