import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem
from sklearn.cluster import KMeans

all_data = sklearn.datasets.fetch_20newsgroups(subset='all')

groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']

train_data = sklearn.datasets.fetch_20newsgroups(subset='train', categories=groups)
test_data = sklearn.datasets.fetch_20newsgroups(subset='test', categories=groups)

"""TODO: guri move these to their own class!!!!! """
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
""" up to here """
vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')

vectorized = vectorizer.fit_transform(train_data.data)
num_samples, num_features = vectorized.shape
#print("#samples: %d, #features: %d" % (num_samples, num_features))

num_clusters = 50

km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1, random_state=3)
km.fit(vectorized)

print("#samples: %d, #features: %d" % (num_samples,num_features))