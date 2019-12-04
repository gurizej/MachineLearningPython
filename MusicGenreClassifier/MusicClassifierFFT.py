#Simply to supress warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


import scipy
import scipy.io.wavfile
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from matplotlib import pylab
from sklearn import preprocessing

genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
GENRE_DIR = "../MusicDataset/genres/"

def create_fft(fn):
    print(fn)
    sample_rate, X = scipy.io.wavfile.read(fn)
    fft_features = abs(scipy.fft(X)[:1000])
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    scipy.save(data_fn, fft_features)

def read_fft(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    labels = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        file_list = glob.glob(genre_dir)
        for fn in file_list:
            fft_features = scipy.load(fn)
            X.append(fft_features[:1000])
            y.append(label)
    return np.array(X), np.array(y)

# For simplicity we are splitting the data into two equal sets
def getTrainTestData(X, target):
    trainData = []
    testData = []
    trainTarget = []
    testTarget = []
    for i in range(0, len(X)):
        if(i%2 == 0):
            trainData.append(X[i])
            trainTarget.append(target[i])
        else:
            testData.append(X[i])
            testTarget.append(target[i])
    
    return np.array(trainData), np.array(testData), np.array(trainTarget), np.array(testTarget)
    
# Pretty confusion matrix:
def plot_confusion_matrix(cm, genre_list, name, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(False)
    pylab.show()


#Read in the music and create fft of it, if they do not already exist
for genre in genre_list:
    musicFolder = os.path.join(GENRE_DIR, genre)
    music = os.listdir(musicFolder)
    shouldCreate = True
    for song in music:
        if ("fft.npy" in song):
            shouldCreate = False
    if(shouldCreate):
        for song in music:
            create_fft(os.path.join(musicFolder,song))

#Read the data in
X, target = read_fft(genre_list)

#Get training and testing data
trainData, testData, trainTarget, testTarget = getTrainTestData(X, target)

logreg = LogisticRegression()
logreg.fit(trainData, trainTarget)
predictedVals = logreg.predict(testData)

cm = confusion_matrix(testTarget, predictedVals)
print(cm)

#Normalize for better visualization
cmNorm = preprocessing.normalize(cm)

plot_confusion_matrix(cmNorm, genre_list, "name", "Confusion Matrix of FFT Classifier")