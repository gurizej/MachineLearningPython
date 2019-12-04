#Simply to supress warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from glob import glob
import mahotas as mh
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut


images = glob('../SimpleImageDataset/*.jpg')
features = []
labels = []

def chist(im):
    im = im // 64
    r,g,b = im.transpose((2,0,1))
    pixels = 1 * r + 4 * b + 16 * g
    hist = np.bincount(pixels.ravel(), minlength=64)
    hist = hist.astype(float)
    hist = np.log1p(hist)
    return hist


# Populate labels and features
for im in images:
    labels.append(im[:-len('00.jpg')])

    imcolor = mh.imread(im)
    im = mh.colors.rgb2gray(imcolor, dtype=np.uint8)
    features.append(np.concatenate([mh.features.haralick(im).ravel(),chist(imcolor),]))


features = np.array(features)
labels = np.array(labels)

clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression())])

cv = LeaveOneOut()
scores = cross_val_score(clf, features, labels, cv=cv)

print('Accuracy: {:.1%}'.format(scores.mean()))