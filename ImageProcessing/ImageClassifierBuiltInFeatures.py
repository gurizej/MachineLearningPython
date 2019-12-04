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

# Populate labels and features
for im in images:
    labels.append(im[:-len('00.jpg')])
    im = mh.imread(im)
    im = mh.colors.rgb2gray(im, dtype=np.uint8)
    features.append(mh.features.haralick(im).ravel())

features = np.array(features)
labels = np.array(labels)

clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression())])

cv = LeaveOneOut()
scores = cross_val_score(clf, features, labels, cv=cv)

print('Accuracy: {:.1%}'.format(scores.mean()))