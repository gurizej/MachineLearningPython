from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


boston = load_boston()

plt.title("Boston Houses")
plt.xlabel('Predicted price')
plt.ylabel('Actual price')

lr = LinearRegression()

x = boston.data
y = boston.target
p = np.zeros_like(y)

kf = KFold(n_splits=5)
kf.get_n_splits(x)

#KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in kf.split(x):
    lr.fit(x[train_index], y[train_index])
    p[test_index] = lr.predict(x[test_index])

rmse_cv = np.sqrt(mean_squared_error(p, y))
print('RMSE on 5-fold CV: {:.2}'.format(rmse_cv))