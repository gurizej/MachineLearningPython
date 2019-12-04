from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score


boston = load_boston()

plt.title("Boston Houses")
plt.xlabel('Predicted price')
plt.ylabel('Actual price')

lr = LinearRegression()

x = boston.data
y = boston.target
lr.fit(x, y)

p = lr.predict(x)
plt.scatter(p, y)
plt.plot([y.min(), y.max()], [[y.min()], [y.max()]], color='r')

#mean squared error
mse = mean_squared_error(y, lr.predict(x))
print("Mean squared error (of training data): {:.3}".format(mse))

#root mean square error
rmse = np.sqrt(mse)
print("RMSE (of training data): {:.3}".format(rmse))

# coefficient of determination
r2 = lr.score(x,y)  #Another way to do it: r2 = r2_score(y, lr.predict(x))
print("R2 shows how well this will predict the prices")
print("R2 (on training data): {:.2}".format(r2))

plt.show()