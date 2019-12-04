from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score


boston = load_boston()

plt.title("Boston Houses")
plt.xlabel("Average Number of Rooms")
plt.ylabel("House Price");

plt.scatter(boston.data[:,5], boston.target, color='b')
#plt.show()

lr = LinearRegression()

x = boston.data[:,5]
y = boston.target
x = np.transpose(np.atleast_2d(x))
lr.fit(x, y)
y_predicted = lr.predict(x)

plt.plot(x, y_predicted, linewidth=2, color='r')

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