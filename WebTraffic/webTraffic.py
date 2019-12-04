import scipy as sp
import matplotlib.pyplot as plt

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")

#This error will be calculated as the squared distance of the model's prediction to the real data
def error(f, x, y):
    return sp.sum((f(x)-y)**2)

def plotThis(x,y):
    #Plotting our data:
    # plot the (x,y) points with dots of size 10
    plt.scatter(x, y, s=10)
    plt.title("Web traffic over the last month (Polynomial degree fit)")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks([w*7*24 for w in range(10)], ['week %i' % w for w in range(10)])
    plt.autoscale(tight=True)
    # draw a slightly opaque, dashed grid
    plt.grid(True, linestyle='-', color='0.75')
    

#Assign column 0 to x, and column 1 to y
x = data[:,0]
y = data[:,1]

#Remove the NaN rows
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

#Fitting polynomial of degree one
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
print("Model parameters: %s" % fp1)
print(residuals)

f1 = sp.poly1d(fp1)
print(error(f1, x, y))

#plotThis(x,y)
fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
plotThis(x,y)
plt.plot(fx, f1(fx), linewidth=4)
plotHandle1 = ["poly d=%i" % f1.order]
plt.legend(plotHandle1, loc="upper left")

#Fitting polynomial of degree two
f2p = sp.polyfit(x, y, 2)
f2 = sp.poly1d(f2p)
plt.plot(fx, f2(fx), linewidth=4)
plt.legend(["poly d=%i" % f1.order, "d=%i" % f2.order], loc="upper left")

f10p = sp.polyfit(x, y, 10)
f10 = sp.poly1d(f10p)
plt.plot(fx, f10(fx), linewidth=4)
plt.legend(["poly d=%i" % f1.order, "poly d=%i" % f2.order, "poly d=%i" % f10.order], loc="upper left")

#This is what overfitting looks like, there is way too much oscilating in the curve to be able to predict anything
f53p = sp.polyfit(x, y, 53)
f53 = sp.poly1d(f53p)
plt.plot(fx, f53(fx), linewidth=4)
plt.legend(["poly d=%i" % f1.order, "poly d=%i" % f2.order, "poly d=%i" % f10.order, "poly d=%i" % f53.order], loc="upper left")

plt.show()

