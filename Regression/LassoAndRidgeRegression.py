from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean

def ridgeRegression(x, y):
    print("Ridge Regression (L2)")
    cross_val_scores_ridge = [] 
    alpha = []

    for i in range(1, 9):
        ridgeModel = Ridge(alpha = i * 0.25)
        #ridgeModel.fit(x_train, y_train) #is this really needed?

        scores = cross_val_score(ridgeModel, x, y, cv = 5)
        print(scores)
        avg_cross_val_score = mean(scores)*100
        cross_val_scores_ridge.append(avg_cross_val_score)
        alpha.append(i * 0.25)

    print('alpha : cross_val_score') 
    for i in range(0, len(alpha)): 
        print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i]))


def lassoRegression(x, y):
    print("Lasso Regression (L1)")
    cross_val_scores_lasso = [] 
    Lambda = []

    for i in range(1, 9): 
        lassoModel = Lasso(alpha = i * 0.25, tol = 0.0925) 
        #lassoModel.fit(X_train, y_train)  #the cross validation below will fit
        scores = cross_val_score(lassoModel, x, y, cv = 10)
        print(scores)
        avg_cross_val_score = mean(scores)*100
        cross_val_scores_lasso.append(avg_cross_val_score) 
        Lambda.append(i * 0.25)

    print('Lambda : cross_val_score') 
    for i in range(0, len(Lambda)): 
        print(str(Lambda[i])+' : '+str(cross_val_scores_lasso[i])) 



boston = load_boston()
x = boston.data
y = boston.target

ridgeRegression(x, y)
print("--------------------------------------------")
lassoRegression(x, y)
print("--------------------------------------------")


    




