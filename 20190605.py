import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('./data/20190606/drowning/' + '80_11_1.csv')

# type = 'drifting'
type = 'drowning'

print(dataset.head())
###########################################
def bestSVMParam(X, y):
    from sklearn import svm
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import GridSearchCV
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

def svmResult(data):
    from sklearn.model_selection import KFold
    from sklearn import svm
    from sklearn.metrics import confusion_matrix

    X = np.array(dataset[dataset.columns[0:2]])
    y = np.array(dataset[dataset.columns[2]])

    # bestSVMParam(X, y)
    model_rbf = svm.SVC(kernel='rbf', gamma=10, C=10) ## 10 10
    rbf = []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("## RBF'")
        model_rbf.fit(X_train, y_train)
        rbf.append(model_rbf.score(X_test, y_test))
        print("훈련 세트 정확도: {:.2f}".format(model_rbf.score(X_train, y_train)))
        print("테스트 세트 정확도: {:.2f}".format(model_rbf.score(X_test, y_test)))

    sum = 0.0
    for item in rbf:
        sum += item
    print("rbf : " + str(sum / len(rbf)))


###########################################
def drawGraphic(data):
    plt.scatter(data.hour,
                data.pred,
                c=data[type])
    plt.xlabel('hour')
    plt.ylabel('prob')
    plt.colorbar()
    plt.show()

drawGraphic(dataset)
svmResult(dataset)