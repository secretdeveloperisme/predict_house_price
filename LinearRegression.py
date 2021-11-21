import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#####################################################################
#                         Linear Regression                         #
#####################################################################

class LinearRegression:
    # constructor
    def __init__(self, eta=0.001, intercept_=0):
        self.coef_ = np.array([])
        self.intercept_ = intercept_
        self.eta = eta
        self.x_train = np.array([])
        self.y_train = np.array([])
    """
       x_changed = (x - mean(x))/(max(x) -  min(x))
    """
    def feature_scaling(self, features):
        x = features.copy()
        n = len(x[0, :])
        # calculate features scale
        for i in range(n):
            if np.average(x[:, i]) > 1:
                max_min = float(x[:, i].max() - x[:, i].min())
                x[:, i] = (x[:, i] - x[:, i].mean()) / max_min
        return x.copy()
    """
      y_changed = (y - mean(y))/(max(y) -  min(y))
    """
    def y_scaling(self, y):
        n = len(y)
        # calculate features scale
        for i in range(n):
            if (np.average(y)) > 10:
                max_min = float(y.max() - y.min())
                y = (y - y.mean()) / max_min
        return y.copy()

    def fit_stochastic(self, x_train, y_train):
        # copy origin values
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        # get number of datasets, number of features
        m = len(x_train)
        n = len(x_train[0, :])
        # init theta : 1 -> n
        self.coef_ = np.zeros((n), dtype=np.float64)
        print("Previous Theta:")
        print("theta0=", self.intercept_)
        print("thetaj=", self.coef_)
        # scale values features and label
        x_scaling = self.feature_scaling(x_train)
        y_scaling = self.y_scaling(y_train)
        # train model with stochastic algorithm
        while True:
            for i in range(m):
                # calculate h(x)
                h_i = np.sum(self.coef_*x_scaling[i, ]) + self.intercept_
                # get previous theta
                previous_theta0 = self.intercept_
                previous_theta = self.coef_.copy()
                # update theta
                self.intercept_ -= self.eta*(h_i - y_scaling[i]) * 1
                self.coef_ -= self.eta*(h_i - y_scaling[i]) * x_scaling[i, ]
                # calculate absolute subtract previous theta from current theta
                delta = np.abs(np.subtract(previous_theta, self.coef_))
                # filter delta wrong condition ( > 10^-10)
                filter_delta = self.coef_[np.where(delta > pow(10, -10))]
                # check convergence
                if abs(previous_theta0 - self.intercept_) < pow(10, -10) and filter_delta.size <= 0:
                    print("Current Theta:")
                    print("theta0=", self.intercept_)
                    print("thetaj=", self.coef_)
                    return
    """
        y_changed = (y - y_mean)/(y_max - y_min)
        
        y = y_changed*(y_max - y_min) + y_mean  
    """
    def predict(self, features):
        features_scale = self.feature_scaling(features)
        max_min = self.y_train.max() - self.y_train.min()
        mean = self.y_train.mean()
        y_pred = np.zeros(shape=(len(features_scale)))
        for index, x in enumerate(features_scale):
            y_pred[index] = (np.sum(x*self.coef_) + self.intercept_)
            y_pred[index] = (y_pred[index] * max_min) + mean
        return y_pred
    """
        (y_true - y_pred)^2 
    """
    def mse_score(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mse = np.average((y_true - y_pred)**2)
        return mse
    """
        r2 = 1 - ((y_true - y_pred)^2)/(y_true - y_mean)^2)
    """
    def r2_score(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
        ss_tot = ((y_true - np.average(y_true)) ** 2).sum(axis=0)
        r_squared_score = 1 - np.average(ss_res / ss_tot)
        return r_squared_score


if __name__ == '__main__':
    house_price_data = pd.read_csv("data/HousingPrices-Amsterdam-August-2021.csv", index_col=0)
    # preprocessing data
    house_price_data.dropna(inplace=True)
    house_price_data.drop(axis=1, columns="Address", inplace=True)
    house_price_data["Zip"] = house_price_data["Zip"].str.extract("(\d+)").astype(float)
    print("#"*50 + "Origin data" + "#"*50)
    print(house_price_data.head().to_string())
    print("#"*50 + "Scaling data" + "#"*50)
    print(LinearRegression().feature_scaling(np.array(house_price_data))[0:5])
    # get features and label
    Y = np.array(house_price_data["Price"])
    X = np.array(house_price_data[["Zip", "Area", "Room", "Lon", "Lat"]])
    total_accuracy = 0
    for i in range(0, 10):
        print("=" * 50 + "Loop:" + str(i) + "=" * 50)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=3 / 10.0, random_state=i+20)
        model = LinearRegression(eta=0.001, intercept_=1)
        model.fit_stochastic(np.float64(X_train), np.float64(y_train))
        y_pred = model.predict(X_test)
        mse_score = model.mse_score(y_test,y_pred)
        accuracy = np.round(model.r2_score(y_test, y_pred)*100, 3)
        total_accuracy += accuracy
        print("MSE:", mse_score)
        print("accuracy={}%".format(accuracy))
    print("="*100)
    print("average accuracy: {} %".format(np.round(total_accuracy/10, 3)))
