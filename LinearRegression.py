import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class LinearRegression:
    def __init__(self):
        self.coef_ = np.array([])
        self.intercept_ = 0.001
        self.eta = 0.001
        self.x = np.array([])
        self.y = np.array([])

    def feature_scaling(self, features):
        x = features.copy()
        n = len(x[0, :])
        for i in range(n):
            if x[0, i] > 1:
                max_min = float(x[:, i].max() - x[:, i].min())
                x[:, i] = (x[:, i] - x[:, i].mean()) / max_min
        return x.copy()


    def y_scaling(self, y):
        n = len(y)
        for i in range(n):
            if y[0] > 100:
                max_min = float(y.max() - y.min())
                y = (y - y.mean()) / max_min
        return y.copy()

    def fit_stochastic(self, x, y):
        self.x = x.copy()
        self.y = y.copy()
        m = len(x)
        n = len(x[0, :])
        self.coef_ = np.zeros((1, 5))
        print("theta0=", self.intercept_)
        print("thetaj=", self.coef_)
        x_scaling = self.feature_scaling(x)
        y_scaling = self.y_scaling(y)
        while True:
            for i in range(m):
                h_i = np.sum(self.coef_*x_scaling[i, ]) + self.intercept_
                previous_theta0 = self.intercept_
                previous_theta = self.coef_.copy()
                self.intercept_ -= self.eta*(h_i - y_scaling[i]) * 1
                self.coef_ -= self.eta*(h_i - y_scaling[i]) * x_scaling[i, ]
                delta = np.abs(np.subtract(previous_theta, self.coef_))
                filter_delta = self.coef_[np.where(delta > pow(10, -10))]
                if abs(previous_theta0 - self.intercept_) < pow(10, -10) and filter_delta.size <= 0:
                    print("theta0=", self.intercept_)
                    print("thetaj=", self.coef_)
                    return

    # def fit_batch(self, x, y):
    #     m = len(x)
    #     n = len(x[0, :])
    #     total0 = 0
    #     total1 = 0
    #     self.coef_ = np.random.randn(n)
    #     x = self.feature_scaling(x)
    #     while True:
    #         for i in range(m):
    #             h_i = self.intercept_ + np.sum(self.coef_ * x[i, ])
    #             total0 += y[i] - h_i * 1
    #             total1 += (y[i] - h_i) * x[i, ]
    #         previous_theta0 = self.intercept_
    #         previous_theta = self.coef_.copy()
    #         self.intercept_ += self.eta * total0
    #         self.coef_ += self.eta * total1
    #         if abs(previous_theta0 - self.intercept_) < pow(10, -3) and np.abs(previous_theta - self.coef_) < np.full((1,5), pow(10,-3)):
    #             break

    def fit_stochastic1(self, x, y):
        m = len(x)
        n = len(x[0, :])
        self.coef_ = np.array([-1.752, -0.25, -2.04, -0.703, 0.441])
        print("theta0", self.intercept_)
        print("theta", self.coef_)
        for i in range(2):
            h_i = round(np.sum(self.coef_ * x[i,]) + self.intercept_, 3)
            # previous = self.intercept_
            pre_theta0 = self.intercept_
            pre_theta = np.round(self.coef_.copy(),3)
            self.intercept_ -= np.round(self.eta * (h_i - y[i]) * 1, 3)
            self.coef_ -= np.round(self.eta * (h_i - y[i]) * x[i,], 3)
            print("h(x) = {theta0} + {x1}*{theta1} + {x2}*{theta2} + {x3}*{theta3} + {x4}*{theta4} + {x5}*{theta5} = {hi}".format(
                theta0=pre_theta0,
                x1=x[i, 0],theta1=pre_theta[0], x2=x[i, 1],theta2=pre_theta[1],
                x3=x[i, 2], theta3=pre_theta[2], x4=x[i, 3],theta4=pre_theta[3],
                x5=x[i, 4], theta5=pre_theta[4], hi=h_i
            ))
            print("theta0 = {theta0} - {eta}*({hi} - {yi})*{xi} = {new_theta}".format(theta0=pre_theta0,
                                                                                      eta=self.eta, hi=h_i,
                                                                                      yi=y[i], xi=1, new_theta=self.intercept_))
            for j in range(n):
                print("theta{i} = {thetai} - {eta}*({hi} - {yi})*{xi} = {new_thetai}".format(i=j+1, eta=self.eta,
                                                                                         thetai=pre_theta[j],
                                                                                         yi=y[i], xi=x[i, j],
                                                                                         hi=h_i,
                                                                                         new_thetai=self.coef_[j]))

            print(self.intercept_)
            print(self.coef_)
            print("=" * 50)

    def predict(self, features):
        # print(features)
        features_scale = self.feature_scaling(features)
        max_min = self.y.max() - self.y.min()
        mean = self.y.mean()
        y_pred = np.zeros(shape=(len(features_scale), 1))
        for index, x in enumerate(features_scale):
            y_pred[index] = (np.sum(x*self.coef_) + self.intercept_)
            y_pred[index] = (y_pred[index] * max_min) + mean
        return y_pred.reshape(-1)

    def mse_score(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mse = np.average((y_true - y_pred)**2)
        return mse

    def r2_score(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
        ss_tot = ((y_true - np.average(y_true)) ** 2).sum(axis=0)
        r_squared_score = 1 - np.average(ss_res / ss_tot)
        return r_squared_score


if __name__ == '__main__':
    house_price_data = pd.read_csv("data/HousingPrices-Amsterdam-August-2021.csv", index_col=0)
    house_price_data.dropna(inplace=True)
    house_price_data["Address"] = pd.factorize(house_price_data.Address)[0] + 1
    house_price_data["Zip"] = house_price_data["Zip"].str.extract("(\d+)").astype(float)
    house_price_data.astype(float)
    Y = np.array(house_price_data["Price"])
    X = np.array(house_price_data[["Zip", "Area", "Room", "Lon", "Lat"]])
    # print(house_price_data.head().to_string())
    for i in range(3):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=3 / 10.0, random_state=i)
        model = LinearRegression()
        model.fit_stochastic(np.float64(X_train), np.float64(y_train))
        y_pred = model.predict(X_test)
        print(X_test,y_test)
        print(r2_score(y_test, y_pred))
        print("accuracy={}%".format(np.round(model.r2_score(y_test, y_pred)*100, 3)))
        print("="*50)