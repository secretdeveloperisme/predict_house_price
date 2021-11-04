import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
house_price_data = pd.read_csv("data/HousingPrices-Amsterdam-August-2021.csv", index_col=0)
house_price_data.dropna(inplace=True)
print(house_price_data.head().to_string())

house_price_data["Address"] = pd.factorize(house_price_data.Address)[0] + 1
house_price_data["Zip"] = house_price_data["Zip"].str.extract("(\d+)")

y = house_price_data["Price"]
X = house_price_data[["Address", "Zip", "Area", "Room", "Lon", "Lat"]]
# house_price_data.drop(["Zip"], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3/10.0, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
print("predict a test dataset: ", model.predict(X_test.head(1)))
print("real value", y_test.head(1))
# print(model.coef_)
# print(model.intercept_)
print("Score :", round(model.score(X_test, y_test)*100, ndigits=2), "%")
print("r2 score: ", r2_score(y_test,  model.predict(X_test)))
