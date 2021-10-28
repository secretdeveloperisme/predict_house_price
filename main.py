import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
housing_price_data = pd.read_csv("data/HousingPrices-Amsterdam-August-2021.csv")
# housing_price_data.info()

# clean data
print(housing_price_data.head().to_string())
housing_price_data.dropna(inplace=True)
# housing_price_data["Address"] = housing_price_data["Address"].str.extract("(\w|\s)+(?=,)")
housing_price_data.drop(["Unnamed: 0"], axis=1, inplace=True)
housing_price_data["Address"] = pd.factorize(housing_price_data.Address)[0] + 1
housing_price_data["Zip"] = housing_price_data["Zip"].str.extract("(\d+)")
print(housing_price_data.Zip.min())
print(housing_price_data.Zip.max())
y = housing_price_data["Price"]
housing_price_data.drop(["Price"], axis=1, inplace=True)
# housing_price_data.drop(["Zip"], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(housing_price_data, y, test_size=3/10.0, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
print("predict a test dataset: ", model.predict(X_test.head(1)))
print("real value", y_test.head(1))
# print(model.coef_)
# print(model.intercept_)
print("Score :", round(model.score(X_test, y_test)*100, ndigits=2), "%")
