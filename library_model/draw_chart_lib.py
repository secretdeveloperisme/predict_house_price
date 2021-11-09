import matplotlib.pyplot as plt
import pandas as pd

evaluate_mse = pd.read_csv("../evaluate/evaluate_mse_lib.csv")
evaluate_score = pd.read_csv("../evaluate/evaluate_r_squared_lib.csv")

evaluate_mse.plot(kind="bar", x="Loop", width=0.8, figsize=(14, 5), ylabel="Mean Squared Error", title="Đánh Giá Theo Chỉ Số MSE")
evaluate_score.plot(kind="bar", x="Loop", width=0.8, figsize=(14, 5), ylabel="Score", title="Đánh Giá Theo Score")
plt.show()
