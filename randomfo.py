import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv(
    r"C:\Users\RavikiranNair\Downloads\archive\MiningProcess_Flotation_Plant_Database.csv",
    sep=",",
    decimal=","
)
df = df.sample(n=50000, random_state=42)

df = df.drop(columns=["date"])
X = df.drop(["% Silica Concentrate"],axis=1)
y = df["% Silica Concentrate"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#random forest
model = RandomForestRegressor(
    n_estimators=10,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
joblib.dump(model, "random_forest_model.pkl")

#Linear regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

mae = mean_absolute_error(y_test, linear_predictions)
mse = mean_squared_error(y_test, linear_predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, linear_predictions)

print("Linear Regression")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 :", r2)

#Decision tree


dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

mae = mean_absolute_error(y_test, dt_predictions)
mse = mean_squared_error(y_test, dt_predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, dt_predictions)

print("Decision Tree")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 :", r2)

correlation = df.corr()

plt.figure(figsize=(15, 10))
plt.imshow(correlation, cmap="coolwarm", aspect="auto")
plt.colorbar()

plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()



# Random Forest Plot

plt.figure(figsize=(6,5))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression")
plt.grid(True)
plt.show()


# Linear Regression Plot
plt.figure(figsize=(6,5))
plt.scatter(y_test, linear_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression")
plt.grid(True)
plt.show()



# Decision Tree Plot

plt.figure(figsize=(6,5))
plt.scatter(y_test, dt_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Decision Tree Regression")
plt.grid(True)
plt.show()



#Correlation Matrix

correlation = df.corr()

plt.figure(figsize=(12,10))
plt.imshow(correlation, cmap="coolwarm", interpolation="nearest")
plt.colorbar()

plt.xticks(range(len(correlation.columns)),correlation.columns,rotation=90)

plt.yticks(range(len(correlation.columns)),correlation.columns)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()



# Compare All Models (R² Score)
models = ["Random Forest", "Linear Regression", "Decision Tree"]
scores = [
    r2_score(y_test, predictions),
    r2_score(y_test, linear_predictions),
    r2_score(y_test, dt_predictions)
]

plt.figure(figsize=(7,5))
plt.bar(models, scores)

plt.xlabel("Models")
plt.ylabel("R² Score")
plt.title("Model Comparison")

for i, score in enumerate(scores):
    plt.text(i, score, f"{score:.3f}", ha="center")

plt.show()


# Actual vs Predicted Comparison

plt.figure(figsize=(12,6))

plt.plot(y_test.values[:100], label="Actual")
plt.plot(predictions[:100], label="Random Forest")
plt.plot(linear_predictions[:100], label="Linear Regression")
plt.plot(dt_predictions[:100], label="Decision Tree")

plt.xlabel("Sample Number")
plt.ylabel("% Silica Concentrate")
plt.title("Actual vs Predicted Comparison")
plt.legend()
plt.grid(True)

plt.show()