"""
TASK 4: SALES PREDICTION USING PYTHON
---------------------------------------
Predicts product sales based on advertising spend (TV, Radio, Newspaper)
using Linear Regression, with full graphical analysis.

Dataset expected columns: TV, Radio, Newspaper, Sales
(this matches the standard "Advertising.csv" dataset)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")

# ----------------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------------
df = pd.read_csv("advertising.csv")

# Drop stray index column some versions of this dataset include
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nSummary stats:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# ----------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (graphs)
# ----------------------------------------------------------------

# 2a. Scatter plots: each ad channel vs Sales
sns.pairplot(df, x_vars=["TV", "Radio", "Newspaper"], y_vars="Sales",
             height=4, kind="scatter")
plt.suptitle("Advertising Spend vs Sales", y=1.03)
plt.savefig("1_eda_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# 2b. Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="Purples", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("2_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# 2c. Distribution of Sales
plt.figure(figsize=(6, 5))
sns.histplot(df["Sales"], kde=True, color="purple")
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.savefig("3_sales_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ----------------------------------------------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------
# 4. TRAIN LINEAR REGRESSION MODEL
# ----------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Intercept:", model.intercept_)
print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")

# ----------------------------------------------------------------
# 5. EVALUATE MODEL
# ----------------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error : {mae:.3f}")
print(f"Mean Squared Error  : {mse:.3f}")
print(f"Root Mean Sq. Error : {rmse:.3f}")
print(f"R2 Score            : {r2:.4f}")

# ----------------------------------------------------------------
# 6. GRAPH: Actual vs Predicted Sales
# ----------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, color="purple", alpha=0.7, edgecolor="k")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="Perfect Prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales (R² = {r2:.3f})")
plt.legend()
plt.savefig("4_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------
# 7. GRAPH: Regression line for TV spend vs Sales (strongest driver)
# ----------------------------------------------------------------
plt.figure(figsize=(7, 6))
sns.regplot(x="TV", y="Sales", data=df, line_kws={"color": "red"},
            scatter_kws={"alpha": 0.6, "color": "purple"})
plt.title("TV Advertising Spend vs Sales (with Regression Line)")
plt.savefig("5_tv_vs_sales_regression.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------
# 8. GRAPH: Residual plot (model diagnostic)
# ----------------------------------------------------------------
residuals = y_test - y_pred
plt.figure(figsize=(7, 6))
plt.scatter(y_pred, residuals, color="purple", alpha=0.7, edgecolor="k")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("6_residual_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------
# 9. PREDICT SALES FOR NEW AD BUDGET
# ----------------------------------------------------------------
new_budget = pd.DataFrame({"TV": [150], "Radio": [25], "Newspaper": [10]})
predicted_sales = model.predict(new_budget)
print(f"\nPredicted Sales for TV=150, Radio=25, Newspaper=10: {predicted_sales[0]:.2f}")
