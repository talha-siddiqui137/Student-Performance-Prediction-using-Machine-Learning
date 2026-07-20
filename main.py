import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#  load and understand data 

df = pd.read_csv("StudentPerformanceFactors.csv")

print(df.info())
print(df.describe())

print(df.isnull().mean()*100)



# data preprocessing

# Fill missing values with mode

df["Teacher_Quality"] = df["Teacher_Quality"].fillna(
    df["Teacher_Quality"].mode()[0]
)

df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(
    df["Parental_Education_Level"].mode()[0]
)

df["Distance_from_Home"] = df["Distance_from_Home"].fillna(
    df["Distance_from_Home"].mode()[0]
)
print(df.isnull().sum())

print("Duplicate rows:", df.duplicated().sum())



# Separate Numerical and Categorical Columns

num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include=["object", "str"]).columns

# EDA Graph


# 1. Distribution of Exam Scores

plt.figure(figsize=(8,5))

plt.hist(df["Exam_Score"], bins=20, color='purple', edgecolor = 'black')

plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Exam Scores")

plt.grid(True)
plt.show()


# 2. Hours Studied vs Exam Score

plt.figure(figsize=(8,5))

plt.scatter(
    df["Hours_Studied"],
    df["Exam_Score"], color='green',alpha=0.5
)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")

plt.grid(True)
plt.show()



# 3. Attendance vs Exam Score

plt.figure(figsize=(8,5))

plt.scatter(
    df["Attendance"],
    df["Exam_Score"], color='orange',alpha=0.5
)

plt.xlabel("Attendance")
plt.ylabel("Exam Score")
plt.title("Attendance vs Exam Score")

plt.grid(True)
plt.show()

# 4. Previous Scores vs Exam Score

plt.figure(figsize=(8,5))

plt.scatter(
    df["Previous_Scores"],
    df["Exam_Score"], color='gray',alpha=0.5
)

plt.xlabel("Previous Scores")
plt.ylabel("Exam Score")
plt.title("Previous Scores vs Exam Score")

plt.grid(True)
plt.show()

# encoding 

df = pd.get_dummies(df, drop_first=True, dtype=int)



# Split X and y

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

# Feature Scaling

scaler = StandardScaler()

# Learn scaling parameters from training data
X_train = scaler.fit_transform(X_train)

# Apply the same parameters to test data
X_test = scaler.transform(X_test)

# Train Model

models = {
    "Random Forest": RandomForestRegressor(n_estimators=300,max_depth=10,min_samples_split=5,random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,max_depth=3,random_state=42),
    "Linear Regression": LinearRegression()
}


results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
# Evaluation

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, rmse, r2])


results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R² Score"]
)

print(results_df)



# 5. Actual vs Predicted Graph

plt.figure(figsize=(7,7))

plt.scatter(
    y_test,
    y_pred,alpha=0.5
)

plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")

plt.title(
    "Actual vs Predicted Exam Scores"
)

plt.grid(True)

plt.show()


# 6. Residual Plot

residuals = y_test - y_pred


plt.figure(figsize=(8,5))

plt.scatter(
    y_pred,
    residuals,color='yellow',alpha=0.5
)

plt.axhline(
    y=0
)

plt.xlabel("Predicted Score")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.grid(True)

plt.show()


# 7. Compare Models Bar Chart

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["R² Score"], color='brown',edgecolor= 'black'
)

plt.xlabel("Models")
plt.ylabel("R² Score")

plt.title("Model Performance Comparison")

plt.show()



# Load Dataset
#         ↓
# Data Understanding
# (info, describe, missing values)
#         ↓
# Data Cleaning
#         ↓
# EDA
# (histogram, scatter plots, heatmap)
#         ↓
# Encoding
#         ↓
# Split X and y
#         ↓
# Train-Test Split
#         ↓
# Scaling
#         ↓
# Train Multiple Models
#         ↓
# Compare Metrics
#         ↓
# Prediction Visualization
# (actual vs predicted, residuals)