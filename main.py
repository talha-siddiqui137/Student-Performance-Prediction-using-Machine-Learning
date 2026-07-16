import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


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

print(num_cols)
print(cat_cols)



# encoding 

df = pd.get_dummies(df, drop_first=True, dtype=int)

print(df.info())

print(df.head(10))



# Split X and y

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)




# Load Dataset
#       ↓
# Understand Data (.info(), .describe())
#       ↓
# Handle Missing Values
#       ↓
# Remove Duplicates
#       ↓
# Encode Categorical Features
#       ↓
# Split X and y
#       ↓
# Train-Test Split
#       ↓
# Feature Scaling
#       ↓
# Train Model
#       ↓
# Prediction
#       ↓
# Evaluation