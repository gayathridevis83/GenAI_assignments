# can you predict the employee attrition in an organization based on the following features. The features and the dataset are given below. use a classification model with KNN algorithm
# Features:
# Age: Age of the employee (numerical).
# JobRole: The job role/position of the employee (categorical).
# MonthlyIncome: Employee's monthly salary (numerical).
# JobSatisfaction: A rating from 1 to 4 indicating the employee's satisfaction with the job (numerical).
# YearsAtCompany: Number of years the employee has been at the company (numerical).
# Attrition: Target label indicating whether the employee left the company (1 for attrition, 0 for no attrition)
 
# Age,JobRole,MonthlyIncome,JobSatisfaction,YearsAtCompany,Attrition
# 29,Sales Executive,4800,3,4,1
# 35,Research Scientist,6000,4,8,0
# 40,Laboratory Technician,3400,2,6,0
# 28,Sales Executive,4300,3,3,1
# 45,Manager,11000,4,15,0
# 25,Research Scientist,3500,1,2,1
# 50,Manager,12000,4,20,0
# 30,Sales Executive,5000,2,5,0
# 37, Laboratory Technician,3100,2,9,0
# 26, Research Scientist,4500,3,2,1

# Solution:

# Import all dependent libraries
import pandas as pd
from io import StringIO

# ColumnTransformer helps apply different preprocessing to different columns
from sklearn.compose import ColumnTransformer

# Pipeline helps combine preprocessing + model into one workflow
from sklearn.pipeline import Pipeline

# OneHotEncoder converts categorical text into numbers
# StandardScaler normalizes numerical values to similar scale
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# KNeighborsClassifier is the KNN classification algorithm
from sklearn.neighbors import KNeighborsClassifier

# LeaveOneOut is a cross-validation method for very small datasets
# cross_val_score evaluates model performance
# cross_val_predict gives prediction for each row using cross-validation
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict

# accuracy_score checks overall accuracy
# confusion_matrix shows correct/incorrect predictions
# classification_report gives precision, recall, F1-score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Store the dataset as a multiline string
csv_data = """Age,JobRole,MonthlyIncome,JobSatisfaction,YearsAtCompany,Attrition
29,Sales Executive,4800,3,4,1
35,Research Scientist,6000,4,8,0
40,Laboratory Technician,3400,2,6,0
28,Sales Executive,4300,3,3,1
45,Manager,11000,4,15,0
25,Research Scientist,3500,1,2,1
50,Manager,12000,4,20,0
30,Sales Executive,5000,2,5,0
37, Laboratory Technician,3100,2,9,0
26, Research Scientist,4500,3,2,1
"""


# Read the CSV string into a pandas DataFrame
# skipinitialspace=True removes spaces after commas while reading
df = pd.read_csv(StringIO(csv_data), skipinitialspace=True)


# Remove extra spaces from JobRole values
df["JobRole"] = df["JobRole"].str.strip()


# Select input features (X)
# These are the columns used to predict attrition
X = df[["Age", "JobRole", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany"]]

# Select target column (y)
# This is what the model tries to predict
y = df["Attrition"]


# List of categorical columns
categorical_features = ["JobRole"]

# List of numerical columns
numeric_features = ["Age", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany"]


# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        # Apply StandardScaler to numerical columns
        ("num", StandardScaler(), numeric_features),

        # Apply OneHotEncoder to categorical column JobRole
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# Create Leave-One-Out Cross Validation object
# Since the dataset is very small, we test one row at a time
loo = LeaveOneOut()

# Empty list to store accuracy results for different K values
results = []


# Try K values from 1 to 5
for k in range(1, 6):
    
    # Build a pipeline:
    # Step 1: preprocess the data
    # Step 2: apply KNN classifier with current k
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    
    # Evaluate model using Leave-One-Out Cross Validation
    # scoring="accuracy" means calculate accuracy
    scores = cross_val_score(model, X, y, cv=loo, scoring="accuracy")
    
    # Store k and its average accuracy
    results.append((k, scores.mean()))


# Convert result list into a DataFrame for easy viewing
results_df = pd.DataFrame(results, columns=["k", "loocv_accuracy"])

# Print accuracy for each k value
print("K search results:")
print(results_df)


# Sort the results by highest accuracy first
# If two k values tie, smaller k is selected
best_k = results_df.sort_values(["loocv_accuracy", "k"], ascending=[False, True]).iloc[0]["k"]

# Convert best_k from float to integer
best_k = int(best_k)

# Print best K
print(f"\nBest K = {best_k}")


# Create final model using the best K value
best_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])


# Generate cross-validated predictions for every row
# Each row is predicted using a model that was trained without that row
cv_preds = cross_val_predict(best_model, X, y, cv=loo)


# Calculate overall accuracy using actual labels and predicted labels
accuracy = accuracy_score(y, cv_preds)

# Print accuracy
print(f"\nAccuracy: {accuracy:.2f}")


# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y, cv_preds))


# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y, cv_preds, zero_division=0))


# Add predictions as a new column in the original DataFrame
df["PredictedAttrition"] = cv_preds

# Print prediction
print("\nPredictions:")
print(df)
