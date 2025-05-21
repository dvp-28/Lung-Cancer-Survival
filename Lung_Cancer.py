# Import datasets & library
import numpy as np
import pandas as pd
import sklearn

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Using Model 
from sklearn.ensemble import RandomForestClassifier as RFC

# Using Model Evaluation
from sklearn.metrics import *

# Read dataset 
df = pd.read_csv("E:\\Unified Mentor\\Projects\\Lung Cancer\\dataset_med.csv")

print("Columns Name")
print(df.columns)
print("_____________________________________")
print("Details about dataset")
print(df.dtypes)
print("_____________________________________")
print("Show missing values")
print(df.isnull().sum())

# Drop unnecessary columns
clear = df.drop(columns=["id", "diagnosis_date", "end_treatment_date"])

# Define Features & Target values
x = clear.drop("survived", axis=1)
y = clear["survived"]

# LabelEncoder
label = {}
for col in x.select_dtypes(include="object").columns:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    label[col] = le

print("Data-Preprocessing Completed")
print("_____________________________________")

# Train-Test data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.33)

# Train Model
model = RFC(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Real-time prediction (User_input)
def predict_survival(): 
    print("Enter patient details :--- ")
    age = float(input("Age : "))
    gender = input("Gender (Male/Female) : ").strip().title()
    country = input("Country : ").strip().title()
    
    valid_stages = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    cancer_stage = input("Cancer Stage (Stage I/Stage II/Stage III/Stage IV): ").strip()

    if cancer_stage not in valid_stages:
        print("Invalid input for cancer_stage. Please choose from:", valid_stages)
        return
    family_history = input("Family History (Yes/No) : ").strip().title()
    
    #smoking_status = input("Smoking Status (Smoker / Non-smoker / Passive Smoker / Former Smoker): ").strip().title()
    smoking_status = input("Smoking Status (Current Smoker / Never Smoked / Passive Smoker / Former Smoker): ").strip().title()
    valid_smoking_statuses = ['Current Smoker', 'Never Smoked', 'Passive Smoker', 'Former Smoker']

    if smoking_status not in valid_smoking_statuses:
        print("Invalid input for Smoking Status. Please choose from:", valid_smoking_statuses)
        return
    
    bmi = float(input("BMI : "))
    cholesterol = int(input("Cholesterol Level : "))
    
    def yes_no_input(prompt):
        value = input(prompt).strip().lower()
        if value in ['yes', 'y', '1']:
            return 1
        elif value in ['no', 'n', '0']:
            return 0
        else:
            print("Invalid input. Please enter Yes or No.")
            return yes_no_input(prompt)
        
    print("1=Yes & 0=No ")
    hypertension = yes_no_input("Hypertension (Yes/No) : ")
    asthma = yes_no_input("Asthma (Yes/No) : ")
    cirrhosis = yes_no_input("Cirrhosis (Yes/No) : ")
    other_cancer = yes_no_input("Other cancer (Yes/No) : ")
    treatment_type = input("Type (Surgery/Chemotherapy/Radiation/Combined) : ")

    # Create a input DataFrame
    data = pd.DataFrame([[
    age, gender, country, cancer_stage, family_history, smoking_status,
    bmi, cholesterol, hypertension, asthma, cirrhosis, other_cancer, treatment_type]],
    columns=["age", "gender", "country", "cancer_stage", "family_history", "smoking_status",
             "bmi", "cholesterol_level", "hypertension", "asthma", "cirrhosis", "other_cancer", "treatment_type"])
    
    # Encode categorical fields
    for col in data.select_dtypes(include="object").columns:
        le = label[col]
        try:
            data[col] = le.transform(data[col])
        except ValueError:
            print(f"Error: Unknown category in '{col}'. Please enter a valid value.")
            return
    
    # Prediction
    pred = model.predict(data)[0]
    report = "Patient is Survive" if pred==1 else "Patient is Not Survive"
    print(report)
    
    return pred

# Run prediction function
pred = predict_survival()

# Evaluate model
if isinstance(pred, (int, np.integer)):
    print("Model Evaluation ( Waiting ):")
    print("Accuracy Score:", accuracy_score(y_test, model.predict(x_test)))
    print("Classification Report:\n", classification_report(y_test, model.predict(x_test)))
