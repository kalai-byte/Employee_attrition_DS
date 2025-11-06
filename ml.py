
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\DELL\Downloads\Employee-Attrition - Employee-Attrition.csv')

#print(df.head())
#print(df.columns)
drop_cols=['EmployeeCount','EmployeeNumber','Over18']
df.drop(columns=drop_cols,inplace=True,errors='ignore')

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0],inplace=True)
for col in df.select_dtypes(exclude=['object']).columns:
    df[col].fillna(df[col].median(),inplace=True)

df['Attrition']=df['Attrition'].map({'Yes':1,'No':0})    
#print(df)
#print(df.describe())

#plt.figure(figsize=(10,6))
#sns.countplot(x='Attrition',data=df)
#plt.title("Attrition Distribution")
#plt.show()

x= df.drop('Attrition',axis=1)
y=df['Attrition']

categ_cols=x.select_dtypes(include=['object']).columns
numer_cols=x.select_dtypes(exclude=['object']).columns

#Pipeline
Catego_transformer= OneHotEncoder (drop='first',handle_unknown='ignore')
Numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categ_cols),
   ('num', StandardScaler(), numer_cols)

])

x_processed = preprocessor.fit_transform(x)
smote = SMOTE(random_state=42)
x_res, y_res = smote.fit_resample(x_processed, y)
print(f"Data balanced using SMOTE: {y.value_counts().to_dict()} → {pd.Series(y_res).value_counts().to_dict()}")

x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(preprocessor.fit_transform(x_train), y_train)
x_test_transformed = preprocessor.transform(x_test)

model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',
        random_state=42
    )
model.fit(x_train_res, y_train_res)

y_prob = model.predict_proba(x_test_transformed)[:, 1]
threshold = 0.35

y_pred = (y_prob >= threshold).astype(int)





print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.3f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import joblib
joblib.dump(model, "employee_attrition_model.joblib")
joblib.dump(preprocessor, "preprocessor.joblib")


loaded_model = joblib.load("employee_attrition_model.joblib")

loaded_preprocessor = joblib.load('preprocessor.joblib')

print("Model and preprocessor loaded successfully!")
#joblib.dump(model, "employee_attrition_model.joblib")

 
