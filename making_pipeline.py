

from datapreprocessing import Xdf, Ydf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import os

# Define categorical columns
categorical_columns = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']

# Mapping dictionary
dict_for_mapping = {'high': 0, 'low': 1, 'medium': 2, 'very_high': 3}

# Convert target variable
Ydf = Ydf.map(dict_for_mapping)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(Xdf, Ydf, train_size=0.7, random_state=42)

# Create a pipeline for encoding, imputation, and classification
def create_pipeline(model):
    return Pipeline([
        ("transform_encoder", ColumnTransformer(
            transformers=[("encoder", OrdinalEncoder(), categorical_columns)],
            remainder='passthrough'
        )),
        ('imputer', KNNImputer()),
        ('model', model)
    ])

# Define model paths
save_main_path = r"C:\Users\Administrator\Desktop\Intern\first_project_sklearn-main\models"

# Ensure directory exists
os.makedirs(save_main_path, exist_ok=True)

### **Random Forest Model**
rf_pipeline = create_pipeline(RandomForestClassifier())
rf_pipeline.fit(X_train, Y_train)
rf_pred = rf_pipeline.predict(X_test)
rf_score = classification_report(Y_test, rf_pred)

# Save Random Forest model
rf_model_path = os.path.join(save_main_path, "random_forest.pkl")
with open(rf_model_path, 'wb') as file:
    pickle.dump(rf_pipeline, file)

### **Support Vector Classifier (SVC)**
svc_pipeline = create_pipeline(SVC())
svc_pipeline.fit(X_train, Y_train)
svc_pred = svc_pipeline.predict(X_test)
svc_score = classification_report(Y_test, svc_pred)

# Save SVC model
svc_model_path = os.path.join(save_main_path, "svc.pkl")
with open(svc_model_path, 'wb') as file:
    pickle.dump(svc_pipeline, file)

# Create a DataFrame of predictions
df_test = pd.DataFrame({'Actual': Y_test, 'Predicted': rf_pred})

# Map labels back to original categories
labeling_map = {0: 'high', 1: 'low', 2: 'medium', 3: 'very_high'}
df_test['Actual'] = df_test['Actual'].map(labeling_map).fillna('unknown')
df_test['Predicted'] = df_test['Predicted'].map(labeling_map).fillna('unknown')

# Print scores
print("Random Forest Classification Report:\n", rf_score)
print("SVC Classification Report:\n", svc_score)
