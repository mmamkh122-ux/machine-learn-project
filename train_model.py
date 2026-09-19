"""
train_model.py
----------------
Trains the regression + classification pipelines from the
Medical_Cost_&_Risk_Analysis_System notebook and saves them with joblib
so the Streamlit app can load them instead of retraining on every run.

Because the notebook's pipelines include the ColumnTransformer
(OneHotEncoder + StandardScaler) as their first step, the saved pipelines
accept RAW input (e.g. sex='male', region='northeast') directly — no
manual encoding needed in the app.

Usage:
    python train_model.py
(expects insurance.csv in the same folder)
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# ============================================================
# Load & clean
# ============================================================

df = pd.read_csv('insurance.csv')
df = df.drop_duplicates()

# ============================================================
# PHASE 1: Target & Feature Selection
# ============================================================

median_charge = df['charges'].median()
df['high_risk'] = (df['charges'] > median_charge).astype(int)

target_column_class = 'high_risk'
target_column_reg = 'charges'

X = df.drop(columns=[target_column_class, target_column_reg])
y_class = df[target_column_class]
y_reg = df[target_column_reg]

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

numerical_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])


# ============================================================
# PHASE 3: Model Training
# ============================================================

lin_reg = LinearRegression()
pipe_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lin_reg)
])
pipe_reg.fit(X_train, y_reg_train)

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

trained_models = {}
train_results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipe.fit(X_train, y_class_train)
    trained_models[name] = pipe
    train_results[name] = pipe.predict(X_test)


# ============================================================
# PHASE 4: Model Evaluation
# ============================================================

y_pred_reg = pipe_reg.predict(X_test)
reg_mse = mean_squared_error(y_reg_test, y_pred_reg)
reg_r2 = r2_score(y_reg_test, y_pred_reg)

evaluation_metrics = []
for name, preds in train_results.items():
    evaluation_metrics.append({
        'Model': name,
        'Accuracy': accuracy_score(y_class_test, preds),
        'Precision': precision_score(y_class_test, preds, zero_division=0),
        'Recall': recall_score(y_class_test, preds, zero_division=0),
        'F1-Score': f1_score(y_class_test, preds, zero_division=0),
    })

eval_df = pd.DataFrame(evaluation_metrics)

cv_results = {}
for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    cv_scores = cross_val_score(pipe, X_train, y_class_train, cv=5, scoring='accuracy')
    cv_results[name] = cv_scores.mean()

eval_df['CV Mean Accuracy'] = eval_df['Model'].map(cv_results)


# ============================================================
# PHASE 5: Model Selection
# ============================================================

best_model_row = eval_df.loc[eval_df['Accuracy'].idxmax()]
best_model_name = best_model_row['Model']
best_model_instance = trained_models[best_model_name]

print("=" * 60)
print("       Medical Cost & Risk Prediction — Training Run")
print("=" * 60)
print("\n--- Linear Regression (Cost Prediction) ---")
print(f"MSE: {reg_mse:.2f} | R2: {reg_r2 * 100:.2f}%")
print("\n--- Classification Models ---")
print(eval_df.to_string(index=False))
print(f"\nBest model selected: {best_model_name}")


# ============================================================
# SAVE ARTIFACTS FOR THE STREAMLIT APP
# ============================================================
# pipe_reg and each classification pipeline already include the
# preprocessor, so they accept raw dataframes (sex='male', region=...)
# directly — nothing else needs saving.

joblib.dump(pipe_reg, 'pipe_reg.pkl')
joblib.dump(trained_models, 'classification_models.pkl')
joblib.dump({
    'best_model_name': best_model_name,
    'eval_df': eval_df,
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
}, 'model_metadata.pkl')

print("\nSaved: pipe_reg.pkl, classification_models.pkl, model_metadata.pkl")
print("You can now run: streamlit run app.py")
