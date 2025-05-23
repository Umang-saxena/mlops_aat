import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from UCI repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]
data = pd.read_csv(url, header=None, names=column_names, na_values=' ?')

# Drop rows with missing values
data = data.dropna()

# Encode target variable
data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Encode categorical features using LabelEncoder
categorical_cols = [
    'workclass', 'education', 'marital_status', 'occupation',
    'relationship', 'race', 'sex', 'native_country'
]

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Features and target
X = data.drop('income', axis=1)
y = data['income']

# Split train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)



# Model Training

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Overall Accuracy:", accuracy_score(y_test, y_pred))

# Check accuracy by sex (0 = female, 1 = male after label encoding)
sex_test = X_test['sex'].values
male_idx = sex_test == 1
female_idx = sex_test == 0

print("Male Accuracy:", accuracy_score(y_test[male_idx], y_pred[male_idx]))
print("Female Accuracy:", accuracy_score(y_test[female_idx], y_pred[female_idx]))



# Bias Mitigation
from sklearn.utils import resample

# Combine train features and labels for resampling
train_df = X_train.copy()
train_df['income'] = y_train
train_df['sex'] = X_train['sex']

# Separate male and female groups in training set
male_df = train_df[train_df['sex'] == 1]
female_df = train_df[train_df['sex'] == 0]

# Oversample female group to match male group size
female_oversampled = resample(
    female_df,
    replace=True,
    n_samples=len(male_df),
    random_state=42
)

# Combine oversampled female data with male data
balanced_train_df = pd.concat([male_df, female_oversampled])

# Shuffle the data
balanced_train_df = balanced_train_df.sample(frac=1, random_state=42)

# Separate features and labels again
X_train_balanced = balanced_train_df.drop(['income'], axis=1)
y_train_balanced = balanced_train_df['income']

# Train model again on balanced data
model_balanced = LogisticRegression(max_iter=1000)
model_balanced.fit(X_train_balanced, y_train_balanced)

y_pred_balanced = model_balanced.predict(X_test)

print("\nAfter Oversampling Females:")
print("Overall Accuracy:", accuracy_score(y_test, y_pred_balanced))
print("Male Accuracy:", accuracy_score(y_test[male_idx], y_pred_balanced[male_idx]))
print("Female Accuracy:", accuracy_score(y_test[female_idx], y_pred_balanced[female_idx]))
