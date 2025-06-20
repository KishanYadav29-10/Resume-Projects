import pandas as pd
import numpy as np
import re
import string
import joblib
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Text Cleaning Utility ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# --- Salary Extraction ---
def extract_avg_salary(salary):
    try:
        numbers = [float(x.replace(',', '')) for x in re.findall(r'\d+(?:\.\d+)?', salary)]
        if len(numbers) == 2:
            return (numbers[0] + numbers[1]) / 2
        elif len(numbers) == 1:
            return numbers[0]
        else:
            return np.nan
    except:
        return np.nan

# --- Predictive Categorical Imputation ---
from sklearn.ensemble import RandomForestClassifier as RFC_for_impute
def predict_and_impute_categorical(df, target_column, feature_columns):
    df = df.copy()
    for col in feature_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    known = df[df[target_column] != "unknown"]
    unknown = df[df[target_column] == "unknown"]

    if unknown.empty:
        return df

    target_le = LabelEncoder()
    y_known = target_le.fit_transform(known[target_column])

    model = RFC_for_impute(n_estimators=100, random_state=42)
    model.fit(known[feature_columns], y_known)
    y_pred = model.predict(unknown[feature_columns])
    df.loc[df[target_column] == "unknown", target_column] = target_le.inverse_transform(y_pred)

    return df

# --- Load and Prepare Dataset ---
def load_and_prepare_data(path, sample_frac=None):
    logging.info("Loading dataset...")
    df = pd.read_csv(path, low_memory=False)

    df['fraudulent'] = pd.to_numeric(df['fraudulent'], errors='coerce')
    df = df[df['fraudulent'].isin([0, 1])].astype({'fraudulent': int})

    logging.info("Cleaning text and extracting features...")
    df['job_text'] = (df['job_desc'].fillna('') + ' ' + df['skills_desc'].fillna('')).apply(clean_text)
    df['has_link'] = df['job_text'].str.contains("http|www", case=False).astype(int)
    df['has_whatsapp'] = df['job_text'].str.contains("whatsapp", case=False).astype(int)
    df['has_fee'] = df['job_text'].str.contains("fee|charge|deposit|registration", case=False).astype(int)

    categorical_columns = ['employment_type', 'industry', 'location', 'salary_range', 'department', 'benefits']
    for col in categorical_columns:
        df[col] = df[col].replace('', 'unknown').astype(str)

    logging.info("Imputing missing categorical values...")
    df = predict_and_impute_categorical(df, 'industry', ['employment_type', 'location', 'salary_range', 'department'])
    df = predict_and_impute_categorical(df, 'department', ['employment_type', 'location', 'salary_range', 'industry'])

    df['avg_salary'] = df['salary_range'].apply(lambda x: extract_avg_salary(str(x)))

    if sample_frac is not None:
        logging.info(f"Sampling {sample_frac * 100}% of data for testing...")
        df = df.sample(frac=sample_frac, random_state=42)

    logging.info(f"Final dataset shape: {df.shape}")
    return df

# --- Train Model with RandomForestClassifier + GridSearchCV ---
def train_model(df, tfidf_max_features=5000):
    logging.info("Starting model training...")

    X = df[['job_text', 'employment_type', 'industry', 'location', 'salary_range',
            'department', 'benefits', 'avg_salary', 'has_link', 'has_whatsapp', 'has_fee']]
    y = df['fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    preprocessor = ColumnTransformer(transformers=[
        ('text', TfidfVectorizer(max_features=tfidf_max_features, stop_words='english'), 'job_text'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['employment_type', 'industry', 'location', 'salary_range', 'department', 'benefits']),
        ('num', SimpleImputer(strategy='median'), ['avg_salary', 'has_link', 'has_whatsapp', 'has_fee'])
    ])

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # ✅ Larger param grid for better tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best Parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    model_path = r"C:\Users\ky321\OneDrive\Desktop\Data science and Ai\resume project\New folder\model.pkl"
    joblib.dump(best_model, model_path)
    logging.info(f"Model saved at: {model_path}")

# --- Main Function ---
def main():
    try:
        file_path = r"C:\Users\ky321\OneDrive\Desktop\Data science and Ai\resume project\New folder\combined_dataset_full.csv"
        df = load_and_prepare_data(file_path, sample_frac=None)  # Set sample_frac=0.2 for faster debugging
        train_model(df, tfidf_max_features=5000)  # You can change tfidf_max_features here
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
