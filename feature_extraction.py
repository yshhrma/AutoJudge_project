import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

OUTPUT_FEATURES = os.path.join(MODELS_DIR, 'features_and_targets.joblib')
OUTPUT_VECTORIZER = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
OUTPUT_ENCODER = os.path.join(MODELS_DIR, 'label_encoder.joblib')

def main():
    print("--- Step 2: Feature Extraction ---")

    # 1. Check prerequisites
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run preprocess.py first.")
        return
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    # 2. Load Cleaned Data
    print("Loading cleaned data...")
    df = pd.read_csv(INPUT_FILE)
    # Fill any remaining NaN in text with empty string to avoid errors
    df['clean_text'] = df['clean_text'].fillna('')

    # --- A. Text Vectorization (TF-IDF) ---
    print("Vectorizing text using TF-IDF...")
    # We limit to top 3000 features to keep the model manageable
    tfidf = TfidfVectorizer(max_features=3000)
    X_text = tfidf.fit_transform(df['clean_text']).toarray()
    print(f"Text vector shape: {X_text.shape}")

    # --- B. Combine with Manual Features ---
    print("Processing manual features (math count, text length)...")
    # Extract manual features
    # Ensure they are 2D arrays for stacking: (n_samples, 1)
    math_feat = df['math_count'].values.reshape(-1, 1)
    len_feat = df['text_len'].values.reshape(-1, 1)

    # IMPORTANT: Normalize manual features.
    # TF-IDF values are small (between 0 and 1). If text length is 5000, 
    # it will overpower the TF-IDF features unless normalized.
    # Simple division by max value brings them roughly to 0-1 range.
    math_feat_norm = math_feat / (math_feat.max() + 1e-5) # add epsilon to avoid div by zero
    len_feat_norm = len_feat / (len_feat.max() + 1e-5)

    # Stack TF-IDF features horizontally with normalized manual features
    print("Combining text features with manual features...")
    X = np.hstack((X_text, math_feat_norm, len_feat_norm))
    print(f"Final feature matrix (X) shape: {X.shape}")

    # --- C. Prepare Targets (y) ---
    print("Preparing targets...")
    # Target 1: Classification (Easy/Medium/Hard) -> Encode to 0/1/2
    le = LabelEncoder()
    y_class = le.fit_transform(df['problem_class'])
    
    # Target 2: Regression (Score)
    y_score = df['problem_score'].values

    # --- 3. Save Assets ---
    print("Saving features, targets, and vectorizers to 'models/' folder...")
    
    # Save the fitted vectorizer and encoder (needed later for the Web UI)
    joblib.dump(tfidf, OUTPUT_VECTORIZER)
    joblib.dump(le, OUTPUT_ENCODER)
    
    # Save the processed data ready for training
    data_to_save = {
        'X': X,
        'y_class': y_class,
        'y_score': y_score
    }
    joblib.dump(data_to_save, OUTPUT_FEATURES)

    print("Done! Feature extraction complete.")

if __name__ == "__main__":
    main()