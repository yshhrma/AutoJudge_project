import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import os

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, 'models')

VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.joblib')
CLF_MODEL_PATH = os.path.join(MODELS_DIR, 'best_classifier.joblib')
REG_MODEL_PATH = os.path.join(MODELS_DIR, 'best_regressor.joblib')


# --- Helper Functions ---
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def clean_text_content(text):
    """Must use SAME cleaning logic as training"""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# --- Main App ---
def main():
    st.set_page_config(page_title="AutoJudge", page_icon="⚖️")
    
    st.title("⚖️ AutoJudge System")
    st.markdown("Predict Programming Problem Difficulty & Score")

    # 1. Load Models
    # We use st.cache_resource so it doesn't reload on every click (faster)
    @st.cache_resource
    def load_models():
        if not os.path.exists(CLF_MODEL_PATH):
            return None, None, None, None
        
        vec = joblib.load(VECTORIZER_PATH)
        le = joblib.load(ENCODER_PATH)
        clf = joblib.load(CLF_MODEL_PATH)
        reg = joblib.load(REG_MODEL_PATH)
        return vec, le, clf, reg

    vectorizer, label_encoder, clf_model, reg_model = load_models()
    setup_nltk()

    if clf_model is None:
        st.error("Error: Models not found! Please run train_models.py first.")
        return

    # 2. Input Section
    col1, col2 = st.columns(2)
    with col1:
        problem_desc = st.text_area("Problem Description", height=200, placeholder="Paste the problem story here...")
    with col2:
        input_desc = st.text_area("Input Description", height=90, placeholder="Input constraints...")
        output_desc = st.text_area("Output Description", height=90, placeholder="Output format...")

    # 3. Prediction Logic
    if st.button("Predict Difficulty"):
        if not problem_desc:
            st.warning("Please enter a problem description.")
        else:
            # A. Prepare Data (Exact same steps as Feature Extraction)
            combined_text = f"{problem_desc} {input_desc} {output_desc}"
            
            # Feature 1: Math Count
            math_count = combined_text.count('$')
            
            # Feature 2: Text Length
            text_len = len(combined_text)
            
            # Feature 3: TF-IDF
            clean_txt = clean_text_content(combined_text)
            tfidf_feat = vectorizer.transform([clean_txt]).toarray()
            
            # Normalize Manual Features (Simple max approximation based on typical data)
            # In a real rigorous system, we'd save the 'max' from training. 
            # Here we use a safe large number to keep scale small (0-1).
            math_norm = math_count / 100.0 
            len_norm = text_len / 5000.0 
            
            # Stack features: [TF-IDF, Math, Length]
            # Reshape manual features to 2D array
            manual_feats = np.array([[math_norm, len_norm]])
            final_features = np.hstack((tfidf_feat, manual_feats))
            
            # B. Predict
            # Classification
            pred_class_idx = clf_model.predict(final_features)[0]
            pred_class = label_encoder.inverse_transform([pred_class_idx])[0]
            
            # Regression
            pred_score = reg_model.predict(final_features)[0]
            
            # 4. Display Results
            st.success("Analysis Complete!")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Predicted Difficulty", pred_class)
            with res_col2:
                st.metric("Predicted Score", f"{pred_score:.2f}")

if __name__ == "__main__":
    main()