import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, 'data', 'problems_data.jsonl')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')

def setup_nltk():
    """Ensure NLTK stopwords are downloaded."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

def clean_text_content(text):
    """
    Cleans text by:
    1. Converting to lowercase
    2. Removing special characters (keeping only words and spaces)
    3. Removing common stopwords (the, a, is, etc.)
    """
    text = str(text).lower()
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    
    return " ".join(words)

def main():
    print("--- Step 1: Data Preprocessing ---")
    
    # 1. Check if data exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        print("Please create a 'data' folder and put 'problems_data.jsonl' inside it.")
        return

    # 2. Load Data
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE, lines=True)
    print(f"Loaded {len(df)} problems.")

    # 3. Feature Engineering (Part 1: Extraction)
    print("Extracting features...")
    
    # Combine all text fields into one for better context
    df['combined_text'] = (
        df['title'].fillna('') + " " + 
        df['description'].fillna('') + " " + 
        df['input_description'].fillna('') + " " + 
        df['output_description'].fillna('')
    )

    # Count '$' symbols (proxy for mathematical complexity)
    df['math_count'] = df['combined_text'].apply(lambda x: x.count('$'))
    
    # Calculate text length
    df['text_len'] = df['combined_text'].apply(len)

    # 4. Text Cleaning
    print("Cleaning text (this may take a moment)...")
    setup_nltk()
    df['clean_text'] = df['combined_text'].apply(clean_text_content)

    # 5. Save Results
    print(f"Saving cleaned data to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done! Preprocessing complete.")

if __name__ == "__main__":
    main()

