import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --- Configuration ---
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, 'models')
INPUT_FEATURES = os.path.join(MODELS_DIR, 'features_and_targets.joblib')
BEST_CLF_PATH = os.path.join(MODELS_DIR, 'best_classifier.joblib')
BEST_REG_PATH = os.path.join(MODELS_DIR, 'best_regressor.joblib')


def train_classification(X_train, X_test, y_train, y_test):
    print("\n=== Training Classification Models (Easy/Medium/Hard) ===")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear') # Linear kernel often works well for text
    }
    
    best_model = None
    best_acc = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"  -> Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\n🏆 Best Classification Model: {best_name} (Accuracy: {best_acc:.4f})")
    
    # Detailed Evaluation for the winner
    print("\n--- Detailed Classification Report ---")
    y_pred_final = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_final, target_names=['Easy', 'Hard', 'Medium'])) # Note: Check label encoder order if needed
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_final))
    
    return best_model

def train_regression(X_train, X_test, y_train, y_test):
    print("\n=== Training Regression Models (Predicting Score) ===")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    
    best_model = None
    best_rmse = float('inf') # Lower is better
    best_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        print(f"  -> RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name
            
    print(f"\n🏆 Best Regression Model: {best_name} (RMSE: {best_rmse:.4f})")
    return best_model

def main():
    print("--- Step 3 & 4: Model Training and Evaluation ---")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FEATURES):
        print(f"Error: {INPUT_FEATURES} not found. Run feature_extraction.py first.")
        return
        
    print("Loading features...")
    data = joblib.load(INPUT_FEATURES)
    X = data['X']
    y_class = data['y_class']
    y_score = data['y_score']
    
    # 2. Split Data (80% Train, 20% Test)
    # We split once so both tasks use the same test set for consistency
    X_train, X_test, y_c_train, y_c_test, y_s_train, y_s_test = train_test_split(
        X, y_class, y_score, test_size=0.2, random_state=42
    )
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")

    # 3. Train Classification
    best_clf = train_classification(X_train, X_test, y_c_train, y_c_test)
    
    # 4. Train Regression
    best_reg = train_regression(X_train, X_test, y_s_train, y_s_test)
    
    # 5. Save the winners
    print("\nSaving best models...")
    joblib.dump(best_clf, BEST_CLF_PATH)
    joblib.dump(best_reg, BEST_REG_PATH)
    print(f"Saved to {MODELS_DIR}/")
    print("Done! Ready for Web Interface.")

if __name__ == "__main__":
    main()