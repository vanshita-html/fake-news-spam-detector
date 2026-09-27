import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for saving PNGs without GUI windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Train model, evaluate metrics on test split, and return evaluation dict.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model_name': model_name,
        'model_obj': model,
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict
    }

def plot_and_save_confusion_matrix(cm, labels, title, filename):
    """Save styled confusion matrix heatmap as PNG in /reports."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(title, fontsize=14, pad=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[SAVED] Confusion matrix image -> {filename}")

def train_task_pipeline(task_name, cleaned_csv_path, target_labels):
    """
    Complete pipeline for a single classification task (News or Spam):
    - Load cleaned data
    - Train/Test Split (80/20 stratified)
    - TF-IDF Vectorization
    - Train & Evaluate Logistic Regression and Multinomial Naive Bayes
    - Save models, vectorizer, and evaluation reports
    """
    print(f"\n==================================================")
    print(f"  TRAINING PIPELINE FOR TASK: {task_name.upper()}")
    print(f"==================================================")
    
    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(f"Cleaned dataset missing at {cleaned_csv_path}. Run data_prep.py first.")
        
    df = pd.read_csv(cleaned_csv_path)
    df = df.dropna(subset=['cleaned_text', 'label'])
    
    X = df['cleaned_text'].values
    y = df['label'].values.astype(int)
    
    # Train / Test split (80/20 stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"[INFO] Dataset loaded: {len(df)} total samples.")
    print(f"[INFO] Split sizes -> Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    print(f"[INFO] TF-IDF Matrix shape: Train {X_train.shape}, Test {X_test.shape}")
    
    # 1. Primary Model: Logistic Regression
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_eval = evaluate_model(lr_model, X_train, y_train, X_test, y_test, model_name="Logistic Regression")
    
    # 2. Baseline Model: Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_eval = evaluate_model(nb_model, X_train, y_train, X_test, y_test, model_name="Multinomial Naive Bayes")
    
    # Print Comparison Table
    metrics_df = pd.DataFrame([
        {
            'Model': lr_eval['model_name'],
            'Accuracy': f"{lr_eval['accuracy']*100:.2f}%",
            'Precision': f"{lr_eval['precision']*100:.2f}%",
            'Recall': f"{lr_eval['recall']*100:.2f}%",
            'F1-Score': f"{lr_eval['f1_score']*100:.2f}%"
        },
        {
            'Model': nb_eval['model_name'],
            'Accuracy': f"{nb_eval['accuracy']*100:.2f}%",
            'Precision': f"{nb_eval['precision']*100:.2f}%",
            'Recall': f"{nb_eval['recall']*100:.2f}%",
            'F1-Score': f"{nb_eval['f1_score']*100:.2f}%"
        }
    ])
    
    print(f"\n--- {task_name.upper()} MODEL COMPARISON TABLE ---")
    print(metrics_df.to_string(index=False))
    
    # Select Best Model based on F1-Score
    best_eval = lr_eval if lr_eval['f1_score'] >= nb_eval['f1_score'] else nb_eval
    print(f"\n[BEST MODEL] Best Performing Model for {task_name.upper()}: {best_eval['model_name']} (F1: {best_eval['f1_score']*100:.2f}%)")
    
    # Save Best Model & Vectorizer
    model_filename = os.path.join("models", f"{task_name.lower()}_model.pkl")
    vectorizer_filename = os.path.join("models", f"{task_name.lower()}_vectorizer.pkl")
    
    joblib.dump(lr_eval['model_obj'], model_filename)  # Logistic Regression is primary for explainability
    joblib.dump(vectorizer, vectorizer_filename)
    
    print(f"[SAVED] Primary Model -> {model_filename}")
    print(f"[SAVED] Vectorizer    -> {vectorizer_filename}")
    
    # Plot & save Confusion Matrix for Logistic Regression
    cm_path = os.path.join("reports", f"{task_name.lower()}_confusion_matrix.png")
    plot_and_save_confusion_matrix(
        np.array(lr_eval['confusion_matrix']),
        labels=target_labels,
        title=f"Confusion Matrix: {task_name.title()} ({lr_eval['model_name']})",
        filename=cm_path
    )
    
    # Save json metrics report
    report_data = {
        'task': task_name,
        'logistic_regression': {
            'accuracy': lr_eval['accuracy'],
            'precision': lr_eval['precision'],
            'recall': lr_eval['recall'],
            'f1_score': lr_eval['f1_score'],
            'confusion_matrix': lr_eval['confusion_matrix']
        },
        'naive_bayes': {
            'accuracy': nb_eval['accuracy'],
            'precision': nb_eval['precision'],
            'recall': nb_eval['recall'],
            'f1_score': nb_eval['f1_score'],
            'confusion_matrix': nb_eval['confusion_matrix']
        },
        'labels': target_labels
    }
    
    json_path = os.path.join("reports", f"{task_name.lower()}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    print(f"[SAVED] Metrics JSON -> {json_path}")
    
    return report_data

def run_all_training():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    news_report = train_task_pipeline("news", os.path.join("data", "cleaned_news.csv"), target_labels=["Real", "Fake"])
    spam_report = train_task_pipeline("spam", os.path.join("data", "cleaned_spam.csv"), target_labels=["Ham", "Spam"])
    
    summary = {
        'news': news_report,
        'spam': spam_report
    }
    
    summary_path = os.path.join("reports", "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\n[COMPLETE] All models successfully trained and evaluated! Summary saved to {summary_path}")


if __name__ == "__main__":
    run_all_training()
