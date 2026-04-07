"""
Resume Screening System - Complete ML Pipeline
Dataset: Uses the UpdatedResumeDataSet.csv from Kaggle
Download: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATASET
# ============================================================
# Download UpdatedResumeDataSet.csv from:
# https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nCategories ({df['Category'].nunique()}):")
print(df['Category'].value_counts())

# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================
def clean_resume(text):
    """Clean resume text by removing URLs, special chars, etc."""
    text = re.sub(r'http\S+', ' ', text)           # Remove URLs
    text = re.sub(r'RT|cc', ' ', text)              # Remove RT, cc
    text = re.sub(r'#\S+', ' ', text)               # Remove hashtags
    text = re.sub(r'@\S+', ' ', text)               # Remove mentions
    text = re.sub(r'[^\w\s]', ' ', text)            # Remove punctuation
    text = re.sub(r'\d+', ' ', text)                # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()        # Remove extra spaces
    return text.lower()

df['cleaned_resume'] = df['Resume'].apply(clean_resume)
print("\n✅ Text preprocessing complete")

# ============================================================
# 3. FEATURE EXTRACTION (TF-IDF)
# ============================================================
tfidf = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
X = tfidf.fit_transform(df['cleaned_resume'])

le = LabelEncoder()
y = le.fit_transform(df['Category'])

print(f"Feature matrix shape: {X.shape}")
print(f"Number of classes: {len(le.classes_)}")

# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# 5. MODEL TRAINING & EVALUATION
# ============================================================
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM (Linear)': LinearSVC(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Naive Bayes': MultinomialNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': acc,
        'predictions': y_pred,
        'model': model
    }
    print(f"\n{'='*50}")
    print(f"Model: {name} | Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================================================
# 6. VISUALIZATIONS
# ============================================================

# 6a. Accuracy Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] * 100 for m in model_names]
colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
bars = ax.barh(model_names, accuracies, color=colors, height=0.5)
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
ax.set_xlim(90, 100)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{acc:.1f}%', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150)
print("\n📊 Saved: accuracy_comparison.png")

# 6b. Confusion Matrix for best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_preds = results[best_model_name]['predictions']
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("📊 Saved: confusion_matrix.png")

# 6c. Category Distribution
fig, ax = plt.subplots(figsize=(12, 6))
df['Category'].value_counts().plot(kind='bar', color='#0E4D64', ax=ax)
ax.set_title('Resume Category Distribution', fontsize=16, fontweight='bold')
ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('category_distribution.png', dpi=150)
print("📊 Saved: category_distribution.png")

# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================
def predict_category(resume_text):
    """Predict the category of a resume."""
    cleaned = clean_resume(resume_text)
    features = tfidf.transform([cleaned])
    best = results[best_model_name]['model']
    prediction = best.predict(features)
    category = le.inverse_transform(prediction)[0]
    return category

# Example usage
sample_resume = """
    Experienced data scientist with 5 years in machine learning, 
    deep learning, Python, TensorFlow, and statistical analysis. 
    Built predictive models for customer churn and recommendation systems.
    Skills: Python, R, SQL, Tableau, Spark, AWS SageMaker
"""
print(f"\n🔮 Sample prediction: {predict_category(sample_resume)}")

# ============================================================
# 8. CROSS-VALIDATION
# ============================================================
print("\n📊 5-Fold Cross-Validation Results:")
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"  {name}: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print("\n✅ Pipeline complete! Best model:", best_model_name, 
      f"with {results[best_model_name]['accuracy']*100:.1f}% accuracy")
