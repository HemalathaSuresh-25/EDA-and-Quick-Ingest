import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load dataset
df = pd.read_csv("C:/Users/hemalatha/Desktop/attest-eda/data/train_dataset.csv")

# Strip any leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Check available columns
print("Columns in CSV:", df.columns.tolist())

# Filter FAIL rows (case-insensitive)
fail_df = df[df["label"].str.upper() == "FAIL"].reset_index(drop=True)

# Check that the root_cause_label column exists
if 'root_cause_label' not in fail_df.columns:
    raise KeyError("Column 'root_cause_label' not found in dataset!")

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=7000, stop_words="english")
X = vectorizer.fit_transform(fail_df['error_msg'])

# Encode reason labels
le = LabelEncoder()
y = le.fit_transform(fail_df['root_cause_label'])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Classifier with class weights to handle data imbalance
clf = RandomForestClassifier(
    n_estimators=200, 
    random_state=42, 
    class_weight='balanced',  # Automatically adjust for class imbalance
    max_depth=15,             # Limit depth to reduce overfitting
    min_samples_leaf=2        # Prevent tiny leaves that can overfit
)

# Cross-validation to monitor overfitting
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Average CV accuracy: {cv_scores.mean():.4f}")

# Train model
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nConfusion Matrix:\n", pd.DataFrame(cm, index=le.classes_, columns=le.classes_))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model & vectorizer
dump(clf, "models/fail_reason_classifier.joblib")
dump(vectorizer, "models/fail_reason_vectorizer.joblib")
dump(le, "models/fail_reason_labelencoder.joblib")

# Save classes for future-proofing (handle unseen labels in future data)
dump(le.classes_, "models/fail_reason_classes.joblib")

print("\nModel training complete. Remember to monitor performance with new data to avoid overfitting.")
