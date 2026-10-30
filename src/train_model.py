
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "cleaned_reviews.csv"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATASET_PATH)

X = df["text_"].fillna("")
y = df["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create and fit TF-IDF using training data only
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define two machine learning models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = []

print("========== MODEL TRAINING AND EVALUATION ==========")

for model_name, model in models.items():
    print(f"\n{'=' * 15} {model_name} {'=' * 15}")

    # Train model
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    results.append({
        "Model": model_name,
        "Accuracy": accuracy
    })

    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["CG", "OR"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["CG", "OR"],
        yticklabels=["CG", "OR"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")
    plt.savefig(PROJECT_DIR / f"confusion_matrix_{filename}.png")
    plt.show()

    # Save trained model
    joblib.dump(model, MODEL_DIR / f"{filename}_model.pkl")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

# Display comparison
results_df = pd.DataFrame(results)
print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(PROJECT_DIR / "model_comparison.csv", index=False)

print("\nTraining and evaluation completed successfully.")
print("Saved trained models in:", MODEL_DIR)
