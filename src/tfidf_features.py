
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "cleaned_reviews.csv"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Load cleaned data
df = pd.read_csv(DATASET_PATH)

# Select input text and target labels
X = df["text_"].fillna("")
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Convert text into TF-IDF numerical features
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the fitted vectorizer for later use
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

print("========== TF-IDF FEATURE EXTRACTION ==========")
print("Total reviews:", len(df))
print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))
print("Training feature matrix shape:", X_train_tfidf.shape)
print("Testing feature matrix shape:", X_test_tfidf.shape)
print("Number of TF-IDF features:", len(vectorizer.get_feature_names_out()))

print("\nSample TF-IDF words:")
print(vectorizer.get_feature_names_out()[:20])

print("\nTF-IDF feature extraction completed successfully.")
print("Saved vectorizer to:", MODEL_DIR / "tfidf_vectorizer.pkl")
