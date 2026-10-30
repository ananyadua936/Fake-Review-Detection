
import pandas as pd
from pathlib import Path

# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "reviews.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "cleaned_reviews.csv"

# Load the original dataset
df = pd.read_csv(DATASET_PATH)

print("Original dataset shape:", df.shape)

# Remove the unnecessary index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Remove rows with missing review text or labels
df = df.dropna(subset=["text_", "label"])

# Remove rows with empty review text
df["text_"] = df["text_"].astype(str).str.strip()
df = df[df["text_"] != ""]

# Standardise text and labels
df["text_"] = df["text_"].str.lower()
df["label"] = df["label"].astype(str).str.strip().str.upper()

# Keep only the expected labels
df = df[df["label"].isin(["CG", "OR"])]

# Remove duplicate review records
df = df.drop_duplicates(
    subset=["text_", "category", "rating", "label"]
)

# Add a readable target column
df["review_type"] = df["label"].map({
    "CG": "Computer-generated",
    "OR": "Original"
})

# Save the cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\n========== CLEANING RESULTS ==========")
print("Cleaned dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nReview type distribution:")
print(df["review_type"].value_counts())
print("\nSaved cleaned dataset to:", OUTPUT_PATH)
print("\nFirst 5 cleaned reviews:")
print(df[["category", "rating", "label", "text_"]].head())
