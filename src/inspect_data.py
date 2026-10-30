
import pandas as pd
from pathlib import Path

# Locate the dataset from the project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "reviews.csv"

# Check whether the dataset exists
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}\n"
        "Please check that reviews.csv is inside the data folder."
    )

# Load the CSV dataset
df = pd.read_csv(DATASET_PATH)

# Display basic information
print("\n========== DATASET INFORMATION ==========")
print("Dataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\n========== UNIQUE VALUES ==========")
for column in df.columns:
    if df[column].nunique() <= 20:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))
