
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "cleaned_reviews.csv"

# Load cleaned dataset
df = pd.read_csv(DATASET_PATH)

# Set graph style
sns.set_theme(style="whitegrid")

print("========== EXPLORATORY DATA ANALYSIS ==========")
print("Dataset shape:", df.shape)

print("\nReview type counts:")
print(df["review_type"].value_counts())

print("\nRating counts:")
print(df["rating"].value_counts().sort_index())

print("\nCategory counts:")
print(df["category"].value_counts())

# Graph 1: Review type distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="review_type", hue="review_type", legend=False)
plt.title("Distribution of Review Types")
plt.xlabel("Review Type")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "review_type_distribution.png")
plt.show()

# Graph 2: Rating distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="rating", hue="rating", legend=False)
plt.title("Distribution of Review Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "rating_distribution.png")
plt.show()

# Graph 3: Review types across ratings
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="rating", hue="review_type")
plt.title("Review Types Across Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.legend(title="Review Type")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "review_types_by_rating.png")
plt.show()

# Graph 4: Top 10 product categories
top_categories = df["category"].value_counts().head(10).index
category_df = df[df["category"].isin(top_categories)]

plt.figure(figsize=(12, 7))
sns.countplot(
    data=category_df,
    y="category",
    order=top_categories,
    hue="category",
    legend=False
)
plt.title("Reviews by Product Category")
plt.xlabel("Number of Reviews")
plt.ylabel("Product Category")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "reviews_by_category.png")
plt.show()

print("\nEDA completed successfully.")
print("Four graphs have been saved in the project root folder.")
