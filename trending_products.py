import pandas as pd
from datetime import datetime, timedelta
import os

# Path to your interaction data (use the fixed/cleaned file)
DATA_PATH = os.path.join("data_csv", "product_interactions_data_fixed.csv")

# Load data
df = pd.read_csv(DATA_PATH)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Define the time window (e.g., last 7 days)
now = datetime.now()
window_days = 7
recent = df[df['timestamp'] > (now - timedelta(days=window_days))]

# You can filter by interactionType if you want only purchases, views, etc.
# For example, to count only purchases:
# recent = recent[recent['interactionType'] == 'purchase']

# Count interactions per product
trending = recent['productId'].value_counts().head(10)

print("Top trending products in the last 7 days:")
for product_id, count in trending.items():
    print(f"Product: {product_id}, Interactions: {count}")

# Optionally, save to a file for your app
trending.to_csv(os.path.join("data_csv", "trending_products.csv"), header=['count']) 