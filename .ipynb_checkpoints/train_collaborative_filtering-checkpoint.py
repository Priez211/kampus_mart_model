import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import pickle
import os

# Path to your cleaned interaction data
DATA_PATH = os.path.join("data_csv", "product_interactions_data_fixed.csv")

# 1. Load data
df = pd.read_csv(DATA_PATH)

# 2. Prepare interaction matrix
# If you have an 'action' column, you may want to map actions to numeric values
# Example: view=1, add-to-cart=2, purchase=3
action_map = {'view': 1, 'add-to-cart': 2, 'purchase': 3}
if 'interactionType' in df.columns:
    action_map = {'view': 1, 'like': 2, 'click': 2, 'add_to_cart': 3, 'purchase': 4}
    df['interaction'] = df['interactionType'].map(action_map).fillna(1)
else:
    df['interaction'] = 1

user_ids = df['userId'].astype('category')
product_ids = df['productId'].astype('category')

# Create a sparse matrix
user_item_matrix = csr_matrix(
    (df['interaction'], (user_ids.cat.codes, product_ids.cat.codes))
)

# 3. Train ALS model
model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
model.fit(user_item_matrix)

# 4. Save the model and mappings
with open("collaborative_filtering_model.pkl", "wb") as f:
    pickle.dump(model, f)
user_map = dict(enumerate(user_ids.cat.categories))
item_map = dict(enumerate(product_ids.cat.categories))
with open("user_id_map.pkl", "wb") as f:
    pickle.dump(user_map, f)
with open("product_id_map.pkl", "wb") as f:
    pickle.dump(item_map, f)

print("Model and mappings saved!")

# 5. Example: Recommend products for a user
user_index = 0  # Change as needed
item_indices, scores = model.recommend(user_index, user_item_matrix[user_index], N=5)
print("Top recommendations for user:", user_map[user_index])
for item_idx, score in zip(item_indices, scores):
    print(f"Product: {item_map[item_idx]}, Score: {score:.2f}") 