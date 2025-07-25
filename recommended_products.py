import pickle
import numpy as np

# Load model and mappings
with open("collaborative_filtering_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("user_id_map.pkl", "rb") as f:
    user_map = pickle.load(f)
with open("product_id_map.pkl", "rb") as f:
    item_map = pickle.load(f)

# Optionally, load the user-item matrix if you want to use it for cold-start handling
try:
    from scipy.sparse import csr_matrix
    import pandas as pd
    df = pd.read_csv("data_csv/product_interactions_data_fixed.csv")
    user_ids = df['userId'].astype('category')
    product_ids = df['productId'].astype('category')
    if 'interactionType' in df.columns:
        action_map = {'view': 1, 'like': 2, 'click': 2, 'add_to_cart': 3, 'purchase': 4}
        df['interaction'] = df['interactionType'].map(action_map).fillna(1)
    else:
        df['interaction'] = 1
    user_item_matrix = csr_matrix(
        (df['interaction'], (user_ids.cat.codes, product_ids.cat.codes))
    )
except Exception as e:
    user_item_matrix = None

# Choose user index (or userId)
user_index = 0  # Change as needed

print("User index to userId mapping:")
for idx, user_id in user_map.items():
    print(f"{idx}: {user_id}")

# Recommend products for the user
if user_item_matrix is not None:
    item_indices, scores = model.recommend(user_index, user_item_matrix[user_index], N=5)
else:
    item_indices, scores = model.recommend(user_index, np.zeros(model.item_factors.shape[0]), N=5)

print(f"Top recommendations for user: {user_map[user_index]}")
for item_idx, score in zip(item_indices, scores):
    if abs(score) < 1e6:
        print(f"Product: {item_map[item_idx]}, Score: {score:.2f}") 