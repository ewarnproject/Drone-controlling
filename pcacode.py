# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load automobile dataset
# For CSV file
df = pd.read_csv("automobile_2.csv")

# If using Excel file, use this instead:
# df = pd.read_excel("automobile.xlsx")

# Display dataset
print("First 5 Rows of Dataset:")
print(df.head())

# Select only numerical columns
X = df.select_dtypes(include=[np.number])

# Fill missing values with column mean
X = X.fillna(X.mean())

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)

principal_components = pca.fit_transform(X_scaled)

# Create PCA DataFrame
pca_df = pd.DataFrame(
    principal_components,
    columns=['PC1', 'PC2']
)

# Display PCA output
print("\nPrincipal Components:")
print(pca_df.head())

# Explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Plot PCA graph
plt.figure(figsize=(8,6))

plt.scatter(
    pca_df['PC1'],
    pca_df['PC2']
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Automobile Dataset")

plt.grid(True)
plt.show()