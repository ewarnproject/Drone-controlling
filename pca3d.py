# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the automobile dataset
df = pd.read_csv("automobile_2.csv")

# Display dataset
print("First 5 Rows:")
print(df.head())

# Select only numerical columns
X = df.select_dtypes(include=[np.number])

# Fill missing values
X = X.fillna(X.mean())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 3 components
pca = PCA(n_components=3)

principal_components = pca.fit_transform(X_scaled)

# Create PCA dataframe
pca_df = pd.DataFrame(
    data=principal_components,
    columns=['PC1', 'PC2', 'PC3']
)

# Display PCA output
print("\n3D PCA Result:")
print(pca_df.head())

# Explained variance ratio
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# 3D Visualization
fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    pca_df['PC1'],
    pca_df['PC2'],
    pca_df['PC3']
)

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

ax.set_title('3D PCA of Automobile Dataset')

plt.show()