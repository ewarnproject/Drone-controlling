import pandas as pd
import matplotlib.pyplot as plt

# Read data file
data = pd.read_csv("car_companies.csv")

# Extract columns
x = data["price"]
y = data["model"]

# Create scatter plot
plt.scatter(x, y)

# Add labels and title
plt.xlabel("price")
plt.ylabel("model")
plt.title("Price vs Model Scatter Plot")

# Show plot
plt.show()