import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read CSV file
data = pd.read_csv("automobile_2.csv")

# Create joint plot
sns.jointplot(x=pd.to_numeric(data["horsepower"]), y=data["price"], kind="reg", color='green')

# Show plot
plt.show()