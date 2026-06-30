import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read CSV file
data = pd.read_csv("automobile_2.csv")

# Create joint plot
sns.jointplot(x=pd.to_numeric(data["stroke"]), y=data["price"], kind="reg", color='purple')

# Show plot
plt.show()