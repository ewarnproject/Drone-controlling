import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("car_companies.csv")

print(df.head())

plt.figure(figsize=(10, 6))

sns.histplot(
    df['price'],
    bins=20,       
    kde=True,      
    color='black'
)

plt.title("Histogram of Price Distribution")
plt.xlabel("Price")
plt.ylabel("company")

plt.show()