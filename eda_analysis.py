import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/va_edd_dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Scatter plots
sns.scatterplot(x=df['IP'], y=df['MRR'])
plt.title("IP vs MRR")
plt.show()

sns.scatterplot(x=df['TON'], y=df['MRR'])
plt.title("TON vs MRR")
plt.show()

sns.scatterplot(x=df['TOFF'], y=df['MRR'])
plt.title("TOFF vs MRR")
plt.show()

sns.scatterplot(x=df['IP'], y=df['SR'])
plt.title("IP vs SR")
plt.show()