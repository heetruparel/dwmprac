import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = {
'Age': [18, 22, 25, 27, 30, 34, 37, 41, 45, 50, 54, 60, 67, 70, 75]
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 4))
sns.histplot(df['Age'], kde=True, bins=10, color='steelblue', edgecolor='black')
plt.title("Original Distribution of 'Age'")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

num_bins = 4
bin_labels = [f'Bin{i+1}' for i in range(num_bins)]
df['Age_Binned'] = pd.cut(df['Age'], bins=num_bins, labels=bin_labels)
print("\n== Discretized Data (Equal Width Bins) ==\n")
print(df)






vis
plt.figure(figsize=(8, 4))
sns.countplot(x='Age_Binned', data=df, palette='Set2', edgecolor='black')
plt.title("Data After Equal Width Discretization", fontsize=14)
plt.xlabel("Bins", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.box(False)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Age'], kde=True, bins=10, color='skyblue', edgecolor='black',
ax=axs[0])
axs[0].set_title('Original Continuous Data')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Frequency')
axs[0].grid(True)

sns.countplot(x='Age_Binned', data=df, palette='pastel', edgecolor='black',
ax=axs[1])
axs[1].set_title('Discretized Data (Equal Width)')
axs[1].set_xlabel('Bins')
axs[1].set_ylabel('Count')
axs[1].grid(True)
