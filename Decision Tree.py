import pandas as pd
import math

# Load dataset
df = pd.read_excel(r'C:\Users\Cinepix\Downloads\Electronics.xlsx')

# Drop R_Id if present
if 'R_Id' in df.columns:
    df = df.drop(columns=['R_Id'])

# Target column (assume last)
target_col = df.columns[-1]
features = [col for col in df.columns if col != target_col]

# Function to compute entropy
def entropy(data):
    total = len(data)
    if total == 0:
        return 0
    counts = data[target_col].value_counts()
    ent = 0
    for c in counts:
        p = c / total
        ent -= p * math.log2(p)
    return ent

# Function to compute information gain
def info_gain(data, feature):
    total_entropy = entropy(data)
    values = data[feature].unique()
    weighted_entropy = 0
    for val in values:
        subset = data[data[feature] == val]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset)
    return total_entropy - weighted_entropy
