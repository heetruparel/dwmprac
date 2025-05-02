import pandas as pd

#For excel file. Jaha bhi tumhara file stored hai, uska location
df = pd.read_excel(r'C:\Users\Cinepix\Downloads\Electronics.xlsx')


# Assume last column is the class (target)
target_col = df.columns[-1]

```Drop R_Id if present. Excel sheet mein check karlo if there is serial Number ka such column```
if 'R_Id' in df.columns:
    df = df.drop(columns=['R_Id'])

# Get feature columns (excluding the target column)
feature_cols = [col for col in df.columns if col != target_col]

# Step 1: Prior probabilities
total_samples = len(df)
classes = df[target_col].unique()
priors = {c: len(df[df[target_col] == c]) / total_samples for c in classes}

print("\nPrior Probabilities:")
for c in priors:
    print(f"P({target_col} = '{c}') = {round(priors[c], 3)}")

# Step 2: Likelihoods
def calc_likelihood(df, feature, feature_value, target_class):
    subset = df[df[target_col] == target_class]
    feature_count = len(subset[subset[feature] == feature_value])
    total_count = len(subset)
    return feature_count / total_count if total_count > 0 else 0

# Let's say this is the input sample to predict:
# (Change these values based on actual test case)
sample = {
    'Age': 'Youth',
    'Income': 'Medium',
    'Student': 'Yes',
    'Credit Rating': 'Fair'
}

# Step 3: Calculate likelihoods and posteriors
likelihoods = {}
posteriors = {}

print("\nLikelihoods:")

for c in classes:
    prob = 1
    print(f"\nFor class = '{c}':")
    for feature in feature_cols:
        feature_val = sample.get(feature)
        likelihood = calc_likelihood(df, feature, feature_val, c)
        prob *= likelihood
        print(f"P({feature} = '{feature_val}' | {target_col} = '{c}') = {round(likelihood, 3)}")
    posteriors[c] = prob * priors[c]

# Step 4: Print posterior probabilities
print("\nPosterior Probabilities (after applying Bayes' theorem):")
for c in posteriors:
    print(f"P(X | {target_col} = '{c}') * P({target_col} = '{c}') = {round(posteriors[c], 3)}")

# Step 5: Predict the class
prediction = max(posteriors, key=posteriors.get)
print(f"\nPredicted class: {target_col} = '{prediction}'")
