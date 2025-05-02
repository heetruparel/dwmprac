def build_tree(data, features, depth=0):
    labels = data[target_col].unique()
    
    # If only one class remains
    if len(labels) == 1:
        return labels[0]
    
    # If no features left
    if not features:
        return data[target_col].mode()[0]
    
    # Find best feature
    gains = {f: info_gain(data, f) for f in features}
    best_feature = max(gains, key=gains.get)

    tree = {best_feature: {}}
    
    for val in data[best_feature].unique():
        subset = data[data[best_feature] == val]
        if subset.empty:
            tree[best_feature][val] = data[target_col].mode()[0]
        else:
            remaining_features = [f for f in features if f != best_feature]
            tree[best_feature][val] = build_tree(subset, remaining_features, depth + 1)
    
    return tree

# Pretty print the tree
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "→", tree)
        return
    for feature, branches in tree.items():
        for val, subtree in branches.items():
            print(f"{indent}If {feature} == '{val}':")
            print_tree(subtree, indent + "  ")

# Build and print the tree
tree = build_tree(df, features)
print("\nDecision Tree:")
print_tree(tree)
