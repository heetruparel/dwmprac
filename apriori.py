import itertools

# Dataset
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Bread'],
    ['Milk', 'Bread', 'Butter', 'Beer'],
    ['Milk', 'Butter'],
    ['Bread', 'Butter']
]

# Step 1: Create list of items
items = set(item for transaction in transactions for item in transaction)

# Step 2: Create candidate itemsets
def create_candidates(items, length):
    return list(itertools.combinations(items, length))

# Step 3: Calculate support
def calculate_support(transactions, candidates):
    support_count = {}
    for candidate in candidates:
        count = 0
        for transaction in transactions:
            if set(candidate).issubset(set(transaction)):
                count += 1
        support_count[candidate] = count / len(transactions)
    return support_count

# Step 4: Filter itemsets with min support
def filter_candidates(support_count, min_support):
    return {item: support for item, support in support_count.items() if support >= min_support}

# Apriori
min_support = 0.6
frequent_itemsets = {}

length = 1
current_items = items

while True:
    candidates = create_candidates(current_items, length)
    support_count = calculate_support(transactions, candidates)
    frequent = filter_candidates(support_count, min_support)
    if not frequent:
        break
    frequent_itemsets.update(frequent)
    current_items = set(item for itemset in frequent for item in itemset)
    length += 1

print("Frequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(itemset, "=>", round(support, 2))
