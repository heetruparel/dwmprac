# Basic Node for FP-Tree
class TreeNode:
    def __init__(self, name, count, parent):
        self.name = name            # Item name
        self.count = count          # Count
        self.parent = parent        # Parent node
        self.children = {}          # Children nodes
        self.link = None            # Link to same item in tree (for header table)

    def increment(self, count):
        self.count += count

    def display(self, level=0):
        print(' ' * level * 2, f'{self.name} ({self.count})')
        for child in self.children.values():
            child.display(level + 1)

# Create initial set
def create_initial_set(dataset):
    ret_dict = {}
    for trans in dataset:
        ret_dict[frozenset(trans)] = 1
    return ret_dict

# Build the FP-tree
def create_fp_tree(dataset, min_support=1):
    header_table = {}

    # First scan: count frequency
    for transaction in dataset:
        for item in transaction:
            header_table[item] = header_table.get(item, 0) + dataset[transaction]

    # Remove items not meeting min_support
    header_table = {k: v for k, v in header_table.items() if v >= min_support}
    freq_item_set = set(header_table.keys())

    if len(freq_item_set) == 0:
        return None, None  # No items meet min_support

    # Initialize header table: {item: [count, node_link]}
    for k in header_table:
        header_table[k] = [header_table[k], None]

    # Create root of FP-tree
    root_node = TreeNode('Null', 1, None)

    # Second scan: build the tree
    for transaction, count in dataset.items():
        localD = {}
        for item in transaction:
            if item in freq_item_set:
                localD[item] = header_table[item][0]  # Get support

        if len(localD) > 0:
            # Sort items in descending order of frequency
            ordered_items = [v[0] for v in sorted(localD.items(), key=lambda p: (-p[1], p[0]))]
            update_tree(ordered_items, root_node, header_table, count)

    return root_node, header_table

# Update tree
def update_tree(items, inTree, header_table, count):
    first_item = items[0]
    if first_item in inTree.children:
        inTree.children[first_item].increment(count)
    else:
        new_node = TreeNode(first_item, count, inTree)
        inTree.children[first_item] = new_node

        # Link it to header table
        if header_table[first_item][1] is None:
            header_table[first_item][1] = new_node
        else:
            update_header(header_table[first_item][1], new_node)

    # Recursively add remaining items
    if len(items) > 1:
        update_tree(items[1:], inTree.children[first_item], header_table, count)

def update_header(node_to_test, target_node):
    while node_to_test.link is not None:
        node_to_test = node_to_test.link
    node_to_test.link = target_node

# Example transactions
transactions = [
    ['milk', 'bread', 'beer'],
    ['bread', 'diapers', 'eggs'],
    ['milk', 'bread', 'diapers', 'beer'],
    ['bread', 'milk', 'diapers', 'eggs'],
    ['milk', 'bread', 'diapers', 'beer']
]

# Convert to dictionary
init_set = create_initial_set(transactions)

# Build tree
fp_tree, header_table = create_fp_tree(init_set, min_support=3)

# Display tree
print("FP-Tree Structure:")
fp_tree.display()
