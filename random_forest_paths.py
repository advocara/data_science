from collections import Counter
from itertools import combinations
from typing import Dict, Tuple
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

def extract_multi_feature_paths(model, df: pd.DataFrame, min_occurrences=5, max_path_length=4):
    """
    Extracts and counts feature combinations (2, 3, 4 features) appearing frequently in decision trees.
    
    Args:
        model: Trained RandomForestClassifier.
        feature_names: List of feature names.
        min_occurrences: Minimum times a feature combination must appear.
        max_path_length: Maximum feature group size (e.g., 2 for pairs, 3 for triples, etc.).

    Returns:
        Dictionary of feature combinations and their frequencies.
    """
    feature_names = df.columns
    feature_types = df.dtypes
    
    # Identify binary columns (both boolean and 0/1 integers)
    binary_columns = [
        "guidelines_support", "guidelines_not_support",
        "soc_support", "soc_not_support"
    ]
    binary_features = set(col for col in feature_names if col in binary_columns)
    
    path_combinations = Counter()

    for tree in model.estimators_:
        tree_ = tree.tree_
        node_count = tree_.node_count

        for leaf in range(node_count):
            if tree_.children_left[leaf] == -1 and tree_.children_right[leaf] == -1:  # Leaf node
                path_features = set()
                node = leaf

                # Traverse up to root, collecting features used in splits
                while node != 0:
                    parent = np.where((tree_.children_left == node) | (tree_.children_right == node))[0]
                    if parent.size == 0:
                        break
                    parent = parent[0]

                    feature = feature_names[tree_.feature[parent]]
                    threshold = tree_.threshold[parent]

                    # Only add True/False suffix for known binary features
                    if feature in binary_features:
                        feature = f"{feature}_True" if threshold >= 0.5 else f"{feature}_False"
                    # Other features (categorical) already have their value suffixes
                    
                    path_features.add(feature)
                    node = parent

                # Store feature combinations of length 2 to max_path_length
                path_features = sorted(path_features)  # Ensure consistency
                for size in range(2, max_path_length + 1):
                    for combo in combinations(path_features, size): # combo selects all ordered sublists of the desired size, including skipping elements
                        path_combinations[combo] += 1
                #TODO: seems this will favor 2-tuples since they will be more common than more detailed combinations
                #      we probably want to find some measure of predictability next, vs just frequency 

    # Filter to show only frequently occurring feature groups
    # feature_value is the feature and the specific value such as gender_Male or guidelines_not_support (if it was bool true)
    return {feature_value: count for feature_value, count in path_combinations.items() if count >= min_occurrences}



# deprecated. use the n-length path extraction above
def extract_size2_decision_paths(model: RandomForestClassifier, feature_names: List[str], top_n_paths: int = 10) -> Dict[Tuple[str, float], int]:
    """
    Extract the most common decision splits in a Random Forest model.

    Args:
        model: Trained RandomForestClassifier.
        feature_names: List of feature names.
        top_n_paths: Number of most common feature splits to return.

    Returns:
        List of most common decision splits and their frequencies.
    """
    path_counter = Counter()

    for tree in model.estimators_:
        tree_ = tree.tree_  # Access the underlying tree structure
        num_nodes = tree_.node_count

        for node in range(num_nodes):
            if tree_.feature[node] != -2:  # -2 indicates a leaf node
                feature = feature_names[tree_.feature[node]]
                threshold = tree_.threshold[node]
                path_counter[(feature, threshold)] += 1

    return path_counter.most_common(top_n_paths)


def extract_feature_combinations(model, feature_names, min_occurrences=5):
    """
    Extract and count feature combinations appearing frequently in decision trees.
    
    Args:
        model: Trained RandomForestClassifier.
        feature_names: List of feature names.
        min_occurrences: Minimum times a feature combination must appear.

    Returns:
        Dictionary of feature combinations and their frequencies.
    """
    path_combinations = Counter()

    for tree in model.estimators_:
        tree_ = tree.tree_
        node_indicator = tree_.children_left != -1  # Only split nodes

        for node in range(tree_.node_count):
            if node_indicator[node]:  # Skip leaf nodes
                path_features = set()
                while node != 0:  # Traverse up to root
                    feature = feature_names[tree_.feature[node]]
                    path_features.add(feature)
                    node = tree_.parent[node]
                
                # Store as a sorted tuple for consistency
                if len(path_features) > 1:
                    path_combinations[tuple(sorted(path_features))] += 1

    # Filter by minimum occurrences
    return {k: v for k, v in path_combinations.items() if v >= min_occurrences}


def visualize_decision_tree(model, feature_names, max_depth=4):
    """
    Visualize a single decision tree from the Random Forest.

    Args:
        model: Trained RandomForestClassifier.
        feature_names: List of feature names.
        max_depth: Depth to limit visualization for readability.

    Returns:
        A matplotlib plot of the tree.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], feature_names=feature_names, filled=True, max_depth=max_depth)
    plt.show()
