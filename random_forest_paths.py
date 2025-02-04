from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Tuple, List

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

def analyze_feature_impact(model, df: pd.DataFrame, min_occurrences=5, max_path_length=4):
    """
    Analyzes feature combinations to determine their impact on denial decisions.
    """
    feature_names = df.columns
    binary_columns = [
        "guidelines_support", "guidelines_not_support",
        "soc_support", "soc_not_support"
    ]
    binary_features = set(col for col in feature_names if col in binary_columns)
    
    # Track both counts and outcomes
    path_stats = defaultdict(lambda: {'count': 0, 'deny': 0, 'approve': 0})

    # Calculate baseline denial rate across all trees
    total_denials = 0
    total_samples = 0
    for tree in model.estimators_:
        tree_ = tree.tree_
        # Get root node values [n_samples_negative, n_samples_positive]
        root_value = tree_.value[0][0]
        total_denials += root_value[1]  # Count of denials
        total_samples += root_value.sum()  # Total samples
    
    baseline_denial_rate = total_denials / total_samples

    for tree in model.estimators_:
        tree_ = tree.tree_
        node_count = tree_.node_count

        # For each leaf node, track the path and the prediction
        for leaf in range(node_count):
            if tree_.children_left[leaf] == -1:  # Leaf node
                path_features = set()
                node = leaf
                
                # Get the prediction for this leaf
                value = tree_.value[leaf][0]
                prediction = np.argmax(value)  # 0 for approve, 1 for deny
                samples_in_leaf = np.sum(value)

                # Traverse up to root
                while node != 0:
                    parent = np.where((tree_.children_left == node) | (tree_.children_right == node))[0][0]
                    
                    feature = feature_names[tree_.feature[parent]]
                    threshold = tree_.threshold[parent]
                    
                    # Handle binary features
                    if feature in binary_features:
                        feature = f"{feature}_True" if threshold >= 0.5 else f"{feature}_False"
                    
                    path_features.add(feature)
                    node = parent

                # Store combinations and their outcomes
                path_features = sorted(path_features)
                for size in range(1, min(max_path_length + 1, len(path_features) + 1)):
                    for combo in combinations(path_features, size):
                        stats = path_stats[combo]
                        stats['count'] += samples_in_leaf
                        if prediction == 1:  # Denial upheld
                            stats['deny'] += samples_in_leaf
                        else:  # Denial overturned
                            stats['approve'] += samples_in_leaf

    # Calculate impact metrics
    impact_metrics = {}
    for combo, stats in path_stats.items():
        if stats['count'] >= min_occurrences:
            deny_rate = stats['deny'] / stats['count']
            impact = abs(deny_rate - baseline_denial_rate)
            
            impact_metrics[combo] = {
                'count': stats['count'],
                'deny_rate': deny_rate,
                'impact': impact,
                'direction': 'increases denials' if deny_rate > baseline_denial_rate else 'decreases denials'
            }

    return impact_metrics

def format_impact_analysis(impact_metrics, top_n=20):
    """
    Formats the impact analysis results into a readable format.
    
    Args:
        impact_metrics: Dictionary from analyze_feature_impact
        top_n: Number of top impactful combinations to show
    
    Returns:
        List of formatted strings describing the most impactful feature combinations
    """
    # Sort by impact
    sorted_impacts = sorted(
        impact_metrics.items(),
        key=lambda x: (x[1]['impact'], x[1]['count']),
        reverse=True
    )[:top_n]
    
    results = []
    for combo, metrics in sorted_impacts:
        feature_str = " + ".join(combo)
        impact_pct = metrics['impact'] * 100
        deny_rate_pct = metrics['deny_rate'] * 100
        
        result = (
            f"Features: {feature_str}\n"
            f"Impact: {impact_pct:.1f}% ({metrics['direction']})\n"
            f"Denial Rate: {deny_rate_pct:.1f}%\n"
            f"Occurrences: {metrics['count']}\n"
        )
        results.append(result)
    
    return results
