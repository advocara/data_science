from typing import Dict, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from random_forest_paths import FeatureImpact, analyze_feature_impact, format_impact_analysis


def train_random_forest(X, y):
    """
    Train a Random Forest model on the given data.
    
    Args:
        X: Feature DataFrame
        y: Target variable Series
    
    Returns:
        tuple: Trained model, X_train, X_test, y_train, y_test
    """
    # Split into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
    clf.fit(X_train, y_train)
    
    return clf, X_train, X_test, y_train, y_test


def run_random_forest(df) -> None:
    """
    Train a Random Forest model on the given data.
    Print the most impactful feature combinations.
    """
    #### Prepare features and target ####
    X = df.drop(columns=["is_denial_upheld"])
    y = df["is_denial_upheld"]
    # Train model
    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    # Feature Importance
    feature_importances = model.feature_importances_

    ### Plot single-feature importance and display bar chart ###
    # plt.barh(X.columns, feature_importances)
    # plt.xlabel("Feature Importance")
    # plt.ylabel("Feature")
    # plt.title("Random Forest Feature Importance in Medical Appeals Prediction")
    # plt.show()
    #### ======

    # **Use our path counting module for Feature Path Analysis**
    print("\n\n\n\n\n===\n===\n=== Feature Impact Analysis ===")
    impact_metrics: Dict[Tuple[str, ...], FeatureImpact] = analyze_feature_impact(model, X, min_occurrences=5, max_path_length=3)
    results = format_impact_analysis(impact_metrics, top_n=50)

    print("\nMost Impactful Feature Combinations:")
    print("------------------------------------")
    for result in results:
        print(result)
        print("------------------------------------")

    # print("\n--- Most Common Multi-Feature Pathways ---")
    # common_combinations = random_forests_paths.extract_feature_combinations(model, X.columns)
    # for features, count in common_combinations.items():
    #     print(f"{features} (Appeared {count} times)")

    # # **Visualize a Sample Decision Tree**
    # random_forests_paths.visualize_decision_tree(model, X.columns, max_depth=3)
