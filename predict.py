import json
import glob
import os
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

from random_forest_paths import extract_multi_feature_paths, analyze_feature_impact, format_impact_analysis, FeatureImpact
from model.appeal import MedicalInsuranceAppeal
from term_normalizer import Normalizer

import random
from deap import base, creator, tools, algorithms
from sklearn.metrics import mutual_info_score

def appeals_as_dataframe_onehot(records):
    """Convert JSON-based medical appeal records into a structured DataFrame."""
    df = pd.DataFrame(records)
    
    # Extract top-level attributes
    df["gender"] = df["patient_info"].apply(lambda x: x["gender"])
    df["age_range"] = df["patient_info"].apply(lambda x: x["age_range"])
    df["disease"] = df["diagnosis"]
    df["treatment_category"] = df["treatment_category"] # NOOP
    df["treatment_subcategory"] = df["treatment_subcategory"] # NOOP
    df["is_denial_upheld"] = df["is_denial_upheld"].astype(int)  # Target variable
    
    # Encode support flags (binary) - be explicit about handling None values
    boolean_columns = [
        "guidelines_support",
        "guidelines_not_support", 
        "soc_support",
        "soc_not_support"
    ]
    
    for col in boolean_columns:
        # Convert None to False and ensure boolean type
        df[col] = df[col].map({True: 1, False: 0, None: 0}).astype(int)

    # Encode treatments (one-hot encoding for dynamic values)
    def encode_treatments(col_name):
        all_treatments = set()
        for record in records:
            all_treatments.update([t["name"] for t in record.get(col_name, [])])
        for treatment in all_treatments:
            df[f"{col_name}_{treatment}"] = df[col_name].apply(lambda x: int(any(t["name"] == treatment for t in x)))
    
    encode_treatments("treatments_requested")
    encode_treatments("treatments_tried_but_failed")
    
    # Encode categorical variables
    categorical_cols = ["gender", "age_range", "disease", "treatment_category", "treatment_subcategory"]
    df = pd.get_dummies(df, columns=categorical_cols, dtype=int)
    
    # Drop original JSON columns now that we have extracted the data to one-hot encoded columns
    df = df.drop(columns=["patient_info", "diagnosis", "secondary_conditions", "complications", "symptoms", 
                          "treatments_requested", "treatments_tried_but_failed", "treatments_tried_and_worked", 
                          "treatments_not_tried", "issues_considered", "guidelines_details", "soc_details", "study_details", 
                          "key_questions", "rationale", "reviewer_credentials", "case_id"], errors="ignore")
    
    return df


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


# Load JSON files from cache directory
medical_appeals = []
normalized_appeal_dir = "D:\\Advocara\\data_science\\appeals-results"
category_substring = "Immuno Disorders-Lupus-norm"

# Read all JSON files in cache directory
for json_file in glob.glob(os.path.join(normalized_appeal_dir, f"*{category_substring}*.json")):
    with open(json_file, 'r') as f:
        medical_appeals.append(MedicalInsuranceAppeal.model_validate(json.load(f)))

print(f"Loaded {len(medical_appeals)} medical appeal records for: {category_substring}")

#### Convert to a dataframe with categories flattened as one-hot encoded columns ####
medical_appeal_dicts = [appeal.model_dump() for appeal in medical_appeals]
df = appeals_as_dataframe_onehot(medical_appeal_dicts) # convert back to dict now that name normalization is done

method = "deap"

if method == "random_forest":
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



elif method == 'FPgrowth':
    #dropping year column
    df_drop_year = df.drop(columns=["year"], axis=1)

    # Split data by denial decision (your target variable)
    df_denial_upheld = df_drop_year[df_drop_year['is_denial_upheld'] == 1].drop('is_denial_upheld', axis=1)
    df_denial_overturned = df_drop_year[df_drop_year['is_denial_upheld'] == 0].drop('is_denial_upheld', axis=1)

    # Fill NaN values with 0 (since this is binary data)
    df_denial_upheld = df_denial_upheld.fillna(0)
    df_denial_overturned = df_denial_overturned.fillna(0)

    # Let's first check if there are any remaining issues
    # print("Upheld denials shape:", df_denial_upheld.shape)
    # print("Overturned denials shape:", df_denial_overturned.shape)

    # Run FP-growth for each class
    # You might want to adjust min_support based on your data size
    patterns_upheld = fpgrowth(df_denial_upheld, 
                            min_support=0.05,  # Appears in at least 5% of cases
                            use_colnames=True)
    # patterns_upheld = patterns_upheld[patterns_upheld['itemsets'].apply(lambda x: len(x) > 1)]


    patterns_overturned = fpgrowth(df_denial_overturned, 
                                min_support=0.05, 
                                use_colnames=True)
    # patterns_overturned = patterns_overturned[patterns_overturned['itemsets'].apply(lambda x: len(x) > 1)]


    # Sort patterns by support
    patterns_upheld = patterns_upheld.sort_values('support', ascending=False)
    patterns_overturned = patterns_overturned.sort_values('support', ascending=False)


   # Print top patterns for each class
    print("Top patterns for Upheld Denials:")
    print(patterns_upheld.head(10))
    print("\nTop patterns for Overturned Denials:")
    print(patterns_overturned.head(10))

    # Analyze discriminative patterns (patterns that appear much more in one class vs the other)
    def get_discriminative_patterns(patterns_1, patterns_2, min_diff=0.1):
        """Find patterns that have significantly different support between classes"""
        patterns_1_dict = dict(zip(patterns_1['itemsets'], patterns_1['support']))
        patterns_2_dict = dict(zip(patterns_2['itemsets'], patterns_2['support']))
        
        discriminative = []
        for itemset, support1 in patterns_1_dict.items():
            support2 = patterns_2_dict.get(itemset, 0)
            if abs(support1 - support2) >= min_diff:
                discriminative.append({
                    'itemset': itemset,
                    'support_upheld': support1,
                    'support_overturned': support2,
                    'difference': support1 - support2
                })
        
        return pd.DataFrame(discriminative).sort_values('difference', ascending=False)

    discriminative_patterns = get_discriminative_patterns(patterns_upheld, patterns_overturned)
    print("\nMost discriminative patterns:")
    print(discriminative_patterns.head())

    
    # discriminative_patterns_opp = get_discriminative_patterns(patterns_overturned, patterns_upheld)
    # print("\nMost discriminative patterns opposite:")
    # print(discriminative_patterns_opp.head())


        
    # # Generate rules
    # rules_upheld = association_rules(patterns_upheld, metric="confidence", min_threshold=0.7, support_only=True)
    # print("Rules upheld \n", rules_upheld.head(10))

    # rules_overturned = association_rules(patterns_overturned, metric="confidence", min_threshold=0.7, support_only=True)
    # print("Rules overturned \n", rules_overturned.head(10))

elif method == "deap":
    X = df.drop(columns=['is_denial_upheld'])    
    y = df['is_denial_upheld']

    # DEAP Setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.000001, 0.000001))  # Maximize mutual info score
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Hyperparameters
    POP_SIZE = 5000
    NUM_GENERATIONS = 30
    MUTATION_RATE = 0.7
    CROSSOVER_RATE = 0.6
    TUPLE_MIN_SIZE = 2
    TUPLE_MAX_SIZE = 7

    # Initialize population with random feature tuples
    def generate_individual():
        # Get unique columns (remove duplicates from X.columns)
        unique_columns = list(dict.fromkeys(X.columns))  # preserves order unlike set()
        
        tuple_size = random.randint(TUPLE_MIN_SIZE, TUPLE_MAX_SIZE)
        # Ensure we don't exceed available unique features
        tuple_size = min(tuple_size, len(unique_columns))
        return creator.Individual(random.sample(unique_columns, tuple_size))

    def evaluate(individual):
        selected_features = list(individual)
        # For upheld denials (class 1)
        upheld_matches = df[((df[selected_features] == 1).all(axis=1)) & (df['is_denial_upheld'] == 1)].shape[0]
        total_upheld = df[df['is_denial_upheld'] == 1].shape[0]
        upheld_precision = upheld_matches / total_upheld if total_upheld > 0 else 0

        # For overturned denials (class 0)
        overturned_matches = df[((df[selected_features] == 1).all(axis=1)) & (df['is_denial_upheld'] == 0)].shape[0]
        total_overturned = df[df['is_denial_upheld'] == 0].shape[0]
        overturned_precision = overturned_matches / total_overturned if total_overturned > 0 else 0



        # Calculate the absolute difference between precisions
        # This rewards feature sets that strongly favor one class over the other
        precision_difference = abs(upheld_precision - overturned_precision)
        
        return (precision_difference, upheld_precision, overturned_precision)

    def mutate(individual):
        if len(individual) == TUPLE_MAX_SIZE:
            # If at max size, only allow removal
            individual.pop(random.randint(0, len(individual)-1))
        elif len(individual) == TUPLE_MIN_SIZE:
            # If at min size, only allow addition 
            available_features = list(set(X.columns) - set(individual))
            if available_features:  # if there are features available to add
                new_feature = random.choice(available_features)
                individual.append(new_feature)
        else:
            # Randomly add or remove
            if random.random() < 0.5:
                individual.pop(random.randint(0, len(individual)-1))
            else:
                available_features = list(set(X.columns) - set(individual))
                if available_features:
                    new_feature = random.choice(available_features)
                    individual.append(new_feature)
        return individual,

    # Crossover: Swap parts of two tuples
    def crossover(ind1, ind2):
        # Convert to sets to handle duplicates
        set1 = set(ind1)
        set2 = set(ind2)
        
        # Check if we have enough elements for crossover
        if len(set1) < 2 or len(set2) < 2:
            # If sets are too small, just return original individuals
            return ind1, ind2
        
        # Random point for crossover
        point = random.randint(1, min(len(set1), len(set2))-1)
        
        # Convert back to lists and perform crossover
        list1 = list(set1)
        list2 = list(set2)
        list1[:point], list2[:point] = list2[:point], list1[:point]
        
        # Remove any duplicates and ensure minimum size
        new_ind1 = list(dict.fromkeys(list1))
        new_ind2 = list(dict.fromkeys(list2))
        
        # Ensure minimum size is maintained
        while len(new_ind1) < TUPLE_MIN_SIZE:
            available = list(set(X.columns) - set(new_ind1))
            if available:
                new_ind1.append(random.choice(available))
                
        while len(new_ind2) < TUPLE_MIN_SIZE:
            available = list(set(X.columns) - set(new_ind2))
            if available:
                new_ind2.append(random.choice(available))
        
        ind1[:] = new_ind1
        ind2[:] = new_ind2
        
        return ind1, ind2

    def make_unique_population(population, k):
        # Convert individuals to tuples of features for hashability
        seen = set()
        unique_population = []
        
        for ind in population:
            ind_tuple = tuple(sorted(ind))  # Sort to treat permutations as same
            if ind_tuple not in seen:
                seen.add(ind_tuple)
                unique_population.append(ind)
                if len(unique_population) == k:
                    break
        
        # If we need more individuals, generate new ones
        while len(unique_population) < k:
            new_ind = toolbox.individual()
            ind_tuple = tuple(sorted(new_ind))
            if ind_tuple not in seen:
                seen.add(ind_tuple)
                # Evaluate new individual before adding
                new_ind.fitness.values = toolbox.evaluate(new_ind)
                unique_population.append(new_ind)
        
        return unique_population

    # Evolutionary Process
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=random.randint(2, 4))

    # Run Evolution
    population = toolbox.population()
    best_fitness = []

    for gen in range(NUM_GENERATIONS):
        offspring = algorithms.varAnd(population, toolbox, CROSSOVER_RATE, MUTATION_RATE)
        fits = list(map(toolbox.evaluate, offspring))
        
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        # First select based on fitness
        selected = toolbox.select(offspring, k=len(population))
        # Then ensure diversity
        population = make_unique_population(selected, k=len(population))

        # Log best fitness of generation
        gen_fits = [ind.fitness.values[0] for ind in population]
        best_fitness.append(max(gen_fits))
        print(f"Generation {gen}: Best Fitness = {best_fitness[-1]:.4f}")

    def analyze_results(population):
    # Sort by fitness
        top_tuples = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
        
        # Create DataFrame with separate columns for features and fitness
        results = []
        for ind in top_tuples:
            results.append({
                'Features': ', '.join(ind),
                'Feature Count': len(ind),
                'Fitness Score': ind.fitness.values[0],
                'upheld_precision': ind.fitness.values[1],
                'overturned_precision': ind.fitness.values[2]
            })
        
        results_df = pd.DataFrame(results)
        return results_df

    results_df = analyze_results(population)
    print("\nTop Feature Combinations:")
    print(results_df)
    results_df.to_csv('deap_results_newdash_eval.csv', index=False, sep=',', encoding='utf-8')
    


# Plot fitness history
    import matplotlib.pyplot as plt
    plt.plot(best_fitness)
    plt.title('Best Fitness Score by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score (Mutual Information)')
    plt.show()
    # Convert to DataFrame for analysis
    # import ace_tools as tools
    # results_df = pd.DataFrame(top_tuples, columns=["Feature Tuple"])
    
    # tools.display_dataframe_to_user(name="Best Feature Tuples Impacting Appeals", dataframe=results_df)

