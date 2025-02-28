
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import pandas as pd

def run_FP(df):
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