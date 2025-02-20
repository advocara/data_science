import random
import pandas as pd
from deap import base, creator, tools, algorithms
import copy

def run_deap(df, filename):
    X = df.drop(columns=['is_denial_upheld'])    
    y = df['is_denial_upheld']

    # DEAP Setup
    # mostly weight the precision difference, but also want to boost cases that are common, and favor longer more explanatory feature sets
    # e.g. if 5 appeals match a combination of 5 features, we want to report all 5, not just 3
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.000001, 0.000001, 0.000001, 0.000001, 0.02, 0.02))  # Maximize mutual info score
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Hyperparameters
    POP_SIZE = 2000
    NUM_GENERATIONS = 30
    MUTATION_RATE = 0.4
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
        ind_len = len(selected_features)

        # Create smaller df with just the selected features and upheld flag. We never need other columns.
        relevant_columns_df = df[selected_features + ['is_denial_upheld']]
        
        # first compute total upheld and overturned regardless of features (TODO cache these numbers)
        total_upheld = (relevant_columns_df['is_denial_upheld'] == 1).sum()
        total_overturned = len(relevant_columns_df) - total_upheld

        # Generate masks (one dim column vectors) for feature/upheld combinations
        mask_all_features = (relevant_columns_df[selected_features] == 1).all(axis=1) # one-dim column with true if all features are 1
        
        # if nothing has all the features, skip later filtering
        if not mask_all_features.any():
            return (0, 0, 0, 0, 0, 0, 0)
        
        mask_upheld_with_all_features = mask_all_features & (df['is_denial_upheld'] == 1) # one-dim column with true if above and also upheld
        mask_overturned_with_all_features = mask_all_features & (df['is_denial_upheld'] == 0) # one-dim column with true if above and also overturned
        
        # Count matches by summing the boolean masks directly (no need to build new data frames)
        total_matches_of_this_set = mask_all_features.sum() # count the True values in the mask
        upheld_matches = mask_upheld_with_all_features.sum()
        overturned_matches = mask_overturned_with_all_features.sum()

        # compute precision
        upheld_precision = upheld_matches / total_upheld if total_upheld > 0 else 0
        overturned_precision = overturned_matches / total_overturned if total_overturned > 0 else 0
        precision_difference = abs(upheld_precision - overturned_precision)
        
        return (precision_difference, upheld_precision, upheld_matches, overturned_precision, overturned_matches, total_matches_of_this_set, ind_len)

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
        if random.random() < 0.3:
            return crossover_addone(ind1, ind2)
        
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

    def crossover_addone(ind1, ind2):
        # Create copies to avoid modifying originals
        new_ind1 = ind1[:]
        new_ind2 = ind2[:]
        
        # For first individual: try to add one random feature from ind2
        available_from_ind2 = list(set(ind2) - set(ind1))
        if available_from_ind2 and len(new_ind1) < TUPLE_MAX_SIZE:
            new_feature = random.choice(available_from_ind2)
            new_ind1.append(new_feature)
        
        # For second individual: try to add one random feature from ind1
        available_from_ind1 = list(set(ind1) - set(ind2))
        if available_from_ind1 and len(new_ind2) < TUPLE_MAX_SIZE:
            new_feature = random.choice(available_from_ind1)
            new_ind2.append(new_feature)

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

    def score_pop(population, toolbox):
        """Score a population of individuals using the toolbox evaluator.
        
        Args:
            population: List of individuals to evaluate
            toolbox: DEAP toolbox containing the evaluate function
            
        Returns:
            The same population with updated fitness values
        """
        fits = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fits):
            ind.fitness.values = fit
        return population
    
    def selective_tournament(individuals, k, tournsize, fitness_threshold=0.5):
        chosen = []
        for _ in range(k):
            # Keep sampling tournaments until we find one with at least one fit individual
            max_attempts = 5  # Limit attempts to prevent infinite loop
            for _ in range(max_attempts):
                aspirants = tools.selRandom(individuals, k=tournsize)
                # Check if any individuals meet our fitness threshold
                fit_aspirants = [ind for ind in aspirants if ind.fitness.values[0] > fitness_threshold]
                if fit_aspirants:
                    # If we found fit individuals, select the best one
                    chosen.append(max(fit_aspirants, key=lambda ind: ind.fitness.values[0]))
                    # print('fitval = ', max(aspirants, key=lambda ind: ind.fitness.values[0]).fitness.values[0])
                    break
            else:
                # If we didn't find any fit individuals after max_attempts, take best from last tournament
                chosen.append(max(aspirants, key=lambda ind: ind.fitness.values[0]))
        return chosen
    
    # Evolutionary Process
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover_addone)
    toolbox.register("mutate", mutate)
    # toolbox.register("select", tools.selTournament, tournsize=random.randint(2, 4))
    toolbox.register("select", selective_tournament, tournsize=3)

    # Run Evolution
    population = toolbox.population()
    population = score_pop(population, toolbox)  # Replace the manual scoring

    best_fitness = []

    for gen in range(NUM_GENERATIONS):
        ELITE_SIZE = 50  # Increase from 20

        offspring = algorithms.varAnd(population, toolbox, CROSSOVER_RATE, MUTATION_RATE)
        offspring = score_pop(offspring, toolbox)

        # First select based on fitness
        # best_offspring = toolbox.select(offspring, k=len(population)-ELITE_SIZE)
        fit_offspring = [ind for ind in offspring if ind.fitness.values[0] > 0.5]  # Minimum fitness threshold

        if len(fit_offspring) > 0:
            best_offspring = toolbox.select(fit_offspring, k=len(population)-ELITE_SIZE)
        else:
            # If no fit offspring, select from current population
            best_offspring = toolbox.select(population, k=len(population)-ELITE_SIZE)

        # Keep the top 20 elite individuals
        top_elite_individuals = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)[:ELITE_SIZE]
        new_population = top_elite_individuals + make_unique_population(best_offspring, k=len(population)-ELITE_SIZE)
        # Then ensure diversity
        population = make_unique_population(new_population, k=len(population))

        # Log best fitness of generation
        gen_fits = [ind.fitness.values[0] for ind in population]
        best_fitness.append(max(gen_fits))
        top_10 = sorted(gen_fits, reverse=True)[:10]
        print(f"Generation {gen}: Best Fitness = {best_fitness[-1]:.4f}----------------------------average of top 10 gen fits = {sum(top_10)/len(top_10):.4f}")
        print()

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
                'upheld_count': ind.fitness.values[2],
                'overturned_precision': ind.fitness.values[3],
                'overturned_count': ind.fitness.values[4],
                'total_matches_of_this_set': ind.fitness.values[5]
            })
        
        results_df = pd.DataFrame(results)
        return results_df

    results_df = analyze_results(population)
    print("\nTop Feature Combinations:")
    print(results_df)
    results_df.to_csv(filename, index=False, sep=',', encoding='utf-8')
    


# Plot fitness history
    # import matplotlib.pyplot as plt
    # plt.plot(best_fitness)
    # plt.title('Best Fitness Score by Generation')
    # plt.xlabel('Generation')
    # plt.ylabel('Fitness Score (Mutual Information)')
    # plt.show()