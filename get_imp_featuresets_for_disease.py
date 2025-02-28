import os
from model.config import IMRConfig, IMRQuery

from scrape import gen_cache
from term_normalizer import store_normalized_appeals
from predict import predict


def main():
    # Create configuration
    disease = "Eczema"
    config = IMRConfig(
        query=IMRQuery(diagnosis_subcategory=disease), # can restrict and alter while keeping same dataset name
        dataset_name=disease, # this drives caching, so if you change it, all prior records will be re-parsed, re-normalized
        method="deap",
        analyze_treatments_conditions=True,
        start_record=0,
        max_records=200,
        # Using default values for other parameters:
        # start_record=0
        # max_records=37000
        # input_csv="data/ca-imr-determinations.csv"
    )

    # Generate cache files based on input query
    gen_cache(config.input_csv, config)

    # Normalize results
    store_normalized_appeals(config.query)

    # Get feature importances
    predict(config)


if __name__ == "__main__":
    main() 