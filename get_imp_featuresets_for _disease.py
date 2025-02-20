from imr_analyzer import IMRQuery

from scrape import gen_cache
from term_normalizer import store_normalized_appeals
from predict import predict


# Configure parameters
input_csv = "data\ca-imr-determinations.csv"
start_record = 0  # Start from first record
max_records = 37000  # Process limit of records
# chunk_size = 4    # Process in chunks of 4

#Define query
query = IMRQuery(
    diagnosis_category="Cancer",
    diagnosis_subcategory="Lung"
    #Leave empty to tackle all categories and/or sub-categories
)

#Generate cache files based on input query
gen_cache(input_csv, query, start_record, max_records)

#Nomalize results
store_normalized_appeals(query)


#Feature importance methodology
method = "deap" #random_forest/FPgrowth/deap

predict(query, method) # Provide parameter result_file_name: str, when using deap to name results file