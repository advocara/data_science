## Data Science Project

### CA IMR Analysis

California has good data on their (external) insurance appeals, submitted to the CA DHMC after internal insurance appeal was denied. 

#### Data
* The raw data is in the `data/ca-imr-determinations.csv` file.
* The JSON Appeals object JSON is in the `cache` directory with a hash of the text + disease category as the filename. 
   * The MedicalInsuranceAppeal object is structured to capture information we think impacts appeal outcomes, based on a read of the appeals (by a human / Damon). This information includes whether guidlines or standards of care were met, what the symptoms were, what the treatments were, etc. 
* A normalized version of the Appeals data with terms normalized (treatment names, conditions, symptoms) is in the `appeals-results` directory with the -norm.json suffix. 
* We also turn the Appeals objects into a pandas dataframe in-memory for analysis and do not store it in any directory. 

Note that NY IMR data is in various `data/ny-<disease>.csv` files and they are not yet used. 

### Human review
Imr_analyzer.py has a function to print some CSV row appeals to a markdown file, which is useful for human review of cases. 

#### Analysis

* The `scrape.py` reads the CSV file (originally downloaded from the CA DHMC website) and converts it to the Appeal object format as JSON.
   * This includes gathering certain structured fields from the CSV, and also extracting information from the text of the appeal (e.g., symptoms, conditions, treatments). 
* The `run_normalization.py` script normalizes the data by gathering all the terms from all appeals in the cache (matching some category/disease name) and asking an LLM to map the terms to a standard set of terms, then rewrites the Appeals JSON.
* `predict.py` is the main script for training and deriving insights from the model. 
    * `predict.py` has three methods:
        * DEAP (a genetic algorithm) to create a model that predicts appeal success based on Appeals data. We think this is the best since it explicitly searches for the best feature combinations vs trying to extract insight from internal model structures of decision trees, which may not be suitable.
        * RandomForests (collections of decision trees) to create a model that predicts appeal success based on Appeals data. 
        * FPgrowth (a frequent pattern mining algorithm) to create a model that predicts appeal success based on Appeals data. 
    * But we don't predict success. Instead `predict.py` also does Feature Impact Analysis to understand which features are most important in predicting appeal success. This is the overall goal of this project, to understand which features are most important to appeal success. 
* Inputs to predict.py are used to run on segments of the data.
    * `analyze_treatments_conditions` - if true, focus on treatments and conditions. This is useful to understand which treatments and conditions should be tried or mentione in appeals, particularly excluding standards features which are highly correlated with appeal success.
    * `consolidate_standards` - if true, consolidate standards of care and guidelines into a single flag. This is useful to understand which standards are most important to appeal success.
    * split_data - if true, split the data into three subsets also to avoid skew due to the standards correlation.

* there is (currently) some dead or unused code, such as an aborted predict_tpot.py script that tried to use a different model, and perhaps some LLM based summarization vs the parse, normalize and predict approach. 

## Running the code

Set the query and desired prediction configuration in `get_imp_featuresets_for_disease.py` and run it.
* the query determines which disease or other subset to analyze

```
get_imp_featuresets_for_disease.py
```



### Algorithm

A key output metric is "impact", which is the absolute difference between the denial rate of a feature path and the overall denial rate.
`impact = abs(deny_rate - baseline_denial_rate)`

Where:

    deny_rate = `stats['deny'] / stats['count']` → Proportion of cases that were denied among cases with this feature path.
    `baseline_denial_rate = total_denials / total_samples` → The overall rate of denials across all appeals.
    impact is the absolute difference between these two rates.

E.g. if the overall denial rate is 30% and the combination of `age_range_41 to 50 + treatment_subcategory_Occupational Therapy + treatments_requested_Glatiramer Acetate` has a denial rate of 60%, then the impact is 30%. 

Impact as defined here is always bad for the consumer - higher percentages mean more denials over the baseline. 

