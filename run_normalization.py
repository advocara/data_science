

from term_normalizer import store_normalized_appeals

## Should not typically be necessary. Deleted old normalizations, and run scrape.py to create new ones and normalize them. 
## scrape.py will use imr_analyzer.py which will read from cache, so that's quick, and then will call this function.
store_normalized_appeals('Immuno Disorders-Lupus')