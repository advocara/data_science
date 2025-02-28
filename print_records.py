from imr_analyzer import IMRAnalyzer, IMRQuery


analyzer = IMRAnalyzer("./data/ca-imr-determinations.csv", None)
query = IMRQuery(diagnosis_category="Immuno Disorders", diagnosis_subcategory="Lupus")
analyzer.print_records_to_markdown(query, n=10)
