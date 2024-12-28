from elasticsearch import Elasticsearch, helpers
import pandas as pd
import numpy as np

es = Elasticsearch(
    "https://localhost:9200",
    ssl_assert_fingerprint="3ebae09247315629b286985054334a473f6be6a1094c564e0485233004d1d5d0",
    basic_auth=("elastic", "HO6XqXf8Z_-UtrCDhuZy")
    )


df = pd.read_csv("./preprocessed_movies_clean.csv")


actions = [
    {
        "_index": "movies",
        "_id": doc["id"],
        "_source": doc
    } 
    for doc in df.to_dict(orient="records")
]

try:
    helpers.bulk(es, actions)
except helpers.BulkIndexError as e:
    print(f"Bulk indexing error: {e}")
    print("Failed documents:")
    for error in e.errors:
        print(error)
