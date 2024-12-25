from elasticsearch import Elasticsearch, helpers
import pandas as pd

es = Elasticsearch(
    "https://localhost:9200",
    ssl_assert_fingerprint="3ebae09247315629b286985054334a473f6be6a1094c564e0485233004d1d5d0",
    basic_auth=("elastic", "HO6XqXf8Z_-UtrCDhuZy")
    )


df = pd.read_csv("./preprocessed_movies_clean.csv")

# def save_row(i, row):
#     print(i)
#     document = row.to_dict()
#     es.index(index="movies", id=document["id"], body=document)

# [save_row(i,row) for i, row in df.iterrows()]  

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

#
# query = {
#     "query": {
#         "multi_match": {
#             "query": "dream thriller",
#             "fields": ["title", "overview"]
#         }
#     }
# }

# query = {
#     "query": {
#         "term": {
#             "search_content": "science",
#         }
#     }
# }

# response = es.search(index="movies", body=query, size=3000)
# for hit in response["hits"]["hits"]:
#     print(hit["_source"]["title"], hit["_score"])

