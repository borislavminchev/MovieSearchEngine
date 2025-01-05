from elasticsearch import Elasticsearch, helpers
import pandas as pd
import numpy as np

es = Elasticsearch(
    "https://localhost:9200",
    ssl_assert_fingerprint="3ebae09247315629b286985054334a473f6be6a1094c564e0485233004d1d5d0",
    basic_auth=("elastic", "HO6XqXf8Z_-UtrCDhuZy")
    )


# df = pd.read_csv("./preprocessed_movies_clean.csv")


# actions = [
#     {
#         "_index": "movies",
#         "_id": doc["id"],
#         "_source": doc
#     } 
#     for doc in df.to_dict(orient="records")
# ]

# try:
#     helpers.bulk(es, actions)
# except helpers.BulkIndexError as e:
#     print(f"Bulk indexing error: {e}")
#     print("Failed documents:")
#     for error in e.errors:
#         print(error)

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
#         "query_string" : {
#                  "query" : "star wars",
#                  "fuzziness":0
#           }
#     }
# }

query = {
    "query": {
        "match": {
            "search_content": "star"
        }
    }
}

# query_sorted = {
#   "_source": ["search_content"],
#     "query": {
#         "match": {
#             "search_content": "star"
#         }
#     },
#   "sort": [
#     {
#       "_script": {
#         "type": "number",
#         "script": {
#           "source": "String target = params.target; int count = 0; if (doc['search_content.raw'].size() > 0) { String text = doc['search_content.raw'].value; int index = 0; while ((index = text.indexOf(target, index)) != -1) { count++; index += target.length(); } } return count;",
#           "params": {
#             "target": "star"
#           }
#         },
#         "order": "desc"
#       }
#     }
#   ]
# }

try:
    response = es.search(index="movies", body=query)
    print(response["hits"]['total']['value'])
    for hit in response["hits"]["hits"]:
        print(hit["_source"]['title'], hit['_score'] )
except Exception as e:
    print(f"Error: {e}")
