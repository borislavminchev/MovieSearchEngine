from elasticsearch import Elasticsearch, helpers
import pandas as pd
import numpy as np

es = Elasticsearch(
    "https://localhost:9200",
    ssl_assert_fingerprint="3ebae09247315629b286985054334a473f6be6a1094c564e0485233004d1d5d0",
    basic_auth=("elastic", "HO6XqXf8Z_-UtrCDhuZy")
    )

def document_count():
    c = es.count(index="movies")
    return c['count']

def get_document_by_id(id):
    query = {
        "query" : {
          "match": {
            "_id": id
          }
        }
    }
    response = es.search(index="movies", body=query)
    return response["hits"]["hits"][0]['_source']

def tf(word, document_id):
    search_content = get_document_by_id(document_id)['search_content']
    words = search_content.split()
    
    word_count = sum(1 for w in words if w == word)
    total_words = len(words)
    
    freq = word_count / total_words if total_words > 0 else 0

    return 1 + np.log10(freq)

def idf(word):
    query = {
        "query" : {
          "match": {
            "search_content": word
          }
        }
    }
    doc_freq = es.count(index="movies", body=query)['count']

    all_documents_count = document_count()
    return np.log10(all_documents_count / doc_freq)

def tfidf(word, document_id):
    return tf(word, document_id) * idf(word)


print(document_count())
print(tf("star", 72448))
print(idf("star"))
print(tfidf("star", 72448))

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

# query = {
#     "query": {
#         "match": {
#             "search_content": "star"
#         }
#     }
# }

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

# try:
#     response = es.search(index="movies", body=query_sorted)
#     print(response["hits"]['total']['value'])
#     for hit in response["hits"]["hits"]:
#         print(hit["_id"], hit["sort"][0],  " ----> ", hit["_source"]["search_content"].lower().count("star"))
# except Exception as e:
#     print(f"Error: {e}")