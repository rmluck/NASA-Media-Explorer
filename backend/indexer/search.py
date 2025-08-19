"""
Searches for a query in the indexed corpus and returns the top results.
"""


import json
import os
import math
from collections import Counter, defaultdict
from indexer import preprocess_text


def load_inverted_index(file_path: str) -> dict[str, dict[str, int]]:
    """
    Load the inverted index from a JSON file.

    Parameters:
        file_path (str): The path to the JSON file containing the inverted index.

    Returns:
        dict: The inverted index mapping terms to document IDs and their token frequencies.
    """

    # Loads the inverted index from a JSON file
    with open(file_path, "r") as file:
        inverted_index = json.load(file)

    # Return the loaded inverted index
    return inverted_index


def load_idf_scores(file_path: str) -> dict[str, float]:
    """
    Load the IDF scores from a JSON file.

    Parameters:
        file_path (str): The path to the JSON file containing the IDF scores.

    Returns:
        dict: A dictionary mapping tokens to their IDF scores.
    """

    # Loads the IDF scores from a JSON file
    with open(file_path, "r") as file:
        idf = json.load(file)

    # Return the loaded IDF scores
    return idf


def load_tf_idf_index(file_path: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """
    Load the TF-IDF index and document norms from JSON files.

    Parameters:
        file_path (str): The path to the JSON file containing the TF-IDF index.

    Returns:
        tuple: A tuple containing the TF-IDF index and document norms.
    """

    # Load the TF-IDF index from a JSON file
    with open(file_path, "r") as file:
        tf_idf_index = json.load(file)

    # Load the document norms from a JSON file
    norms_file_path = os.path.join("data", "doc_norms.json")
    with open(norms_file_path, "r") as file:
        doc_norms = json.load(file)

    # Return the loaded TF-IDF index and document norms
    return tf_idf_index, doc_norms


def score_query(query_terms: dict[str, int], inverted_index: dict[str, dict[str, int]], idf_scores: dict[str, float], tf_idf_index: dict[str, dict[str, float]], doc_norms: dict[str, float]) -> list[tuple[str, float]]:
    """
    Score the query against the TF-IDF index and return the ranked documents.

    Parameters:
        query_terms (dict): The preprocessed query terms and their frequencies.
        inverted_index (dict): The inverted index mapping terms to document IDs and their token frequencies.
        idf_scores (dict): The IDF scores for each term.
        tf_idf_index (dict): The TF-IDF index mapping document IDs to tokens and their TF-IDF scores.
        doc_norms (dict): The document norms mapping document IDs to their normalized lengths.
        
    Returns:
        list: The list of tuples containing document IDs and their scores, sorted by score in descending order.
    """

    # Compute the query tf-idf weights
    query_tf = Counter(query_terms)
    query_weights = {}
    for term, freq in query_tf.items():
        if term in idf_scores:
            query_weights[term] = (freq / len(query_terms)) * idf_scores[term]

    # Get the candidate documents from the inverted index
    candidate_docs = set()
    for term in query_weights.keys():
        if term in inverted_index:
            candidate_docs.update(inverted_index[term].keys())

    # Score the candidate documents using cosine similarity
    scores = defaultdict(float)
    for doc_id in candidate_docs:
        doc_weights = tf_idf_index[doc_id]
        score = sum(query_weights[term] * doc_weights.get(term, 0) for term in query_weights)
        scores[doc_id] = score / doc_norms[doc_id]

    # Sort the documents by their scores in descending order
    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # Return the ranked documents
    return ranked_docs


def search_query(query: str, inverted_index: dict[str, dict[str, int]], idf_scores: dict[str, float], tf_idf_index: dict[str, dict[str, float]], doc_norms: dict[str, float]) -> list[tuple[str, float]]:
    """
    Search for a query in the indexed corpus and return the top results.
    
    Parameters:
        query (str): The search query.
        inverted_index (dict): The inverted index mapping terms to document IDs and their token frequencies.
        idf_scores (dict): The IDF scores for each term.
        tf_idf_index (dict): The TF-IDF index mapping document IDs to tokens and their TF-IDF scores.
        doc_norms (dict): The document norms mapping document IDs to their normalized lengths.

    Returns:
        list: The list of tuples containing document IDs and their scores, sorted by score in descending order.
    """

    # Preprocess the query to create a list of tokens
    query_terms = preprocess_text(query)

    # Calculate the document scores for the query against the TF-IDF index
    ranked_docs = score_query(query_terms, inverted_index, idf_scores, tf_idf_index, doc_norms)

    # Return the ranked documents
    return ranked_docs


def get_doc_metadata(doc_id: str, doc_lookup: dict, metadata_cache: dict) -> dict:
    """
    Get the metadata for a document by its ID.

    Parameters:
        doc_id (str): The document ID.
        doc_lookup (dict): The document lookup information mapping document IDs to their metadata.
        metadata_cache (dict): A cache to store previously retrieved metadata.

    Returns:
        dict: The metadata for the document.
    """
        
    lookup_info = doc_lookup.get(doc_id, None)
    if lookup_info is None:
        return {"error": "Document not found"}
    year, index = lookup_info["year"], lookup_info["index"]

    # Check if the metadata is already cached
    if year not in metadata_cache:
        # Load the metadata for the document
        doc_metadata_file_path = os.path.join("data/nasa_full_corpus", f"nasa_data_{year}.json")
        with open(doc_metadata_file_path, "r") as file:
            metadata_cache[year] = json.load(file)
    
    # Return the metadata for the document
    return metadata_cache[year][doc_lookup[doc_id]["index"]]


if __name__ == "__main__":
    corpus_dir = "data"
    inverted_index = load_inverted_index(os.path.join(corpus_dir, "inverted_index.json"))
    idf_scores = load_idf_scores(os.path.join(corpus_dir, "idf_scores.json"))
    tf_idf_index, doc_norms = load_tf_idf_index(os.path.join(corpus_dir, "tf_idf_index.json"))

    query = "moon landing"
    ranked_docs = search_query(query, inverted_index, idf_scores, tf_idf_index, doc_norms)

    # Load the document lookup information from a JSON file
    doc_lookup_file_path = os.path.join("data", "doc_lookup.json")
    with open(doc_lookup_file_path, "r") as file:
        doc_lookup = json.load(file)

    metadata_cache = {}

    print(f"Top results for query '{query}':")
    for doc_id, score in ranked_docs[:10]:
        # Get metadata for the document
        doc_metadata = get_doc_metadata(doc_id, doc_lookup, metadata_cache)
        title = doc_metadata.get("title", "Unknown Title")

        print(f"Title: {title}, Score: {score:.4f}")