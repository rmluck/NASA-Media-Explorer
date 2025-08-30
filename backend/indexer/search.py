"""
Searches for a query in the indexed corpus and returns the top results.
"""


import json
import os
# import gdown
import sqlite3
from collections import defaultdict, OrderedDict
from .indexer import preprocess_text

# Define weights and parameters for scoring
# PHRASE_WEIGHT = 2.0
BM25_K = 1.5
BM25_B = 0.75

# Define the data directory and paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
SQLITE_INDEX_PATH = os.path.join(DATA_DIR, "inverted_index.sqlite")

class LRUCache(OrderedDict):
    def __init__(self, max_size=100):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        if len(self) >= self.max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)


# def load_inverted_index(file_path: str) -> dict[str, dict[str, float]]:
#     """
#     Download the inverted index from a remote source.

#     Returns:
#         dict: The inverted index mapping terms to document IDs and their token frequencies.
#     """

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Inverted index not found at {file_path}. Please ensure the index file is present.")

#     with gzip.open(file_path, "rb") as file:
#         inverted_index = pickle.load(file)

#     return inverted_index


# def download_sqlite_index():
#     file_id = "1DTtPFdYiNzlHiQDEV30gzQG9ZANoICPp"
#     url = f"https://drive.google.com/uc?id={file_id}"

#     gdown.download(url, SQLITE_INDEX_PATH, quiet=False)


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


def load_doc_lengths(file_path: str) -> dict[str, float]:
    """
    Load the document lengths from a JSON file.

    Parameters:
        file_path (str): The path to the JSON file containing the document lengths.

    Returns:
        dict: A dictionary mapping document IDs to their lengths.
    """

    # Load the document lengths from a JSON file
    with open(file_path, "r") as file:
        doc_lengths = json.load(file)

    # Return the loaded document lengths
    return doc_lengths


def load_avg_doc_length(file_path: str) -> float:
    """
    Load the average document length from a JSON file.

    Parameters:
        file_path (str): The path to the JSON file containing the average document length.

    Returns:
        float: The average document length.
    """

    # Load the average document length from a JSON file
    with open(file_path, "r") as file:
        avg_doc_length = json.load(file)["avg_doc_length"]

    # Return the loaded average document length
    return avg_doc_length


def load_doc_lookup(file_path: str) -> dict[str, dict]:
    """
    Load the document lookup information from a JSON file.

    Parameters:
        file_path (str): The path to the JSON file containing the document lookup information.

    Returns:
        dict: A dictionary mapping document IDs to their metadata.
    """

    # Load the document lookup information from a JSON file
    with open(file_path, "r") as file:
        doc_lookup = json.load(file)

    # Return the loaded document lookup information
    return doc_lookup


def get_postings(conn: sqlite3.Connection, token: str) -> dict[str, int]:
    """
    Get the postings list for a token from the SQLite inverted index.
    
    Parameters:
        conn (sqlite3.Connection): The SQLite database connection.
        token (str): The token to look up.
    Returns:
        dict: A dictionary mapping document IDs to their token frequencies.
    """

    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, freq FROM inverted_index WHERE token = ?", (token,))
    return {doc_id: freq for doc_id, freq in cursor.fetchall()}


def score_query(query_tokens: list[str], conn: sqlite3.Connection, idf_scores: dict[str, float], doc_lengths: dict[str, float], avg_doc_length: float) -> list[tuple[str, float]]:
    """
    Score the query against the TF-IDF index and return the ranked documents.

    Parameters:
        query_tokens (list): The list of tokens in the query.
        conn (sqlite3.Connection): The SQLite database connection.
        idf_scores (dict): The IDF scores for each term.
        doc_lengths (dict): The document lengths mapping document IDs to their lengths.
        avg_doc_length (float): The average document length in the corpus.

    Returns:
        list: The list of tuples containing document IDs and their scores, sorted by score in descending order.
    """

    # Get the candidate documents from the inverted index
    candidate_docs = set()
    token_postings = {}
    for token in query_tokens:
        token_postings[token] = get_postings(conn, token)
        candidate_docs.update(token_postings[token].keys())

    # Score the candidate documents using cosine similarity
    scores = {}
    for doc_id in candidate_docs:
        doc_len = doc_lengths[doc_id]
        score = 0
        for token in query_tokens:
            freq = token_postings[token].get(doc_id, 0)
            idf = idf_scores.get(token, 0)
            score += idf * ((freq * (BM25_K + 1)) / (freq + BM25_K * (1 - BM25_B + BM25_B * (doc_len / avg_doc_length))))
            # score += PHRASE_WEIGHT * phrase_score(doc_id, query_tokens, inverted_index)
        scores[doc_id] += score

    # Sort the documents by their scores in descending order
    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # Return the ranked documents
    return ranked_docs


def phrase_score(doc_id: str, query_tokens: list[str], inverted_index: dict[str, dict[str, float]]) -> int:
    """
    Count how many times the exact phrase appears in the document.
    
    Parameters:
        doc_id (str): The document ID.
        query_tokens (list): The list of tokens in the query phrase.
        inverted_index (dict): The inverted index mapping terms to document IDs and their token positions.

    Returns:
        int: The count of the exact phrase occurrences in the document.
    """

    # If the query has less than 2 tokens, it cannot be a phrase
    if len(query_tokens) < 2:
        return 0
    
    # Get the positions of the first token in the query phrase
    first_token = query_tokens[0]
    if first_token not in inverted_index or doc_id not in inverted_index[first_token]:
        return 0
    
    # Get the positions of the first token in the document
    first_positions = inverted_index[first_token][doc_id]["positions"]

    # Iterate through the positions of the first token to find matching phrases
    phrase_count = 0
    for pos in first_positions:
        # Check if the subsequent tokens in the phrase match the positions
        match = True
        # Check for each subsequent token in the phrase
        for offset, token in enumerate(query_tokens[1:], start=1):
            if token not in inverted_index or doc_id not in inverted_index[token]:
                match = False
                break
            if pos + offset not in inverted_index[token][doc_id]["positions"]:
                match = False
                break
        
        # If all tokens matched in sequence, increment the phrase count
        if match:
            phrase_count += 1

    # Return the count of the exact phrase occurrences
    return phrase_count


def search_query(query: str, conn: sqlite3.Connection, idf_scores: dict[str, float], doc_lengths: dict[str, float], avg_doc_length: float) -> list[tuple[str, float]]:
    """
    Search for a query in the indexed corpus and return the top results.
    
    Parameters:
        query (str): The search query.
        conn (sqlite3.Connection): The SQLite database connection.
        idf_scores (dict): The IDF scores for each term.
        doc_lengths (dict): The document lengths mapping document IDs to their lengths.
        avg_doc_length (float): The average document length in the corpus.

    Returns:
        list: The list of tuples containing document IDs and their scores, sorted by score in descending order.
    """

    # Preprocess the query to create a list of tokens
    query_terms = preprocess_text(query)

    # Calculate the document scores for the query
    ranked_docs = score_query(query_terms, conn, idf_scores, doc_lengths, avg_doc_length)

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
        
    # Check if the document ID exists in the lookup
    lookup_info = doc_lookup.get(doc_id, None)
    if not lookup_info:
        return {"error": "Document not found"}
    year, index = lookup_info["year"], lookup_info["index"]

    # Check if the metadata is already cached
    if year not in metadata_cache:
        # Load the metadata for the document
        doc_metadata_file_path = os.path.join(DATA_DIR, "nasa_full_corpus", f"nasa_data_{year}.json")
        with open(doc_metadata_file_path, "r") as file:
            metadata_cache[year] = [json.loads(line) for line in file if line.strip()]

    # Return the metadata for the document
    return metadata_cache[year][doc_lookup[doc_id]["index"]]


if __name__ == "__main__":
    # Load the index data
    idf_scores = load_idf_scores(os.path.join(DATA_DIR, "idf_scores.json"))
    # tf_idf_index, doc_norms = load_tf_idf_index(os.path.join(corpus_dir, "tf_idf_index.json"))
    doc_lengths = load_doc_lengths(os.path.join(DATA_DIR, "doc_lengths.json"))
    avg_doc_length = load_avg_doc_length(os.path.join(DATA_DIR, "avg_doc_length.json"))

    # Connect to the SQLite database
    # if not os.path.exists(SQLITE_INDEX_PATH):
    #     print("Downloading SQLite inverted index...")
    #     download_sqlite_index()
    #     print("Download complete.")
    conn = sqlite3.connect(SQLITE_INDEX_PATH)

    # # Run a sample query
    # query = "Apollo 11"
    # ranked_docs = search_query(query, conn, idf_scores, doc_lengths, avg_doc_length)

    # # Load the document lookup information from a JSON file
    # doc_lookup = load_doc_lookup(os.path.join(DATA_DIR, "doc_lookup.json"))

    # # Print the top 20 results with their titles and scores
    # metadata_cache = {}
    # for doc_id, score in ranked_docs[:20]:
    #     # Get metadata for the document
    #     doc_metadata = get_doc_metadata(doc_id, doc_lookup, metadata_cache)
    #     title = doc_metadata.get("title", "Unknown Title")

    #     print(f"Title: {title}, Score: {score:.4f}")

    conn.close()