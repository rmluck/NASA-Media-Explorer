"""
Searches for a query in the indexed corpus and returns the top results.
"""


import json
import os
import pickle
import gdown
from collections import defaultdict
from .indexer import preprocess_text

# Define weights and parameters for scoring
PHRASE_WEIGHT = 2.0
BM25_K = 1.5
BM25_B = 0.75

# Define the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

# Define the URL for downloading large index files
INVERTED_INDEX_URL = "https://drive.google.com/uc?export=download&id=1H704CQ5xR9JBiGbRj-aGXGw1EGQqJ1TO"

# Define the path to the local inverted index file
INVERTED_INDEX_PATH = "/tmp/inverted_index.pkl"


def download_inverted_index() -> dict[str, dict[str, list[int]]]:
    """
    Download the inverted index from a remote source.
    """

    if not os.path.exists(INVERTED_INDEX_PATH):
        print("Downloading inverted index from Google Drive...")
        gdown.download(INVERTED_INDEX_URL, INVERTED_INDEX_PATH, quiet=False)
        print("Download complete.")

    # Load the downloaded inverted index from the pickle file
    with open(INVERTED_INDEX_PATH, "rb") as file:
        inverted_index = pickle.load(file)

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


def score_query(query_tokens: list[str], inverted_index: dict[str, dict[str, list[int]]], idf_scores: dict[str, float], doc_lengths: dict[str, float], avg_doc_length: float) -> list[tuple[str, float]]:
    """
    Score the query against the TF-IDF index and return the ranked documents.

    Parameters:
        query_tokens (list): The list of tokens in the query.
        inverted_index (dict): The inverted index mapping terms to document IDs and their token frequencies.
        idf_scores (dict): The IDF scores for each term.
        doc_lengths (dict): The document lengths mapping document IDs to their lengths.
        avg_doc_length (float): The average document length in the corpus.

    Returns:
        list: The list of tuples containing document IDs and their scores, sorted by score in descending order.
    """

    # Get the candidate documents from the inverted index
    candidate_docs = set()
    for token in query_tokens:
        if token in inverted_index:
            candidate_docs.update(inverted_index[token].keys())

    # Score the candidate documents using cosine similarity
    scores = defaultdict(float)
    for doc_id in candidate_docs:
        doc_len = doc_lengths[doc_id]
        score = 0
        for token in query_tokens:
            if token in inverted_index and doc_id in inverted_index[token]:
                freq = inverted_index[token][doc_id]["freq"]
                idf = idf_scores.get(token, 0)
                score += idf * ((freq * (BM25_K + 1)) / (freq + BM25_K * (1 - BM25_B + BM25_B * (doc_len / avg_doc_length))))
        score += PHRASE_WEIGHT * phrase_score(doc_id, query_tokens, inverted_index)
        scores[doc_id] += score

    # Sort the documents by their scores in descending order
    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # Return the ranked documents
    return ranked_docs


def phrase_score(doc_id: str, query_tokens: list[str], inverted_index: dict[str, dict[str, list[int]]]) -> int:
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


def search_query(query: str, inverted_index: dict[str, dict[str, list[int]]], idf_scores: dict[str, float], doc_lengths: dict[str, float], avg_doc_length: float) -> list[tuple[str, float]]:
    """
    Search for a query in the indexed corpus and return the top results.
    
    Parameters:
        query (str): The search query.
        inverted_index (dict): The inverted index mapping terms to document IDs and their token frequencies.
        idf_scores (dict): The IDF scores for each term.
        doc_lengths (dict): The document lengths mapping document IDs to their lengths.
        avg_doc_length (float): The average document length in the corpus.

    Returns:
        list: The list of tuples containing document IDs and their scores, sorted by score in descending order.
    """

    # Preprocess the query to create a list of tokens
    query_terms = preprocess_text(query)

    # Calculate the document scores for the query against the TF-IDF index
    ranked_docs = score_query(query_terms, inverted_index, idf_scores, doc_lengths, avg_doc_length)

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
    if lookup_info is None:
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
    index_dir = "data"
    inverted_index = download_inverted_index(os.path.join(index_dir, "inverted_index.json"))
    idf_scores = load_idf_scores(os.path.join(index_dir, "idf_scores.json"))
    # tf_idf_index, doc_norms = load_tf_idf_index(os.path.join(corpus_dir, "tf_idf_index.json"))
    doc_lengths = load_doc_lengths(os.path.join(index_dir, "doc_lengths.json"))
    avg_doc_length = load_avg_doc_length(os.path.join(index_dir, "avg_doc_length.json"))

    # Run a sample query
    query = "Apollo 11"
    ranked_docs = search_query(query, inverted_index, idf_scores,  doc_lengths, avg_doc_length)

    # Load the document lookup information from a JSON file
    doc_lookup = load_doc_lookup(os.path.join(index_dir, "doc_lookup.json"))

    # Print the top 20 results with their titles and scores
    metadata_cache = {}
    for doc_id, score in ranked_docs[:20]:
        # Get metadata for the document
        doc_metadata = get_doc_metadata(doc_id, doc_lookup, metadata_cache)
        title = doc_metadata.get("title", "Unknown Title")

        print(f"Title: {title}, Score: {score:.4f}")