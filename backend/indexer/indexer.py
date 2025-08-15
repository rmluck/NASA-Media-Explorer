"""
Indexes the retrieved data from the NASA API to build a searchable index to more easily find images and videos based on metadata.
"""


import spacy
import json
import os
import math

# Load the spaCy model for text processing
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text: str) -> list[str]:
    """
    Preprocess the text by removing unnecessary characters and stopwords and normalizing it through tokenization and lemmatization.
    
    Parameters:
        text (str): The input text to preprocess.

    Returns:
        list[str]: The list of processed tokens.
    """

    doc = nlp(text)

    # Tokenize the text, remove stopwords and punctuation, convert to lowercase, strip whitespace, and lemmatize
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            lemma = token.lemma_.lower().strip()
            if lemma:
                tokens.append(lemma)

    # Return the list of processed tokens
    return tokens
        

def build_inverted_index(corpus_directory: str, save_to_file: bool = False) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """
    Build an inverted index from the provided data.

    Parameters:
        corpus_directory (str): The directory containing the JSON files with the data.
        save_to_file (bool): Whether to save the inverted index to a JSON file.

    Returns:
        dict: The inverted index mapping terms to document IDs.
        dict: The document lengths mapping document IDs to their lengths.
    """

    # Initialize the inverted index and document lengths
    inverted_index = {}
    doc_lengths = {}

    # Iterate through each JSON file in the corpus directory
    for filename in os.listdir(corpus_directory):
        if not filename.endswith(".json"):
            continue

        # Load the JSON data from the file
        file_path = os.path.join(corpus_directory, filename)
        with open(file_path, "r") as file:
            data = json.load(file)

        # Process each document in the JSON file
        for doc in data:
            # Retrieve the document ID
            doc_id = doc.get("nasa_id")
            if not doc_id:
                continue

            # Preprocess the text fields to create a tokenized representation of the document
            text = " ".join([
                doc.get("title", ""),
                doc.get("description", ""),
                " ".join(doc.get("keywords", [])),
                doc.get("center", ""),
                doc.get("location", ""),
                doc.get("photographer", ""),
                doc.get("secondary_creator", ""),
            ])
            tokens = preprocess_text(text)
            doc_lengths[doc_id] = len(tokens)

            # Update the inverted index with the tokens and their frequencies
            for token in tokens:
                if token not in inverted_index:
                    inverted_index[token] = {}
                if doc_id not in inverted_index[token]:
                    inverted_index[token][doc_id] = 0
                inverted_index[token][doc_id] += 1

    # Optionally save the inverted index to a JSON file
    if save_to_file:
        file_path = os.path.join("data", f"inverted_index.json")
        with open(file_path, "w") as file:
            json.dump(inverted_index, file, indent=2)
        print(f"Inverted index saved to {file_path}.")

    # Return the inverted index and document lengths
    return inverted_index, doc_lengths


def compute_idf(inverted_index: dict[str, dict[str, int]], doc_count: int) -> dict[str, float]:
    """
    Compute the Inverse Document Frequency (IDF) for each token in the inverted index.
    
    Parameters:
        inverted_index (dict): The inverted index mapping terms to document IDs and their token frequencies.
        doc_count (int): The total number of documents in the corpus.

    Returns:
        dict: A dictionary mapping tokens to their IDF scores.
    """

    # Initialize the inverse document frequency (IDF) dictionary
    idf = {}
    # Calculate the IDF for each token in the inverted index
    for token, doc_freqs in inverted_index.items():
        df = len(doc_freqs)
        idf[token] = math.log((1 + doc_count) / (1 + df)) if df > 0 else 0

    # Return the IDF dictionary
    return idf


def build_tf_idf_index(inverted_index: dict[str, dict[str, int]], doc_lengths: dict[str, int], idf: dict[str, float], save_to_file: bool = False) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """
    Build a TF-IDF index from the inverted index and document lengths.
    
    Parameters:
        inverted_index (dict): The inverted index mapping terms to document IDs and their token frequencies.
        doc_lengths (dict): The document lengths mapping document IDs to their lengths.
        idf (dict): The IDF scores for each term.
        save_to_file (bool): Whether to save the TF-IDF index to a JSON file.

    Returns:
        dict: The TF-IDF index mapping document IDs to tokens and their TF-IDF scores.
        dict: The dictionary mapping document IDs to their normalized lengths.
    """

    # Initialize the TF-IDF index and document norms
    tf_idf_index = {}
    doc_norms = {}

    # Calculate the TF-IDF scores for each document in the inverted index
    for doc_id in doc_lengths:
        tf_idf_index[doc_id] = {}
        norm_sq = 0
        for token, doc_freqs in inverted_index.items():
            if doc_id in doc_freqs:
                tf = doc_freqs[doc_id] / doc_lengths[doc_id]
                tf_idf = tf * idf[token]
                tf_idf_index[doc_id][token] = tf_idf
                norm_sq += tf_idf ** 2
        doc_norms[doc_id] = math.sqrt(norm_sq)

    # Optionally save the TF-IDF index to a JSON file
    if save_to_file:
        file_path = os.path.join("data", f"tf_idf_index.json")
        with open(file_path, "w") as file:
            json.dump(tf_idf_index, file, indent=2)
        print(f"TF-IDF index saved to {file_path}.")

    # Return the TF-IDF index and document norms
    return tf_idf_index, doc_norms


def score_query(query: str, tf_idf_index: dict[str, dict[str, float]], doc_norms: dict[str, float], idf: dict[str, float]) -> list[tuple[str, float]]:
    """
    Score the query against the TF-IDF index and return the ranked documents.

    Parameters:
        query (str): The search query.
        tf_idf_index (dict): The TF-IDF index mapping document IDs to tokens and their TF-IDF scores.
        doc_norms (dict): The document norms mapping document IDs to their normalized lengths.
        idf (dict): The IDF scores for each term.
        
    Returns:
        list: The list of tuples containing document IDs and their scores, sorted by score in descending order.
    """
    
    # Preprocess the query to create a list of tokens
    query_tokens = preprocess_text(query)

    # Create a term frequency (TF) vector for the query
    query_tf = {}
    for token in query_tokens:
        tf = query_tokens.count(token) / len(query_tokens)
        if token in idf:
            query_tf[token] = tf * idf[token]

    # Normalize the query vector
    query_norm = math.sqrt(sum(val ** 2 for val in query_tf.values()))
    if query_norm == 0:
        return []

    # Score each document based on the query using TF-IDF
    scores = {}
    for doc_id, tf_idf in tf_idf_index.items():
        dot = sum(tf_idf.get(token, 0.0) * weight for token, weight in query_tf.items())
        if doc_norms[doc_id] != 0:
            scores[doc_id] = dot / (doc_norms[doc_id] * query_norm)
        else:
            scores[doc_id] = 0.0
    
    # Sort the documents by their scores in descending order
    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # Return the ranked documents
    return ranked_docs


def search_query(corpus_dir: str, query: str):
    """
    Search for a query in the indexed corpus and return the top results.
    
    Parameters:
        corpus_dir (str): The directory containing the indexed data.
        query (str): The search query.

    Returns:
        list: A list of tuples containing document IDs and their scores.
    """

    # Build the inverted index and compute the IDF scores
    inverted_index, doc_lengths = build_inverted_index(corpus_dir, save_to_file=True)
    doc_count = len(doc_lengths)
    idf = compute_idf(inverted_index, doc_count)

    # Build the TF-IDF index
    tf_idf_index, doc_norms = build_tf_idf_index(inverted_index, doc_lengths, idf, save_to_file=True)

    ranked_docs = score_query(query, tf_idf_index, doc_norms, idf)

    # Return the ranked documents
    return ranked_docs


if __name__ == "__main__":
    corpus_dir = "data/nasa_api_results"
    query = "moon landing"
    ranked_docs = search_query(corpus_dir, query)

    print(f"Top results for query '{query}':")
    for doc_id, score in ranked_docs[:10]:
        print(f"Document ID: {doc_id}, Score: {score:.4f}")
