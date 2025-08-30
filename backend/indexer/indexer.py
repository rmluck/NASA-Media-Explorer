"""
Indexes the retrieved data from the NASA API to build a searchable index to more easily find images and videos based on metadata.
"""


import spacy
import json
import os
import math
import re

# Define domain-specific stopwords
DOMAIN_STOPWORDS = {"nasa", "nasa_id", "nasa_url", "media_type", "title", "description", "keywords", "location", "date_created", "image", "video", "audio", "thumbnail", "media", "photo", "photograph", "space", "nasa", "center", "science"}

# Define general NASA-related patterns to preserve during text preprocessing
GENERAL_NASA_PATTERNS = {
    r"\b[A-Z][a-zA-Z]+[- ]?\d+\b",  # Matches patterns like "Apollo 11", "Voyager-1"
    r"\b[A-Z]{2,}-?\d*\b",      # Matches patterns like "ISS-123", "HST-456"
    r"\b[A-Z]{2,}\b",          # Matches patterns like "NASA", "ESA"
    r"\b\d{4}\b",               # Matches years like "1969", "2020"
    r"\b\d{3,}\b",              # Matches any number with 3 or more digits
}

# Define field weights for indexing
FIELD_WEIGHTS = {
    "title": 3.0,
    "keywords": 2.0,
    "description": 1.0,
    "location": 0.5
}

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

    # Find and preserve specific NASA-related patterns before general preprocessing
    preserved_tokens = []
    for pattern in GENERAL_NASA_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        preserved_tokens.extend([match.lower() for match in matches])
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    doc = nlp(text)

    # Tokenize the text, remove stopwords and punctuation, convert to lowercase, strip whitespace, and lemmatize (if not a proper noun)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            if token.pos_ == "PROPN":
                lemma = token.text.lower().strip()
            else:
                lemma = token.lemma_.lower().strip()
            if lemma and len(lemma) > 2 and lemma not in DOMAIN_STOPWORDS:
                tokens.append(lemma)

    # Return the list of processed tokens
    return preserved_tokens + tokens
        

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

    # Initialize the inverted index, document lengths, and document lookup information
    inverted_index = {}
    doc_lengths = {}
    doc_lookup = {}

    # Iterate through each JSON file in the corpus directory
    for year in range(1920, 2025):
        filename = f"nasa_data_{year}.json"
        file_path = os.path.join(corpus_directory, filename)
        if not os.path.exists(file_path):
            print(f"File {filename} does not exist in the directory {corpus_directory}. Skipping...")
            continue

        print(f"Processing file: {filename}...")
        # Load the JSON data from the file
        docs = []
        with open(file_path, "r") as file:
            docs = [json.loads(line) for line in file if line.strip()]

        # Process each document in the JSON file
        for i, doc in enumerate(docs):
            # Retrieve the document ID
            doc_id = doc.get("nasa_id")
            if not doc_id:
                continue

            # Store the document ID in the lookup dictionary
            doc_lookup[doc_id] = {
                "year": year,
                "index": i
            }

            # Initialize the document length
            doc_length = 0

            # Process each field separately with weighting
            for field, weight in FIELD_WEIGHTS.items():
                if field in doc:
                    # Preprocess the text field to create a tokenized representation of the document
                    text = doc[field]
                    if isinstance(text, list):
                        text = " ".join(text)
                    tokens = preprocess_text(text)

                    for token in tokens:
                        if token not in inverted_index:
                            inverted_index[token] = {}
                        if doc_id not in inverted_index[token]:
                            inverted_index[token][doc_id] = 0
                        inverted_index[token][doc_id] += weight

                    # Update the document length
                    doc_length += len(tokens)

            # Store the document length
            doc_lengths[doc_id] = doc_length

    # Filter out rare tokens that appear in fewer than 2 documents
    rare_threshold = 2
    inverted_index = {token: docs for token, docs in inverted_index.items() if len(docs) >= rare_threshold}

    # Calculate and store the average document length
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)        

    # Optionally save the inverted index, document lookup information, document lengths, and average document length to JSON files
    if save_to_file:
        file_path = os.path.join("data", "inverted_index.json")
        with open(file_path, "w") as file:
            json.dump(inverted_index, file, indent=2)
        print(f"Inverted index saved to {file_path}.")

        file_path = os.path.join("data", "doc_lookup.json")
        with open(file_path, "w") as file:
            json.dump(doc_lookup, file, indent=2)
        print(f"Document lookup information saved to {file_path}.")

        file_path = os.path.join("data", "doc_lengths.json")
        with open(file_path, "w") as file:
            json.dump(doc_lengths, file, indent=2)
        print(f"Document lengths saved to {file_path}.")

        file_path = os.path.join("data", "avg_doc_length.json")
        with open(file_path, "w") as file:
            json.dump({"avg_doc_length": avg_doc_length}, file)
        print(f"Average document length saved to {file_path}.")

    # Return the inverted index and document lengths
    return inverted_index, doc_lengths


def compute_idf(inverted_index: dict[str, dict[str, int]], doc_count: int, save_to_file: bool = False) -> dict[str, float]:
    """
    Compute the Inverse Document Frequency (IDF) for each token in the inverted index.
    
    Parameters:
        inverted_index (dict): The inverted index mapping terms to document IDs and their token frequencies.
        doc_count (int): The total number of documents in the corpus.
        save_to_file (bool): Whether to save the IDF scores to a JSON file.

    Returns:
        dict: A dictionary mapping tokens to their IDF scores.
    """

    # Initialize the inverse document frequency (IDF) dictionary
    idf_scores = {}
    # Calculate the IDF for each token in the inverted index
    for token, doc_freqs in inverted_index.items():
        df = len(doc_freqs)
        idf_scores[token] = math.log((doc_count - df + 0.5) / (df + 0.5) + 1) if df > 0 else 0

    # Optionally save the inverted index to a JSON file
    if save_to_file:
        file_path = os.path.join("data", f"idf_scores.json")
        with open(file_path, "w") as file:
            json.dump(idf_scores, file, indent=2)
        print(f"IDF scores saved to {file_path}.")

    # Return the IDF dictionary
    return idf_scores


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
                tf = len(doc_freqs[doc_id]) / doc_lengths[doc_id]
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
        file_path = os.path.join("data", f"doc_norms.json")
        with open(file_path, "w") as file:
            json.dump(doc_norms, file, indent=2)
        print(f"Document norms saved to {file_path}.")

    # Return the TF-IDF index and document norms
    return tf_idf_index, doc_norms


def build_index(corpus_dir: str):
    """
    Build the index for the corpus directory and save the inverted index, IDF scores, TF-IDF index, and document norms to file.
    
    Parameters:
        corpus_dir (str): The directory containing the corpus data.
    """

    # Build the inverted index
    print("Building inverted index...")
    inverted_index, doc_lengths = build_inverted_index(corpus_dir, save_to_file=True)
    print("Inverted index built successfully.")

    # Compute the IDF scores
    doc_count = len(doc_lengths)
    print("Computing IDF scores...")
    idf_scores = compute_idf(inverted_index, doc_count, save_to_file=True)
    print("IDF scores computed successfully.")

    # Build the TF-IDF index
    # print("Building TF-IDF index...")
    # tf_idf_index, doc_norms = build_tf_idf_index(inverted_index, doc_lengths, idf_scores, save_to_file=True)
    # print("TF-IDF index built successfully.")


if __name__ == "__main__":
    # Build the index for the corpus directory
    corpus_dir = "data/nasa_full_corpus"
    build_index(corpus_dir)