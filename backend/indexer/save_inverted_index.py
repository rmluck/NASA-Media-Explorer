"""
Saves inverted index.
"""

import json
import os
import pickle


def save_inverted_index():
    """
    Converts inverted index from JSON to pickle format for faster loading.
    """
    with open(os.path.join("data", "inverted_index.json"), "r") as file:
        inverted_index = json.load(file)

    with open(os.path.join("data", "inverted_index.pkl"), "wb") as file:
        pickle.dump(inverted_index, file)


if __name__ == "__main__":
    save_inverted_index()