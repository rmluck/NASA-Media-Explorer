import pickle
import shelve


def convert_index_to_shelve_file(inverted_index_path: str, shelve_path: str):
    """
    Convert the inverted index from a pickle file to a shelve file for faster access.

    Parameters:
        inverted_index_path (str): Path to the pickle file containing the inverted index.
        shelve_path (str): Path to the output shelve file.
    """

    with open(inverted_index_path, "rb") as file:
        inverted_index = pickle.load(file)

    with shelve.open(shelve_path, writeback=False) as db:
        for term, postings in inverted_index.items():
            db[term] = postings