import json
# import pickle
# import shelve
import sqlite3


# def convert_index_to_pickle(inverted_index_path: str, pickle_path: str):
#     """
#     Convert the inverted index from a JSON file to a pickle file for faster loading.

#     Parameters:
#         inverted_index_path (str): Path to the JSON file containing the inverted index.
#         pickle_path (str): Path to the output pickle file.
#     """

#     with open(inverted_index_path, "r") as file:
#         inverted_index = json.load(file)

#     with open(pickle_path, "wb") as file:
#         pickle.dump(inverted_index, file)


# def convert_index_to_shelve_file(inverted_index_path: str, shelve_path: str):
#     """
#     Convert the inverted index from a pickle file to a shelve file for faster access.

#     Parameters:
#         inverted_index_path (str): Path to the pickle file containing the inverted index.
#         shelve_path (str): Path to the output shelve file.
#     """

#     with open(inverted_index_path, "rb") as file:
#         inverted_index = pickle.load(file)

#     with shelve.open(shelve_path, writeback=False) as db:
#         for term, postings in inverted_index.items():
#             db[term] = postings


# def convert_shelve_to_sqlite(shelve_path: str, sqlite_path: str):
#     """
#     Convert the inverted index from a shelve file to a SQLite database for efficient querying.

#     Parameters:
#         shelve_path (str): Path to the shelve file containing the inverted index.
#         sqlite_path (str): Path to the output SQLite database file.
#     """

#     with shelve.open(shelve_path, flag="r") as db:
#         conn = sqlite3.connect(sqlite_path)
#         cursor = conn.cursor()

#         # Create table for inverted index
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS inverted_index (
#                 token TEXT,
#                 doc_id TEXT,
#                 freq INTEGER,
#                 positions TEXT,
#                 PRIMARY KEY (token, doc_id)
#             )
#         """)

#         # Insert data into the table
#         for token in db:
#             for doc_id, info in db[token].items():
#                 positions_json = json.dumps(info["positions"])
#                 cursor.execute("INSERT OR REPLACE INTO inverted_index (token, doc_id, freq, positions) VALUES (?, ?, ?, ?)",
#                                (token, doc_id, info["freq"], positions_json))

#         conn.commit()
#         conn.close()


def convert_index_to_sqlite(inverted_index_path: str, sqlite_path: str):
    """
    Convert the inverted index from a JSON file to a SQLite database for efficient querying.

    Parameters:
        inverted_index_path (str): Path to the JSON file containing the inverted index.
        sqlite_path (str): Path to the output SQLite database file.
    """

    with open(inverted_index_path, "r") as file:
        inverted_index = json.load(file)

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Create table for inverted index
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inverted_index (
            token TEXT,
            doc_id TEXT,
            freq INTEGER,
            PRIMARY KEY (token, doc_id)
        )
    """)

    # Insert data into the table
    rows = []
    for token, postings in inverted_index.items():
        for doc_id, info in postings.items():
            rows.append((token, doc_id, info))

    cursor.executemany("INSERT OR REPLACE INTO inverted_index (token, doc_id, freq) VALUES (?, ?, ?)", rows)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_token ON inverted_index (token)")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    # convert_index_to_pickle("data/inverted_index.json", "data/inverted_index.pkl")
    # convert_index_to_shelve_file("data/inverted_index.pkl", "data/inverted_index_shelve")
    # convert_shelve_to_sqlite("data/inverted_index_shelve", "data/inverted_index.sqlite")
    convert_index_to_sqlite("data/inverted_index.json", "data/inverted_index.sqlite")