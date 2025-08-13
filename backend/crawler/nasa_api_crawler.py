"""
Crawler for NASA's API to fetch images and videos.
"""


import requests
import json
import os

API_URL = "https://images-api.nasa.gov/search"


def crawl_nasa_api(query: str = "space", max_results: int = 50, save_to_file: bool = False):
    """
    Fetch images and videos from NASA's API based on a query.
    
    Parameters:
        query (str): The search term to query NASA's API.
        max_results (int): The maximum number of results to return.
        save_to_file (bool): Whether to save the results to a JSON file.

    Returns:
        list: A list of dictionaries containing metadata about the retrieved images and videos.
    """

    # Set up the parameters for the API request
    parameters = {"q": query, "media_type": "image,video"}

    # Fetch data from the API
    response = requests.get(API_URL, params=parameters)
    data = response.json()

    # Check if the response is valid
    if response.status_code != 200 or "collection" not in data:
        print("Error fetching data from NASA API")
        return []
    
    # Process the results by extracting relevant fields from each item of the response
    results = []
    count = 0
    for item in data.get("collection", {}).get("items", []):
        # Stop if we have reached the maximum number of results
        if count >= max_results:
            break

        # Extract relevant fields from the item
        metadata = item.get("data", [])[0]
        links = item.get("links", [{}])

        # Add the extracted data to the results list
        results.append({
            "nasa_id": metadata.get("nasa_id", ""),
            "title": metadata.get("title", ""),
            "description": metadata.get("description", ""),
            "keywords": metadata.get("keywords", []),
            "center": metadata.get("center", ""),
            "media_type": metadata.get("media_type", ""),
            "date_created": metadata.get("date_created", ""),
            "location": metadata.get("location", ""),
            "photographer": metadata.get("photographer", ""),
            "secondary_creator": metadata.get("secondary_creator", ""),
            "album": metadata.get("album", []),
            "links": links,
        })

        count += 1
    
    # Optionally save the results to a JSON file
    if save_to_file:
        file_path = os.path.join("data", f"nasa_api_results_{query}.json")
        with open(file_path, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Results saved to {file_path}")

    # Return the list of results
    return results

if __name__ == "__main__":
    # Example usage
    crawl_nasa_api("moon", max_results=20, save_to_file=True)
