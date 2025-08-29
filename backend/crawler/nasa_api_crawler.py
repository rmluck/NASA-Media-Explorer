"""
Crawler for NASA's API to fetch images and videos.
"""


import requests
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the NASA API URL
API_URL = "https://images-api.nasa.gov/search"

# Output directory for the crawled data
OUTPUT_DIR = "data/nasa_full_corpus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Progress file to track the last crawled year
PROGRESS_FILE = "data/nasa_crawl_progress.json"


def crawl_nasa_api_by_year(start_year: int = 1920, end_year: int = 2025, delay: int = 1, save_per_year: bool = True):
    """
    Crawl the NASA API incrementally by year and save results.
    
    Parameters:
        start_year (int): The starting year for the crawl.
        end_year (int): The ending year for the crawl.
        delay (int): Delay in seconds between API requests to avoid rate limiting.
        save_per_year (bool): Whether to save results per year or in a single file.
    """

    # Initialize a list to hold the results
    results = []

    # Iterate through each year in the specified range
    for year in range(start_year, end_year + 1):
        print(f"Crawling year {year}...")
        year_results = []
        page = 1

        # Fetch data from the API for each page of the specified year
        while True:
            # Set up the parameters for the API request
            params = {
                "media_type": "image,video",
                "year_start": year,
                "year_end": year,
                "page": page
            }

            # Fetch data from the API
            response = requests.get(API_URL, params=params)
            if response.status_code != 200:
                print(f"Error fetching year {year}, page {page}: {response.status_code}")
                break

            # Parse the JSON response
            data = response.json()
            items = data.get("collection", {}).get("items", [])
            if not items:
                save_progress(year)
                break

            # Process each item in the response
            for item in items:
                metadata = item.get("data", [])[0]
                collection = requests.get(item.get("href", ""))
                if not collection:
                    continue
                collection = collection.json()
                asset = [link for link in collection if "~orig" in link][0]
                thumbnail = [link for link in collection if "~thumb" in link][-1]
                year_results.append({
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
                    "asset": asset,
                    "thumbnail": thumbnail
                })

            print(f"   Retrieved {len(items)} items from year {year}, page {page}.")

            # Save progress after each page
            save_progress(year, page)

            page += 1

            # Delay to avoid hitting the API rate limit
            time.sleep(delay)
        
        # Save results for the year if requested
        if save_per_year and year_results:
            year_file = os.path.join(OUTPUT_DIR, f"nasa_data_{year}.json")
            with open(year_file, "a") as file:
                for item in year_results:
                    file.write(json.dumps(item) + "\n")
            print(f"Saved {len(year_results)} items for year {year} to {year_file}.")

        results.extend(year_results)

    # Save all results to a single file if not saving per year
    if not save_per_year and results:
        merged_file = os.path.join(OUTPUT_DIR, "nasa_data_merged.json")
        with open(merged_file, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Saved {len(results)} total items to {merged_file}.")


def save_progress(year: int):
    """
    Save the progress of the crawl to a file.
    
    Parameters:
        year (int): The year of the last crawled data.
    """

    with open(PROGRESS_FILE, "w") as file:
        json.dump({"year": year}, file)
    print(f"Progress saved: Year {year}.")


def load_progress() -> dict:
    """
    Load the progress of the crawl from a file.

    Returns:
        dict: A dictionary containing the last crawled year.
    """

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as file:
            return json.load(file)

    return {"year": None}


if __name__ == "__main__":
    # Load the last crawled year from the progress file
    progress = load_progress()
    start_year = progress["year"] or 1920

    # Crawl everything from 1920 to 2025 with a delay of 1 second between requests
    crawl_nasa_api_by_year(start_year=start_year, end_year=2025, delay=1, save_per_year=True)
    print("Crawling completed.")