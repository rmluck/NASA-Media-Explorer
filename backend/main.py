"""
Main entry point for the NASA Media Explorer backend.
"""


import os
import httpx
import json
import pytz
from urllib.parse import quote
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from indexer.search import download_inverted_index, load_doc_lengths, load_idf_scores, load_doc_lookup, load_avg_doc_length, search_query, get_doc_metadata

INDEX_DIR = "../data"
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")


# Load index and metadata once on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inverted_index = download_inverted_index()
    app.state.idf_scores = load_idf_scores(os.path.join(INDEX_DIR, "idf_scores.json"))
    app.state.doc_lengths = load_doc_lengths(os.path.join(INDEX_DIR, "doc_lengths.json"))
    app.state.avg_doc_length = load_avg_doc_length(os.path.join(INDEX_DIR, "avg_doc_length.json"))
    app.state.doc_lookup = load_doc_lookup(os.path.join(INDEX_DIR, "doc_lookup.json"))
    app.state.metadata_cache = {}

    yield

# Create FastAPI app instance
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="../frontend/templates")


@app.get("/")
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("home.html", {"request": request, "year": datetime.now().year})


@app.get("/search", response_class=HTMLResponse)
def search_results(request: Request, query: str = Query(...), limit: int = 20, offset: int = 0) -> HTMLResponse:
    # Run the search query using the loaded index and metadata
    results = search_query(query, app.state.inverted_index, app.state.idf_scores, app.state.doc_lengths, app.state.avg_doc_length)

    # Bound the results by limit and offset
    total_results = len(results)
    paged_results = results[offset:offset + limit]

    # Fetch document metadata for the results
    top_results = []
    for doc_id, score in paged_results:
        metadata = get_doc_metadata(doc_id, app.state.doc_lookup, app.state.metadata_cache)
        
        raw_asset = metadata.get("asset", "").replace("http://", "https://")
        raw_thumbnail = metadata.get("thumbnail", "").replace("http://", "https://")
        asset = quote(raw_asset, safe=":/~")
        thumbnail = quote(raw_thumbnail, safe=":/~")

        date = metadata.get("date_created", "Unknown Date")
        if date != "Unknown Date" and len(date) >= 10:
            try:
                parsed_date = datetime.strptime(date[:10], "%Y-%m-%d")
                date = parsed_date.strftime("%B %d, %Y")
            except ValueError:
                pass
        
        top_results.append({
            "doc_id": doc_id,
            "media_type": metadata.get("media_type", "Unknown"),
            "title": metadata.get("title", "Unknown Title"),
            "description": metadata.get("description", "No description available"),
            "keywords": metadata.get("keywords", []),
            "center": metadata.get("center", "Unknown Center"),
            "date_created": date,
            "location": metadata.get("location", "Unknown Location"),
            "photographer": metadata.get("photographer", "Unknown Photographer"),
            "asset": asset,
            "thumbnail": thumbnail,
            "score": score,
        })

    return templates.TemplateResponse(
        "search_results.html",
        {
            "request": request,
            "results": top_results,
            "query": query,
            "limit": limit,
            "offset": offset,
            "total_results": total_results,
            "year": datetime.now().year
        }
    )

@app.get("/api/search")
def search_api(query: str = Query(...), limit: int = 20, offset: int = 0, start_year: int = 1920, end_year: int = 2025, media_type: str = None):
    # Run the search query using the loaded index and metadata
    results = search_query(query, app.state.inverted_index, app.state.idf_scores, app.state.doc_lengths, app.state.avg_doc_length)

    # Fetch document metadata for the results
    filtered_results = []
    for doc_id, score in results:
        metadata = get_doc_metadata(doc_id, app.state.doc_lookup, app.state.metadata_cache)

        year = int(metadata.get("date_created", "0")[:4])
        if year < start_year:
            continue
        if year > end_year:
            continue

        if media_type and metadata.get("media_type") != media_type:
            continue

        raw_asset = metadata.get("asset", "").replace("http://", "https://")
        raw_thumbnail = metadata.get("thumbnail", "").replace("http://", "https://")
        asset = quote(raw_asset, safe=":/~")
        thumbnail = quote(raw_thumbnail, safe=":/~")

        date = metadata.get("date_created", "Unknown Date")
        if date != "Unknown Date" and len(date) >= 10:
            try:
                parsed_date = datetime.strptime(date[:10], "%Y-%m-%d")
                date = parsed_date.strftime("%B %d, %Y")
            except ValueError:
                pass

        filtered_results.append({
            "doc_id": doc_id,
            "media_type": metadata.get("media_type", "Unknown"),
            "title": metadata.get("title", "Unknown Title"),
            "description": metadata.get("description", "No description available"),
            "keywords": metadata.get("keywords", []),
            "center": metadata.get("center", "Unknown Center"),
            "date_created": date,
            "location": metadata.get("location", "Unknown Location"),
            "photographer": metadata.get("photographer", "Unknown Photographer"),
            "asset": asset,
            "thumbnail": thumbnail,
            "score": score,
        })

    # Bound the results by limit and offset
    total_results = len(filtered_results)
    paged_results = filtered_results[offset:offset + limit]

    return {"results": paged_results, "total_results": total_results}

@app.get("/apod")
async def get_apod():
    nasa_timezone = pytz.timezone("US/Eastern")
    today_date = datetime.now(nasa_timezone).strftime("%Y-%m-%d")
    apod_file = os.path.join(INDEX_DIR, "apod.json")

    try:
        with open(apod_file, "r") as file:
            cached_apod = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        cached_apod = {}

    if cached_apod.get("date") == today_date:
        return JSONResponse(cached_apod)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}")
            response.raise_for_status()
            apod_data = response.json()
        except Exception as e:
            return JSONResponse(
                {"error": "Failed to fetch APOD", "details": str(e)}, status_code=500
            )
        
    try:
        with open(apod_file, "w") as file:
            json.dump(apod_data, file, indent=2)
    except Exception as e:
        print("Failed to save APOD to JSON: ", e)
    
    return JSONResponse(apod_data)