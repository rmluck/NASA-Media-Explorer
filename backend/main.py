"""
Main entry point for the NASA Media Explorer backend.
"""


import os
import httpx
import json
import pytz
import sqlite3
from urllib.parse import quote
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from .indexer.search import load_doc_lengths, load_idf_scores, load_doc_lookup, load_avg_doc_length, search_query, get_doc_metadata

DATA_DIR = os.environ.get("PERSISTENT_DISK_PATH", os.path.join(os.path.dirname(__file__), "../data"))
INDEX_FILE = os.path.join(DATA_DIR, "inverted_index.sqlite")
APOD_FILE = os.path.join(DATA_DIR, "apod.json")
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")


# Load index and metadata once on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.idf_scores = load_idf_scores(os.path.join(DATA_DIR, "idf_scores.json"))
    app.state.doc_lengths = load_doc_lengths(os.path.join(DATA_DIR, "doc_lengths.json"))
    app.state.avg_doc_length = load_avg_doc_length(os.path.join(DATA_DIR, "avg_doc_length.json"))
    app.state.doc_lookup = load_doc_lookup(os.path.join(DATA_DIR, "doc_lookup.json"))

    app.state.sqlite_conn = sqlite3.connect(INDEX_FILE, check_same_thread=False)

    try:
        yield
    finally:
        if hasattr(app.state, "sqlite_conn"):
            app.state.sqlite_conn.close()

# Create FastAPI app instance
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "../frontend/templates"))


@app.get("/")
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("home.html", {"request": request, "year": datetime.now().year})


@app.get("/search", response_class=HTMLResponse)
def search_results(request: Request, query: str = Query(...), limit: int = 20, offset: int = 0) -> HTMLResponse:
    # Run the search query using the loaded index and metadata
    results = search_query(query, app.state.sqlite_conn, app.state.idf_scores, app.state.doc_lengths, app.state.avg_doc_length)

    # Bound the results by limit and offset
    total_results = len(results)
    paged_results = results[offset:offset + limit]

    # Fetch document metadata for the results
    top_results = []
    for doc_id, score in paged_results:
        metadata = get_doc_metadata(doc_id, app.state.doc_lookup)

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
    results = search_query(query, app.state.sqlite_conn, app.state.idf_scores, app.state.doc_lengths, app.state.avg_doc_length)

    # Fetch document metadata for the results
    filtered_results = []
    for doc_id, score in results:
        metadata = get_doc_metadata(doc_id, app.state.doc_lookup)

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

    try:
        with open(APOD_FILE, "r") as file:
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
        tmp_file = APOD_FILE + ".tmp"
        with open(tmp_file, "w") as file:
            json.dump(apod_data, file, indent=2)
        os.replace(tmp_file, APOD_FILE)
    except Exception as e:
        print("Failed to save APOD to JSON: ", e)
    
    return JSONResponse(apod_data)