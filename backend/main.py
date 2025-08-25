"""
Main entry point for the NASA Media Explorer backend.
"""


import os
from urllib.parse import quote
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from indexer.search import load_inverted_index, load_doc_lengths, load_idf_scores, load_doc_lookup, load_avg_doc_length, search_query, get_doc_metadata

INDEX_DIR = "../data"


# Load index and metadata once on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inverted_index = load_inverted_index(os.path.join(INDEX_DIR, "inverted_index.json"))
    app.state.idf_scores = load_idf_scores(os.path.join(INDEX_DIR, "idf_scores.json"))
    app.state.doc_lengths = load_doc_lengths(os.path.join(INDEX_DIR, "doc_lengths.json"))
    app.state.avg_doc_length = load_avg_doc_length(os.path.join(INDEX_DIR, "avg_doc_length.json"))
    app.state.doc_lookup = load_doc_lookup(os.path.join(INDEX_DIR, "doc_lookup.json"))
    app.state.metadata_cache = {}

    yield

# Create FastAPI app instance
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


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
def search_api(query: str = Query(...), limit: int = 20, offset: int = 0, start_year: int = None, end_year: int = None, media_type: str = None):
    # Run the search query using the loaded index and metadata
    results = search_query(query, app.state.inverted_index, app.state.idf_scores, app.state.doc_lengths, app.state.avg_doc_length)

    # Fetch document metadata for the results
    filtered_results = []
    for doc_id, score in results:
        metadata = get_doc_metadata(doc_id, app.state.doc_lookup, app.state.metadata_cache)

        year = int(metadata.get("date_created", "0")[:4])
        if start_year and year < start_year:
            continue
        if end_year and year > end_year:
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