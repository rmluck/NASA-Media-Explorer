"""
Main entry point for the NASA Media Explorer backend.
"""


import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from indexer.search import load_inverted_index, load_doc_lengths, load_idf_scores, load_doc_lookup, load_avg_doc_length, search_query, get_doc_metadata


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


# Define the directory where the index files are stored
INDEX_DIR = "data"

# Create FastAPI app instance
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/search")
def search_results(query: str = Query(...), limit: int = 20, offset: int = 0):
    # Run the search query using the loaded index and metadata
    results = search_query(query, app.state.inverted_index, app.state.idf_scores, app.state.doc_lengths, app.state.avg_doc_length)

    # Fetch document metadata for the results
    top_results = []
    for doc_id, score in results[offset:offset + limit]:
        metadata = get_doc_metadata(doc_id, app.state.doc_lookup, app.state.metadata_cache)
        links = metadata.get("links", [])
        top_results.append({
            "doc_id": doc_id,
            "media_type": metadata.get("media_type", "Unknown"),
            "title": metadata.get("title", "Unknown Title"),
            "description": metadata.get("description", "No description available"),
            "keywords": metadata.get("keywords", []),
            "center": metadata.get("center", "Unknown Center"),
            "date_created": metadata.get("date_created", "Unknown Date"),
            "location": metadata.get("location", "Unknown Location"),
            "photographer": metadata.get("photographer", "Unknown Photographer"),
            "links": links,
            "score": score,
        })

    # Return the search results as JSON
    return JSONResponse({"query": query, "results": top_results})