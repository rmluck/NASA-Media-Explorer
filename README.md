# NASA Media Explorer

*By Rohan Mistry - Last updated August 31, 2025*

---

## 📖 Overview

Search engine application for exploring NASA's public image and video library. Built with Python for data crawling and indexing, spaCy for natural language processing, and FastAPI for serving search results. Frontend features dynamic interface with HTML, CSS, and JavaScript, offering an interactive grid layout, infinite scrolling, and advanced filtering for seaamless media discovery.

**Target Users** are space enthusiasts, researchers, and the general public interesteding in exploring NASA's media archive in a visually engaging manner.

**Dataset**: The crawler retrieves metadata and media links for the full dataset from the [NASA Image and Video Library API](https://images.nasa.gov), indexes them for efficient search, and powers the custom UI. The library consists of hundreds of thousands of images and videos from NASA's archive dating back to 1920. Each document includes a NASA ID, title, text description, keywords, and date along with optional fields such as location, center, and photographer.

![](/static/img/nasa_library_api.png)

---

## 📁 Contents

```bash
├── backend/
│   └── crawler/
│       └── nasa_api_crawler.py     # Crawls NASA API and saves metadata
│   └── indexer/
│       └── indexer.py              # Builds index
│       └── search.py               # Searches queries over index
│   └── main.py                     # FastAPI app entry points
├── data/
│   └── nasa_full_corpus/           # Full corpus of library data, sorted by year
│   └── inverted_index.pkl.gz       # Inverted index
├── frontend/
│   └── templates/                  # Jinja2 HTML templates
│       └── home.html               # Home page
│       └── search_results.html     # Search results page
├── LICENSE                         # MIT License
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

---

## 🌟 Features

* **Search NASA Media**: Query NASA's full image and video archives.
* **Astronomy Picture of the Day**: View the Astronomy Picture of the Day (updated every day by NASA) on the home page.
* **Year Range Filter**: Interactive slider to filter media by creation date (1920-2025).
* **Media Type Filter**: Choose between images or videos.
* **Inverted Index**: Natural language processing implemented using spaCy for stopword removal, field weighting, tokenization, and lemmatization to build index.
* **Context-Aware Search**: Handles queries like "Apollo 11" and "STS-135" intelligently.
* **Document Scoring**: Candidate documents for given query scored and ranked using cosine similarity and Okapi BM25.
* **Dynamic Results Grid**: Thumbnail previews of each result displayed in grid view with hover overlay for titles. Assigned media-type badges.
* **Lazy Loading**: Lazy loading implemented for optimal performance.
* **Infinite Scrolling**: Smoothly loads more results as you scroll.
* **Media Modal**: Pop-up media modal for each result card when clicked upon. Shows image/video in full resolution with additional information and metadata. Full-resolution media can be downloaded as well.
* **Control Panel**: Mission control-inspired top bar for filters and search.

---

## 🛠️ Installation Instructions

To run the app locally.
1. Clone the repository:
```bash
git clone https://github.com/rmluck/NASA-Media-Explorer.git
cd NASA-Media-Explorer
```
2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Set up API key. Retrieve private user-specific NASA API key from [https://api.nasa.gov](https://api.nasa.gov). Create `.env` file in root directory and paste `NASA_API_KEY=[YOUR_KEY]` in file. The key is necessary for the Astronomy Picture of the Day on the home page.
5. (Optional) Full NASA library corpus has already been saved in [data/nasa_full_corpus](/data/nasa_full_corpus/). If you would like to run the crawler script yourself rather than use the provided data, run:
```bash
python -m backend.crawler.nasa_api_crawler
```
6. (Optional) Inverted index file has already been saved as [inverted_index.pkl.gz](/data/inverted_index.pkl.gz). If you would like to run the indexer script yourself rather than use the provided index, run:
```bash
python -m backend.indexer.indexer
python -m backend.indexer.save_inverted_index
```
7. Run backend server
```bash
uvicorn backend.main:app --reload
```
8. Open local frontend in browser at [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 💡 Usage

**Step 1: Home Page**

View NASA's Astronomy Picture of the Day (APOD) on the home page. The API is updated every day at approximately midnight EST.

![](/static/img/home_page.png)

**Step 2: Enter Query**

Search for missions, astronauts, spacecraft, locations, or events. Examples: `nebula` or `Apollo 11`

![](/static/img/search.png)

**Step 3: Browse Results Grid**

Scroll through results with infinite loading. Hover thumbnails to view titles. Hover on video cards to view video preview. Badges indicate media type.

![](/static/video/scroll.mp4)

**Step 5: Filter Results**

Use the year slider to narrow the timeline or choose media type (Image / Video) from the dropdown menu.

![](/static/video/filters.mp4)

**Step 6: Explore Media**

Click a result card to open the full-reoslution image or video along with further details and metadata information directly from NASA's library. Click "Download" button to open media in new tab.

![](/static/video/modal.mp4)

---

## 🚧 Future Improvements

* Deploy as web application (I tried doing this but was having troubles with deploying on Render due to memory limit issues from the large index files.)
* Implement auto-complete search suggestions.
* Enhance filter panel with keyword filters.
* Add "Trending" section to home page.
* Dark mode toggle for more flexible UX.
* Implement as mobile application.

---

## 🧰 Tech Stack

* Python, JavaScript, HTML, CSS
* **Frontend**: Jinja2 templates, Bootstrap
* **Backend**: FastAPI (Python), Uvicorn
* **Crawling**: HTTP requests
* **Natural Language Processing**: `spaCy`, `regex`
* **Indexing**: JSON,`pickle`

---

## 🙏 Contributions / Acknowledgements

This project was built independently as a portfolio project to demonstrate full-stack development, API development, information retrieval and natural language processing, and UI/UX design with real-world space data. 

**Citations**

NASA Image and Video Library API: [https://images.nasa.gov](https://images.nasa.gov)

---

## 🪪 License

This project is licensed under the [MIT License](/LICENSE).