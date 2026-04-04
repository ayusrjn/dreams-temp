# DREAMS
Digitization for Recovery: Exploring Arts with Mining for Societal well-being.

DREAMS is an extension of the Beehive project, focused on exploring time and ordering across photo memories to better understand personal recovery journeys. The goal is to build tools that help track and analyze visual narratives over time using data mining and intelligent processing.

## Current Progress

- Set up core infrastructure using Flask and Hugging Face models.
- Implemented a basic **Caption Sentiment Analysis API** to classify emotional tone in user-submitted captions.
- Integrating this API into Beehive to capture sentiment when users upload photos.
- Exploring time-based data structuring and narrative analysis features.

### [View the API Module](./dreamsApp/README.md)

## Repositories

- Beehive: [github.com/KathiraveluLab/beehive](https://github.com/KathiraveluLab/Beehive)
- DREAMS: [github.com/KathiraveluLab/DREAMS](https://github.com/KathiraveluLab/DREAMS)


## Repository Structure

```text
DREAMS/
├── dreamsApp/                  # Main application package
│   ├── app/                    # Flask app package (app factory + blueprints)
│   │   ├── __init__.py         # create_app() factory
│   │   ├── config.py           # App configuration
│   │   ├── models.py           # Database models
│   │   ├── auth.py             # Authentication routes
│   │   │
│   │   ├── ingestion/          # Image ingestion & processing
│   │   │   ├── __init__.py
│   │   │   └── routes.py
│   │   │
│   │   ├── dashboard/          # Dashboard & analytics views
│   │   │   ├── __init__.py
│   │   │   └── main.py
│   │   │
│   ├── core/                   # Decoupled Core ML / NLP Engine
│   │   ├── pipeline.py         # Standalone ingestion orchestrator
│   │   ├── graph/              # Temporal narrative modeling & analytics
│   │   ├── database.py         # SQLite / Vector storage logic
│   │   └── sentiment.py        # RoBERTa classification logic
│   │
│   └── docs/                   # Project documentation
│
├── data_integrity/             # Data validation utilities
├── location_proximity/         # Location-based analysis (future)
├── dream-integration/          # Integration & experimental code
├── tests/                      # Unit and integration tests
│
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
└── README.md                   # Project documentation
```
 
## Installation and Setup

### Clone the repository
```bash
git clone https://github.com/KathiraveluLab/DREAMS.git
cd DREAMS
```

### (Optional but recommended) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install the required dependencies
```bash
pip install -r requirements.txt
```
#### For CPU-only installations
```bash
pip install -r requirements-cpu.txt
```

### Run tests to verify everything is working
```bash
pytest
```

### Start the Flask server in debug mode
```bash
flask --app "dreamsApp.app:create_app()" run --debug
```

### Run the Core Pipeline Standalone
The ML algorithm layer (`dreamsApp/core/`) is designed to operate completely independently from the Flask API layer. The Flask UI runs on MongoDB, whereas the native standalone pipeline operates safely on embedded SQLite / ChromaDB databases for fast local research testing.

For an in-depth, step-by-step tutorial on how to programmatically execute the DREAMS pipeline and generate visual trajectory plots, please see **`Example.ipynb`**.
