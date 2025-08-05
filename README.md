# Streamlit Introduction

This project is a simple introduction to Streamlit, a Python library for creating web applications.

## Project Structure

```
streamlit_intro/
├── src/
│   └── main.py         # Main Streamlit application
├── requirements.txt    # Python dependencies
├── pyproject.toml     # Project configuration with uv
├── uv.lock            # Lock file for reproducible builds
├── Dockerfile         # Docker container configuration
├── .dockerignore      # Docker ignore patterns
└── README.md          # This file
```

## Installation

### Using uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management:

1. Clone this repository
2. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```

### Using pip

Alternatively, you can use traditional pip:

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Local Development

To run the Streamlit application locally:

```bash
# With uv
uv run streamlit run src/main.py

# With pip
streamlit run src/main.py
```

The application will be available at `http://localhost:8501`.

### Using Docker

You can also run the application using Docker:

1. Build the Docker image:
   ```bash
   docker build -t streamlit-intro .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 streamlit-intro
   ```

The application will be available at `http://localhost:8501`.

