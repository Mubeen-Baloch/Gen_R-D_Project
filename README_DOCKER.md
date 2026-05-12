# 🐳 Docker Deployment Guide

To satisfy the project rubric requirements, this system is now fully containerized.

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

## How to Run

1.  **Open a terminal** in the project root directory.
2.  **Build and Start** the services:
    ```bash
    docker-compose up --build
    ```
3.  **Access the Application**:
    - Streamlit Dashboard: [http://localhost:8501](http://localhost:8501)
    - GROBID Service: [http://localhost:8070](http://localhost:8070)

## Services
- **research-agent**: The main Python application running the multi-agent framework.
- **grobid**: High-fidelity PDF parsing service required for extracting text from research papers.

## Note on API Keys
Ensure your `.env` file contains your `GOOGLE_API_KEY` (or others). Docker Compose is configured to mount your `.env` file into the container automatically.
