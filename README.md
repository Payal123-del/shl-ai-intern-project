SHL Assessment Recommendation System

Project Overview

This project is a smart recommendation and search system built on top of the SHL product catalog.

It scrapes SHL assessment data, processes and converts it into embeddings, and uses FAISS vector search to find the most relevant assessments based on user queries.

Additionally, a FastAPI backend is used to expose the system as a REST API, making it easy to integrate with frontend or other services.



Tech Stack

- Python
- Pandas
- NumPy
- BeautifulSoup
- Selenium
- FAISS (Facebook AI Similarity Search)
- Sentence Transformers
- FastAPI
- Pydantic
- Groq API (LLM integration)
how to run
1. Clone the repository
```bash
git clone https://github.com/Payal123-del/shl-ai-intern-project
cd shl-ai-intern-project
2. Install req
pip install -r requirements.txt
3.run retriever
python retriever.py
4. Start FastAPI server
uvicorn retriever:app --reload
