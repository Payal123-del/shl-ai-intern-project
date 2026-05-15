import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel

from llm import generate_llm_response


df = pd.read_csv("catalog.csv")

required_cols = ["name", "description", "test_type", "url"]
for col in required_cols:
    if col not in df.columns:
        df[col] = ""


df["combined_text"] = (
    "Assessment Name: " + df["name"].fillna("") + ". " +
    "Description: " + df["description"].fillna("") + ". " +
    "Test Type: " + df["test_type"].fillna("")
)

def enrich_text(text):
    text_lower = text.lower()
    extra = []

    if "java" in text_lower:
        extra.append("java backend developer spring boot coding software engineer")

    if "personality" in text_lower:
        extra.append("communication teamwork behavior soft skills leadership")

    if "numerical" in text_lower:
        extra.append("analytics quantitative reasoning math aptitude statistics")

    return text + " " + " ".join(extra)

df["combined_text"] = df["combined_text"].apply(enrich_text)


model = SentenceTransformer("all-MiniLM-L6-v2")


embeddings = model.encode(
    df["combined_text"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
).astype("float32")

faiss.normalize_L2(embeddings)


dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

def preprocess_query(query):
    q = query.lower()

    boost = []

    if "data analyst" in q:
        boost += ["excel", "sql", "python", "data analysis", "dashboard"]

    if "numerical" in q or "reasoning" in q:
        boost += ["quantitative reasoning", "math", "statistics", "logic", "aptitude"]

    if "developer" in q:
        boost += ["coding", "software engineering", "backend", "java", "python"]

    return query + " " + " ".join(boost)
def rerank(results, query):
    q = query.lower()

    for r in results:
        boost = 0

        name = r["name"].lower()

        if "data analyst" in q:
            keywords = ["account", "accounting", "report", "analysis", "data", "excel", "sql"]
            if any(k in name for k in keywords):
                boost += 0.03

        if "numerical" in q:
            keywords = ["simulation", "accounting", "math", "statistics", "data"]
            if any(k in name for k in keywords):
                boost += 0.05

        r["score"] += boost

    return sorted(results, key=lambda x: x["score"], reverse=True)
def detect_domain(query):
    q = query.lower()

    if "java" in q:
        return "java"
    if ".net" in q or "c#" in q:
        return ".net"
    if "python" in q:
        return "python"

    return "general"
def domain_filter(results, query):
    domain = detect_domain(query)

    filtered = []

    for r in results:
        name = r["name"].lower()
        desc = r.get("description", "").lower()

        if domain == "java":
            if "java" in name or "coding" in name or "software" in name:
                filtered.append(r)

        elif domain == ".net":
            if ".net" in name or "mvc" in name or "wcf" in name:
                filtered.append(r)

        else:
            filtered.append(r)

    return filtered
def search_assessments(query, top_k=5):

    query = preprocess_query(query)

    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k * 2)

    results = []

    for idx, score in zip(indices[0], distances[0]):
        if idx == -1:
            continue

        row = df.iloc[idx]

        results.append({
            "name": row["name"],
            "url": row["url"],
            "test_type": row["test_type"],
            "score": float(score)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    results = rerank(results, query)

    results = domain_filter(results, query)

    filtered = [r for r in results if r["score"] > 0.25]

    if len(filtered) == 0:
        filtered = results[:3]

    return filtered[:top_k]
app = FastAPI()

class ChatRequest(BaseModel):
    messages: list


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(data: ChatRequest):

    user_message = data.messages[-1]["content"].strip().lower()


    blocked_words = ["salary", "legal", "ignore instructions"]

    if any(word in user_message for word in blocked_words):
        return {
            "reply": "I can only assist with SHL assessment recommendations.",
            "recommendations": [],
            "end_of_conversation": True
        }

   

    if len(user_message.split()) < 4:
        return {
            "reply": "Please provide role, skills, or experience level for better recommendations.",
            "recommendations": [],
            "end_of_conversation": False
        }

    if "difference" in user_message:
        return {
            "reply": "Comparison feature will be added soon.",
            "recommendations": [],
            "end_of_conversation": False
        }


    results = search_assessments(user_message)

    llm_output = generate_llm_response(user_message, results)

    return {
        "reply": llm_output,
        "recommendations": results,
        "end_of_conversation": False
}
if __name__ == "__main__":

    query = input("Enter your query: ")

    results = search_assessments(query)   # tera retriever function

    print("\n===== INPUT =====")
    print(query)

    print("\n===== OUTPUT RESULTS =====")
    for r in results:
        print(f"{r['name']} | Score: {r['score']:.2f}")
