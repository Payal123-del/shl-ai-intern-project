import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_llm_response(query, results):

    if not results:
        return "No assessments found in database."

    context = "\n".join([
        f"Name: {r.get('name','')}\nType: {r.get('test_type','')}\nDescription: {r.get('description','')}\n---"
        for r in results
    ])

    prompt = f"""
You are an AI assistant helping to match SHL assessments with user queries.

User Query:
{query}

Available Assessments:
{context}

Instructions:
1. Give a short explanation (2-3 lines)
2. Pick the best matching assessment name
3. Explain why it matches
4. If no good match exists, say so clearly
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content