from qdrant_client import QdrantClient
import openai
from server.core.config import config


def create_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding


def retrieve_embedding_data(qd_client: QdrantClient, query, collection_name, k=5):
    response = qd_client.query_points(
        collection_name=collection_name,
        query=create_embeddings(query),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    retrieved_scores = []
    retrieved_context_ratings = []
    
    for point in response.points:
        retrieved_context_ids.append(point.payload["parent_asin"])
        retrieved_context.append(point.payload["description"])
        retrieved_scores.append(point.score)
        retrieved_context_ratings.append(point.payload["average_rating"])

    # return dictionary of retrieved data
    return {
        "context_ids": retrieved_context_ids,
        "context": retrieved_context,
        "scores": retrieved_scores,
        "context_ratings": retrieved_context_ratings
    }

def format_context(retrived_context):
    formatted_context = ""
    for id, chunk, rating in zip(retrived_context["context_ids"], retrived_context["context"], retrived_context["context_ratings"]):
        formatted_context += f"Product ID: {id}, rating: {rating}, description: {chunk.strip()}\n"
    return formatted_context

def build_prompt(preprocessed_context, question):
    prompt = f"""
You are a specialized Product Expert Assistant. Your goal is to answer customer questions accurately using ONLY the provided product information.

### Instructions:
1. **Source of Truth:** Answer strictly based on the provided "Available Products" section below. Do not use outside knowledge or make assumptions.
2. **Handling Missing Info:** If the answer cannot be found in the provided products, politely state that you do not have that information. Do not make up features.
3. **Tone:** Be helpful, professional, and concise.
4. **Terminology:** Never refer to the text below as "context" or "data." Refer to it naturally as "our current inventory" or "available products."

### Available Products:
<inventory_data>
{preprocessed_context}
</inventory_data>

### Customer Question:
{question}

### Answer:
"""
    return prompt

def generate_llm_response(prompt, model="gpt-5-nano"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def integrated_rag_pipeline(question, model="gpt-5-nano"):
    
    qdrant_client = QdrantClient(   
        url=config.qdrant_url,
    )
    # Step 1: Retrieve relevant context
    retrieved_context = retrieve_embedding_data(
        qdrant_client,
        question,
        collection_name="amazon_items-collection-00",
        k=5
    )
    # Step 2: Format context
    formatted_context = format_context(retrieved_context)   
    # Step 3: Build prompt
    prompt = build_prompt(formatted_context, question)
    # Step 4: Generate response
    response = generate_llm_response(prompt, model)
    return response