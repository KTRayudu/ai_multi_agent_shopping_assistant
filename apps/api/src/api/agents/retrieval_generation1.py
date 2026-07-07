from qdrant_client import QdrantClient
import openai
from server.core.config import config
import google.generativeai as genai
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os



# def create_embeddings(text, model="text-embedding-3-small"):
#     response = openai.embeddings.create(
#         model=model,
#         input=text
#     )
#     return response.data[0].embedding


# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# Initialize client
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

# 3. Define the embedding function with custom dimensionality
def get_embedding(text, model="gemini-embedding-001",):
    """
    Generates an embedding vector for the given text using Gemini's current model.
    """
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001", 
        contents=text,
        config=types.EmbedContentConfig(
        #     task_type=task_type
        #     # 🔴 THE FIX: Force the output to be exactly 1536 dimensions
        #     # output_dimensionality=1536  
        )
    )
    
    return response.embeddings[0].values


# def retrieve_embedding_data(qd_client: QdrantClient, query, collection_name, k=5):
#     response = qd_client.query_points(
#         collection_name=collection_name,
#         query=create_embeddings(query),
#         limit=k
#     )

#     retrieved_context_ids = []
#     retrieved_context = []
#     retrieved_scores = []
#     retrieved_context_ratings = []
    
#     for point in response.points:
#         retrieved_context_ids.append(point.payload["parent_asin"])
#         retrieved_context.append(point.payload["description"])
#         retrieved_scores.append(point.score)
#         retrieved_context_ratings.append(point.payload["average_rating"])

#     # return dictionary of retrieved data
#     return {
#         "context_ids": retrieved_context_ids,
#         "context": retrieved_context,
#         "scores": retrieved_scores,
#         "context_ratings": retrieved_context_ratings
#     }




def retrieve_data(query, qdrant_client, k=5):

    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-03",
        query=query_embedding,
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }




def process_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context



def build_prompt(preprocessed_context, question):

    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructtions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as the available products.

Context:
{preprocessed_context}

Question:
{question}
"""

    return prompt



# def generate_answer(prompt):

#     response = openai.chat.completions.create(
#         model="gpt-5-nano",
#         messages=[{"role": "system", "content": prompt}],
#         reasoning_effort="minimal"
#     )

#     return response.choices[0].message.content


# AVAILABLE MODELS FOR gemini_client.models.generate_content:
# -----------------------------------------------------------
# "gemini-2.5-flash"        : Best for speed, low latency, and general tasks.
# "gemini-2.0-flash"        : High-speed, versatile, supports multi-modal input.
# "gemini-2.0-pro-exp-02-05": High performance, complex reasoning, and deep analysis.
# "gemini-2.0-flash-thinking-exp": Specialized for reasoning, math, and code logic.
# "gemini-1.5-flash"        : Legacy fast model, still widely used.
# "gemini-1.5-pro"          : Deep context understanding and complex multi-step reasoning.
# -----------------------------------------------------------

def generate_answer(prompt, model_name="gemini-2.5-flash"):
    """
    Generates a response using the specified Gemini model.
    """
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt,
        # You can add config here if you need system instructions or temperature control
    )
    
    return response.text







def rag_pipeline(question, top_k=5):

    qdrant_client = QdrantClient(url="http://localhost:6333")

    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)

    return answer




























# def format_context(retrived_context):
#     formatted_context = ""
#     for id, chunk, rating in zip(retrived_context["context_ids"], retrived_context["context"], retrived_context["context_ratings"]):
#         formatted_context += f"Product ID: {id}, rating: {rating}, description: {chunk.strip()}\n"
#     return formatted_context

# def build_prompt(preprocessed_context, question):
#     prompt = f"""
# You are a specialized Product Expert Assistant. Your goal is to answer customer questions accurately using ONLY the provided product information.

# ### Instructions:
# 1. **Source of Truth:** Answer strictly based on the provided "Available Products" section below. Do not use outside knowledge or make assumptions.
# 2. **Handling Missing Info:** If the answer cannot be found in the provided products, politely state that you do not have that information. Do not make up features.
# 3. **Tone:** Be helpful, professional, and concise.
# 4. **Terminology:** Never refer to the text below as "context" or "data." Refer to it naturally as "our current inventory" or "available products."

# ### Available Products:
# <inventory_data>
# {preprocessed_context}
# </inventory_data>

# ### Customer Question:
# {question}

# ### Answer:
# """
#     return prompt

# def generate_llm_response(prompt, model="gpt-5-nano"):
#     response = openai.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": prompt},
#         ]
#     )
#     return response.choices[0].message.content

# def integrated_rag_pipeline(question, model="gpt-5-nano"):
    
#     qdrant_client = QdrantClient(   
#         url=config.qdrant_url,
#     )
#     # Step 1: Retrieve relevant context
#     retrieved_context = retrieve_embedding_data(
#         qdrant_client,
#         question,
#         collection_name="amazon_items-collection-00",
#         k=5
#     )
#     # Step 2: Format context
#     formatted_context = format_context(retrieved_context)   
#     # Step 3: Build prompt
#     prompt = build_prompt(formatted_context, question)
#     # Step 4: Generate response
#     response = generate_llm_response(prompt, model)
#     return response