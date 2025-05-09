import os
import json
import time
import traceback

from flask import Flask, request, jsonify

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Configuration ---
GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', 'gemini-2.5-flash-preview-04-17')
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
EMAIL_DATA_PATH = 'data/emails.json'
EMBEDDING_SAVE_PATH = 'data/email_embeddings.npy'
TOP_N_RETRIEVAL = int(os.environ.get('TOP_N_RETRIEVAL', 3))

# --- Global Variables / Models (Load once at startup) ---
EMAILS_DF = None
EMAIL_EMBEDDINGS = None
SBERT_MODEL = None
GEMINI_CLIENT = None

def load_resources():
    """Loads models and data needed by the RAG pipeline."""
    global EMAILS_DF, EMAIL_EMBEDDINGS, SBERT_MODEL, GEMINI_CLIENT

    print("--- Loading Resources for API ---")
    start_time = time.time()

    # 1. Load Email Data
    try:
        print(f"Loading email data from {EMAIL_DATA_PATH}...")
        EMAILS_DF = pd.read_json(EMAIL_DATA_PATH)
        EMAILS_DF['full_text'] = "Sender: " + EMAILS_DF['sender'] + "\nSubject: " + EMAILS_DF['subject'] + "\n\n" + EMAILS_DF['body']
        print(f"Loaded {len(EMAILS_DF)} emails.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load email data from {EMAIL_DATA_PATH}. Error: {e}")
        raise RuntimeError(f"Failed to load email data: {e}")

    # 2. Load Sentence Transformer Model
    try:
        print(f"Loading Sentence Transformer model '{EMBEDDING_MODEL_NAME}'...")
        SBERT_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Sentence Transformer model loaded.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load Sentence Transformer model. Error: {e}")
        raise RuntimeError(f"Failed to load SBERT model: {e}")

    # 3. Load or Generate Email Embeddings
    try:
        if os.path.exists(EMBEDDING_SAVE_PATH):
            print(f"Loading pre-computed embeddings from {EMBEDDING_SAVE_PATH}...")
            EMAIL_EMBEDDINGS = np.load(EMBEDDING_SAVE_PATH)
            print(f"Loaded embeddings. Shape: {EMAIL_EMBEDDINGS.shape}")
            if EMAIL_EMBEDDINGS.shape[0] != len(EMAILS_DF):
                print("WARNING: Embeddings file length mismatch with email data. Regenerating...")
                raise FileNotFoundError # Trigger regeneration
        else:
            print(f"Embeddings file not found at {EMBEDDING_SAVE_PATH}. Generating...")
            raise FileNotFoundError # Trigger regeneration

    except (FileNotFoundError, ValueError, Exception) as e: 
        if not isinstance(e, FileNotFoundError):
             print(f"Warning: Issue loading embeddings ({e}). Will regenerate.")
        if SBERT_MODEL is not None and EMAILS_DF is not None and not EMAILS_DF.empty:
            try:
                print(f"Generating embeddings for {len(EMAILS_DF)} emails...")
                email_contents_to_embed = EMAILS_DF['full_text'].tolist()
                EMAIL_EMBEDDINGS = SBERT_MODEL.encode(email_contents_to_embed) 
                print(f"Embeddings generated. Shape: {EMAIL_EMBEDDINGS.shape}")
                os.makedirs(os.path.dirname(EMBEDDING_SAVE_PATH), exist_ok=True)
                np.save(EMBEDDING_SAVE_PATH, EMAIL_EMBEDDINGS)
                print(f"Embeddings saved to {EMBEDDING_SAVE_PATH}")
            except Exception as gen_e:
                print(f"FATAL ERROR: Failed to generate/save embeddings: {gen_e}")
                raise RuntimeError(f"Failed to generate/save embeddings: {gen_e}")
        else:
             print("FATAL ERROR: Cannot generate embeddings - model or data not loaded.")
             raise RuntimeError("Cannot generate embeddings - model or data not loaded.")

    # 4. Initialize Gemini Client
    try:
        print("Initializing Gemini client...")
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("FATAL ERROR: GOOGLE_API_KEY environment variable not set.")
        GEMINI_CLIENT = genai.Client(api_key=api_key)
        print("Gemini client initialized successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize Gemini client: {e}")
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")

    end_time = time.time()
    print(f"--- Resource loading complete. Took {end_time - start_time:.2f} seconds ---")


# --- RAG Core Logic Functions

def find_similar_emails_api(query, top_n=TOP_N_RETRIEVAL):
    """API version: Finds top_n similar emails using global resources."""
    if SBERT_MODEL is None or EMAIL_EMBEDDINGS is None or EMAILS_DF is None:
         print("Error in find_similar_emails_api: Resources not loaded")
         raise ValueError("Core resources (model, embeddings, data) not loaded.")
    if EMAIL_EMBEDDINGS.size == 0:
        return [], [] 

    try:
        query_embedding = SBERT_MODEL.encode([query])
        similarities = cosine_similarity(query_embedding, EMAIL_EMBEDDINGS)[0]

        num_emails = EMAIL_EMBEDDINGS.shape[0]
        actual_top_n = min(top_n, num_emails)
        if actual_top_n <= 0: return [], []

        top_indices_sorted = np.argsort(similarities)
        top_indices = top_indices_sorted[-actual_top_n:][::-1]

        similar_emails_content_api = []
        for index in top_indices:
            if 0 <= index < len(EMAILS_DF):
                email_row = EMAILS_DF.iloc[index]
                email_info = f"Email ID: {email_row.get('id', 'N/A')}\nSender: {email_row.get('sender', 'N/A')}\nDate: {email_row.get('date', 'N/A')}\nSubject: {email_row.get('subject', 'N/A')}\nBody:\n{email_row.get('body', 'N/A')}"
                similar_emails_content_api.append(email_info)
        return similar_emails_content_api
    except Exception as e:
        print(f"Error during similarity search in API: {e}")
        raise RuntimeError("Failed during similarity search.")


def generate_email_response_gemini_api(user_query, retrieved_emails_content):
    """API version: Generates response using Gemini client."""
    if GEMINI_CLIENT is None:
         print("Error in generate_email_response_gemini_api: Gemini client not loaded")
         raise ValueError("Gemini client not initialized.")

    if not retrieved_emails_content:
        return "I couldn't find any relevant past emails to answer your query. Please try rephrasing."

    email_context = "\n\n---\n\n".join(retrieved_emails_content)
    prompt = f"""You are an Email Wizard Assistant. Your task is to answer the user's query based *only* on the provided email context below. Be concise and factual based on the emails. If the answer is not found in the emails, state that clearly. Do not make up information.

Retrieved Email(s) Context:
---
{email_context}
---

User Query: "{user_query}"

Assistant's Answer:
"""
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt
        )

        prompt_feedback = getattr(response, 'prompt_feedback', None)
        if prompt_feedback is not None:
            block_reason = getattr(prompt_feedback, 'block_reason', None)
            if block_reason is not None:
                reason_msg = getattr(prompt_feedback, 'block_reason_message', str(block_reason))
                print(f"Gemini response blocked. Reason: {reason_msg}")
                return f"Sorry, the request could not be completed due to content safety filters ({reason_msg})."

        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'parts') and response.parts:
            return "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
        else:
            print(f"Warning: Gemini returned empty/non-text response: {response}")
            return "Sorry, the AI model returned an empty or unexpected response."

    except Exception as e:
        print(f"ERROR during Gemini API call in API: {type(e).__name__} - {e}")
        raise RuntimeError(f"AI service communication error ({type(e).__name__}).")


def rag_email_assistant_api(user_query):
    """API orchestrator: retrieve, then generate."""
    print(f"Processing API query: '{user_query}'")
    retrieved_snippets = find_similar_emails_api(user_query)
    response = generate_email_response_gemini_api(user_query, retrieved_snippets)
    print(f"Generated response length: {len(response)}")
    return response


# --- API Endpoint ---
@app.route('/query_email', methods=['POST'])
def query_email_endpoint():
    """Handles POST requests to /query_email"""
    request_start_time = time.time()
    print(f"\nReceived request at {request_start_time:.2f}")

    # 1. Validate Input
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            print("Error: Missing 'query' in request JSON")
            return jsonify({"error": "Missing 'query' field in JSON request body"}), 400

        user_query = data['query']
        if not isinstance(user_query, str) or not user_query.strip():
            print(f"Error: Invalid 'query' field: {user_query}")
            return jsonify({"error": "'query' must be a non-empty string"}), 400
        print(f"Valid query received: '{user_query[:100]}...'") 

    except Exception as e: 
        print(f"Error parsing request JSON: {e}")
        return jsonify({"error": "Invalid JSON format in request body"}), 400

    # 2. Process Query using RAG pipeline
    try:
        assistant_response = rag_email_assistant_api(user_query)
        response_status = 200
        response_body = {"response": assistant_response}

        if isinstance(assistant_response, str) and assistant_response.startswith("ERROR:"):
            response_status = 500 # Internal Server Error
            response_body = {"error": assistant_response}


    except ValueError as ve:
         print(f"ValueError during RAG processing: {ve}")
         response_status = 400 
         response_body = {"error": str(ve)}
    except RuntimeError as rte: 
         print(f"RuntimeError during RAG processing: {rte}")
         response_status = 500
         response_body = {"error": f"An internal error occurred: {rte}"}
    except Exception as e:
        print(f"FATAL UNEXPECTED ERROR in /query_email: {type(e).__name__} - {e}")
        traceback.print_exc() 
        response_status = 500
        response_body = {"error": "An unexpected internal server error occurred. Please try again later."}

    request_end_time = time.time()
    print(f"Request processed in {request_end_time - request_start_time:.2f} seconds. Status: {response_status}")
    return jsonify(response_body), response_status


# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        load_resources()
        port = int(os.environ.get("PORT", 5000))
        print(f"Starting Flask server on host 0.0.0.0, port {port}...")
        app.run(host='0.0.0.0', port=port, debug=False)
    except RuntimeError as e:
        print(f"ERROR: Failed to start the Flask application due to resource loading failure: {e}")
    except Exception as e:
         print(f"ERROR: An unexpected error occurred on startup: {e}")