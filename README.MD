# Email Wizard Assistant

## 1. Summary of the Solution

The Email Wizard Assistant is a Retrieval-Augmented Generation (RAG) system designed to help users quickly find answers to their email queries by retrieving relevant past emails and generating intelligent replies based on that context.

This project implements the following:
*   **Dataset:** A diverse set of 50 sample emails covering personal, professional, and fictional scenarios.
*   **Embedding:** Emails are processed and converted into dense vector embeddings using the `all-MiniLM-L6-v2` sentence transformer model. These embeddings capture the semantic meaning of the email content.
*   **Similarity Search:** When a user submits a query, the query is also embedded, and cosine similarity is used to find the most semantically similar past emails from the dataset.
*   **Generative AI:** The user's query along with the content of the retrieved similar emails are then passed as context to Google's Gemini Pro model (via the `google-genai` SDK).
*   **Response Generation:** The Gemini model generates a coherent answer based on the query and the provided email context.
*   **API:** A Flask API provides a `POST /query_email` endpoint to interact with the assistant.
*   **Development Notebook:** A Jupyter notebook (`notebook.ipynb`) details the step-by-step implementation and experimentation of the RAG pipeline.

## 2. Project Structure

```
email-wizard-assistant/
├── .gitignore
├── README.md
├── requirements.txt
├── notebook.ipynb           # Main development and experimentation notebook
├── data/
│   └── emails.json          # Sample email dataset
│   └── email_embeddings.npy # (Optional) Saved embeddings for faster startup
└── api/
    └── app.py               # Flask API application
```

## 3. Setup Instructions

Follow these steps to set up and run the project locally.

### 3.1. Prerequisites
*   Python 3.8+
*   Git

### 3.2. Clone the Repository
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL_HERE>
cd email-wizard-assistant
```

### 3.3. Set up a Virtual Environment
It's highly recommended to use a virtual environment.
```bash
# Create a virtual environment (e.g., named 'venv')
python -m venv venv

# Activate the virtual environment
# On Windows:
# venv\Scripts\activate.bat  (for Command Prompt)
# .\venv\Scripts\Activate.ps1 (for PowerShell - you might need to Set-ExecutionPolicy Unrestricted -Scope Process)
# On macOS/Linux:
# source venv/bin/activate
```

### 3.4. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3.5. Obtain and Set Google Gemini API Key
This project uses the Google Gemini API for response generation.
1.  Obtain an API key from [Google AI Studio](https://aistudio.google.com/) by clicking "Get API key".
2.  **Set the API key as an environment variable:**
    *   **For running the Jupyter Notebook locally (if making API calls from it):**
        ```bash
        # On Linux/macOS:
        export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        # On Windows Command Prompt:
        set GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        # On Windows PowerShell:
        $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```
    *   **For running the Flask API (this is essential):**
        You need to set this environment variable in the terminal session where you will run `python api/app.py`.

## 4. How to Run the Jupyter Notebook (`notebook.ipynb`)

The Jupyter notebook provides a detailed walkthrough of the RAG pipeline development, including data loading, embedding, similarity search, and generation.

1.  Ensure your virtual environment is activated and dependencies are installed.
2.  If you plan to execute cells that call the Gemini API from the notebook, ensure the `GOOGLE_API_KEY` environment variable is set (see section 3.5).
3.  Start JupyterLab (or Jupyter Notebook):
    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```
4.  Open `notebook.ipynb` from the Jupyter interface in your browser.
5.  You can run cells individually or "Run All Cells" from the menu. The notebook is structured to load data, generate/load embeddings, and test the RAG pipeline.

## 5. How to Run the Flask API (`api/app.py`)

The Flask API exposes an endpoint to interact with the Email Wizard Assistant.

1.  Ensure your virtual environment is activated and dependencies are installed.
2.  **Crucially, set the `GOOGLE_API_KEY` environment variable** in your terminal session (see section 3.5). The API will not start or function correctly without it.
3.  Navigate to the project's root directory in your terminal.
4.  Run the Flask application:
    ```bash
    python api/app.py
    ```
5.  The server will start, typically on `http://127.0.0.1:5000/` (or `http://0.0.0.0:5000/`). You'll see log messages in the terminal, including resource loading information.

## 6. How to Test the API and Interact with the Email Assistant

Once the Flask API server is running, you can send `POST` requests to the `/query_email` endpoint.

The request body must be JSON and contain a `query` field.
Example: `{"query": "What is the status of Project Phoenix?"}`

### 6.1. Using `curl` (from a new terminal window)

```bash
# Example for Linux/macOS/Git Bash on Windows (Ensure proper quote escaping for your shell)
# For one line:
# curl -X POST -H "Content-Type: application/json" -d "{\"query\": \"Any news about Project Phoenix?\"}" http://127.0.0.1:5000/query_email
# For multi-line (easier to read):
curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "Any news about Project Phoenix?"}' \
     http://127.0.0.1:5000/query_email

# Example for Windows PowerShell (using Invoke-WebRequest)
Invoke-WebRequest -Uri http://127.0.0.1:5000/query_email -Method POST -ContentType "application/json" -Body '{"query": "Any news about Project Phoenix?"}'
```

**Expected Success Response (JSON):**
```json
{
  "response": "The Project Phoenix Phase 2 is currently in progress. The initial data migration has been completed, and the development team is now focusing on implementing the new user interface modules. Testing for Module A is scheduled to begin next Monday..."
}
```

**Expected Error Response (e.g., Bad Request):**
```json
{
  "error": "Missing 'query' field in JSON request body"
}
```

### 6.2. Using a Python `requests` script

Create a Python script (e.g., `test_api.py`) in the project root:
```python
# test_api.py
import requests
import json

api_url = "http://127.0.0.1:5000/query_email"
test_queries = [
    "What is the status of Project Phoenix?",
    "Are dogs allowed at the Summer Street Fair?",
    "Tell me about the Italy trip planning",
    "What did Perplexed Pete report about his emails?"
]

for query in test_queries:
    print(f"\n--- Testing Query: '{query}' ---")
    payload = {"query": query}
    try:
        response = requests.post(api_url, json=payload, timeout=45) # Increased timeout for LLM calls
        response.raise_for_status() # Check for HTTP errors
        print("Status Code:", response.status_code)
        response_data = response.json()
        print("Assistant Response:", response_data.get("response", "No 'response' key found."))
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        if response is not None and response.content:
            try:
                print("Error details:", response.json())
            except json.JSONDecodeError:
                print("Error details (non-JSON):", response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
```
Run it from your project root (with the Flask server running):
```bash
python test_api.py
```

## 7. Design Decisions & Challenges

*   **Embedding Model:** `all-MiniLM-L6-v2` was chosen for its balance of performance and efficiency, suitable for semantic similarity tasks on general text.
*   **Similarity Search:** Exact cosine similarity search is used, which is appropriate and accurate for the dataset size (50-60 emails). For larger datasets, Approximate Nearest Neighbor (ANN) techniques (e.g., using FAISS) would be considered to maintain performance, trading a small amount of accuracy for significant speed gains.
*   **Generative Model:** Google's Gemini Pro model (or specify the exact one like `gemini-2.5-flash-preview-04-17` if you used that) is used via the `google-genai` SDK for its strong contextual understanding and generation capabilities.
*   **API Key Management:** API keys are managed via environment variables for security, preventing them from being hardcoded into the source.
*   **Error Handling:** The API includes basic validation for incoming requests and `try-except` blocks to catch potential errors during processing, returning user-friendly JSON error messages.

## 8. Future Improvements

*   **ANN Integration:** For larger datasets, implement FAISS or a similar library for faster similarity searches.
*   **User Interface:** Develop a simple web front-end for easier interaction.