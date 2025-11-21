import os
import re
import string
from dotenv import load_dotenv

from flask import Flask, render_template, request
from google import genai
from google.genai.errors import APIError

# --- Initialization ---

# 1. Initialize the Flask Application Instance
app = Flask(__name__)
print("DEBUG 1: Flask app instance created.")

# Load environment variables (API Key) from the .env file for local testing
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
print(f"DEBUG 2: API_KEY loaded: {bool(API_KEY)}")

# 2. Initialize the Gemini Client
client = None
if not API_KEY:
    print("WARNING: GEMINI_API_KEY not set. API calls will fail.")
else:
    try:
        client = genai.Client(api_key=API_KEY)
        print("DEBUG 3: Gemini Client initialized successfully.")
    except Exception as e:
        print(f"DEBUG 3 FAILED: Error initializing Gemini client: {e}")
        client = None


# --- Utility Functions ---

def basic_preprocess(text: str) -> str:
    """Applies basic text preprocessing: lowercasing and punctuation removal."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    processed_text = " ".join(tokens)
    return processed_text

def get_llm_response(question: str) -> str:
    """Sends the question to the Gemini model and returns the response."""
    if client is None:
        return "ERROR: LLM Client failed to initialize. Check GEMINI_API_KEY setting."

    system_instruction = "You are a helpful and concise Question-Answering system. Answer the user's question directly and professionally."
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=question,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except APIError as e:
        return f"An API Error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default values for the template
    context = {
        'user_question': '',
        'original_question': '',
        'processed_question': '',
        'answer': None
    }
    
    if request.method == 'POST':
        # Get the question from the HTML form (name="question")
        user_question = request.form.get('question', '').strip()
        
        if user_question:
            # 1. Store original question
            context['original_question'] = user_question
            
            # 2. Apply preprocessing
            processed_q = basic_preprocess(user_question)
            context['processed_question'] = processed_q
            
            # 3. Get LLM Response
            llm_answer = get_llm_response(user_question)
            context['answer'] = llm_answer
            
            # Retain the question text in the form field after submission
            context['user_question'] = user_question 
        else:
            # Handle empty submission
            context['answer'] = "Please enter a question."

    # Render the HTML template, passing the context dictionary
    return render_template('index.html', **context)


if __name__ == '__main__':
    print("DEBUG 4: Starting Flask server...")
    # Using 0.0.0.0 makes the server accessible externally (useful for deployment testing)
    app.run(debug=True, host='0.0.0.0', port=5000)