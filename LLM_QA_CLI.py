import os
import re
import string
from dotenv import load_dotenv

# Import the Google GenAI SDK
from google import genai
from google.genai.errors import APIError

# --- Configuration ---
# Load environment variables (API Key) from the .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY not found. Please set it in a .env file.")
    exit()

# Initialize the Gemini client
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    exit()

# --- Core Logic ---
def basic_preprocess(text: str) -> str:
    """Applies basic text preprocessing: lowercasing and punctuation removal."""
    
    # Lowercasing
    text = text.lower()
    
    # Punctuation Removal
    # Uses str.maketrans to efficiently remove all standard punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization (demonstrated by splitting and re-joining)
    tokens = text.split()
    processed_text = " ".join(tokens)
    
    return processed_text

def get_answer_from_llm(question: str) -> str:
    """Constructs a prompt and gets an answer from the Gemini model."""
    
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


def main():
    print("--- LLM Q&A Command-Line Interface ---")
    
    # 1. Accept natural-language question
    user_question = input("\nEnter your question: ")
    
    if not user_question.strip():
        print("No question entered. Exiting.")
        return

    # 2. Apply basic preprocessing (for display/requirement fulfillment)
    processed_q = basic_preprocess(user_question)
    
    print(f"\nOriginal Question: {user_question}")
    print(f"Preprocessed Text: {processed_q}")
    
    # 3. Construct prompt and send to LLM API
    print("\nSending question to LLM...")
    # The original question is sent for better LLM performance
    llm_answer = get_answer_from_llm(user_question) 
    
    # 4. Display the final answer
    print("\n==================================")
    print("🤖 Final LLM Answer:")
    print(llm_answer)
    print("==================================")


if __name__ == "__main__":
    main()