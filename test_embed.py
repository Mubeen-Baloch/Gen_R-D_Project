import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

for model_name in [
    "models/gemini-embedding-001",
    "models/gemini-embedding-2-preview",
    "models/gemini-embedding-2"
]:
    try:
        print(f"Testing {model_name}...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name, 
            google_api_key=api_key
        )
        res = embeddings.embed_query("Hello world")
        print(f"Success {model_name}! len: {len(res)}", flush=True)
    except Exception as e:
        print(f"Fail {model_name}: {e}")


