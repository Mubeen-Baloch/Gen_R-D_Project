import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

print("\n--- Test LangChain ChatGoogleGenerativeAI with Gemma 3 ---", flush=True)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    response = llm.invoke("Say hello world in exactly 2 words.")
    print(f"Success! Response: {response.content.strip()}", flush=True)
except Exception as e:
    print(f"Test Failed: {type(e).__name__}: {str(e)}", flush=True)
