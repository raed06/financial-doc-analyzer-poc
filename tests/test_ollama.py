from utils.llm_manager import LLMManager

print("Testing Ollama connection...")
if LLMManager.test_connection():
    print("+ Ollama connection successful!")
else:
    print("+ Ollama connection failed!")
    print("Make sure Ollama is running: ollama serve")