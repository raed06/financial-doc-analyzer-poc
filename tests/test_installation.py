import sys
print("Python version:", sys.version)

try:
    import streamlit
    print("+ Streamlit installed")

    import crewai
    print("+ CrewAI installed")

    import langchain
    print("+ LangChain installed")

    import sentence_transformers
    print("+ Sentence Transformers installed")

    import faiss
    print("+ FAISS installed")

    import PyPDF2
    print("+ PyPDF2 installed")

    import pandas
    print("+ Pandas installed")

    print("All dependencies installed successfully")

except ImportError as e:
    print(f"Import error: {e}")
    print("Please run: pip install -r requirements.txt")