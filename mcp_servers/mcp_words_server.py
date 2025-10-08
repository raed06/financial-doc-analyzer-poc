import os
import re
import sys
import logging
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier

from config.settings import settings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a shared secret for symmetric key verification
verifier = JWTVerifier(
    public_key=settings.mcp_api_key,  
    issuer="internal-auth-service",
    audience="mcp-internal-api",
    algorithm="HS256" 
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

mcp = FastMCP(name="MCP WORDS", auth=verifier)

# --- UTILITY FUNCTION ---

def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract the most frequent non-stopword keywords from a text."""
    logger.info("Extracting keywords from input text...")
    logger.info("Text: " + text)
    
    words = re.findall(r'\b\w+\b', text.lower())

    stopwords = set([
        'the', 'is', 'and', 'a', 'an', 'in', 'of', 'to', 'we', 'our', 'with',
        'as', 'for', 'on', 'that', 'this', 'by', 'are', 'be', 'or', 'it',
        'at', 'from', 'their', 'they', 'have', 'has', 'had'
    ])

    frequency = {}
    for word in words:
        if word not in stopwords and len(word) > 2:
            frequency[word] = frequency.get(word, 0) + 1

    sorted_keywords = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in sorted_keywords[:top_n]]

    logger.info("Extracted keywords: %s", keywords)
    return keywords

# --- MCP TOOL ---

@mcp.tool("extract_keywords")
def extract_keywords_tool(answer: str) -> dict:
    """
    Extracts keywords from a given text.

    Args:
        input: A text string from which to extract keywords.

    Returns:
        A dictionary with the extracted keywords.
    """
    logger.info("Received request to extract keywords.")
    try:
        keywords = extract_keywords(answer)
        return {"output": ", ".join(keywords)}
    except Exception as e:
        logger.exception("Failed to extract keywords.")
        return {"error": str(e)}

# --- START SERVER ---

if __name__ == "__main__":
    logger.info("Starting MCP server for keyword extraction...")
    mcp.run(transport="http", port=settings.mcp_words_port)
