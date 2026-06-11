from fastmcp import FastMCP
from search_chroma import search_chroma
import json

# Initialize FastMCP
mcp = FastMCP("EcomServer")

@mcp.tool()
def search_products(query: str, n_results: int = 5, product_id: str = None) -> str:
    """
    Search for a product based on the semantic query and return relevant information.
    
    Args:
        query (str): The search query containing product identifiers or attributes.
        n_results (int): Number of top results to return.
        product_id (str): Optional specific product ID to filter by.
        
    Returns:
        str: A JSON-formatted string containing the search results.
    """
    try:
        results = search_chroma(query_text=query, n_results=n_results, product_id=product_id)
        
        # Format the results
        formatted_results = []
        if results and "documents" in results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if "metadatas" in results else {}
                dist = results["distances"][0][i] if "distances" in results else None
                formatted_results.append({
                    "id": results["ids"][0][i] if "ids" in results else None,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                })
        return json.dumps(formatted_results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.prompt()
def product_search_prompt() -> str:
    """
    Provides instructions and context for searching products and their information.
    """
    return (
        "You are an AI assistant capable of searching for products, their information, and their availability. "
        "Use the `search_products` tool to query the Chroma vector database to find products that match the user's requirements. "
        "Provide clear, concise answers and present the product information retrieved from the search."
    )

if __name__ == "__main__":
    mcp.run()