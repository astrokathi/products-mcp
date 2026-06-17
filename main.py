from fastmcp import FastMCP
from search_chroma import search_chroma
from embeddings import load_image, get_embedding_image, collection
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
    
@mcp.tool()
def visual_search(image_source: str, top_k: int = 5) -> str:
    """
    Perform a visual semantic search to find similar products.
    
    Args:
        image_source: Local file path or HTTP/HTTPS URL of the image.
        top_k: Number of similar image results to return (default: 5).
        
    Returns:
        A list of matching image URLs and their associated product IDs.
    """
    image = load_image(image_source)
    if image is None:
        return f"Failed to load image from source: {image_source[:100]}..."
    
    query_emb = get_embedding_image(image)
    if query_emb is None:
        return "Failed to generate embedding for the image using Voyage AI."
    
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=top_k,
        include=["metadatas"]
    )
    
    if not results or not results["metadatas"] or not results["metadatas"][0]:
        return "No results found."
        
    metadatas = results["metadatas"][0]
    
    response_lines = ["Found the following matching images:"]
    for i, meta in enumerate(metadatas):
        path = meta.get("path", "Unknown path")
        product_id = meta.get("product_id", "Unknown product")
        response_lines.append(f"{i+1}. Product ID: {product_id} | Image URL: {path}")
        
    return "\n".join(response_lines)

@mcp.prompt()
def product_search_prompt() -> str:
    """
    Provides instructions and context for searching products and their information.
    """
    return (
        "You are an AI assistant capable of searching for products, their information, and their availability. "
        "Use the `search_products` tool to query the Chroma vector database to find products that match the user's requirements. "
        "Provide clear, concise answers and present the product information retrieved from the search, Shop now url has to be constructed in such a way https://nike.ae/en/<parent-id>.html"
        "Dont assume retailer apart from NIKE UAE and Sun & Sand Sports, apart from these two rest should be ignored"
    )

if __name__ == "__main__":
    mcp.run()