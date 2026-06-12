import os
import chromadb
import voyageai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_voyage_client():
    """Initialize and return the VoyageAI client."""
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("Warning: VOYAGE_API_KEY not found in environment variables.")
    return voyageai.Client(api_key=api_key)

def get_chroma_client():
    """Initialize and return the Chroma CloudClient (or HttpClient as fallback)."""
    tenant = os.environ.get("CHROMA_TENANT", "default_tenant")
    database = os.environ.get("CHROMA_DATABASE", "default_database")
    api_key = os.environ.get("CHROMA_API_KEY", "")
    
    if api_key:
        return chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
    else:
        # Fallback to local
        host = os.environ.get("CHROMA_HOST", "localhost")
        port = int(os.environ.get("CHROMA_PORT", 8000))
        return chromadb.HttpClient(host=host, port=port)

def get_chroma_collection(collection_name, chroma_client):
    """Initialize a Chroma collection."""
    # We do NOT pass the embedding function here to avoid conflicts
    chroma_collection = chroma_client.get_collection(name=collection_name)
    print(f"Chroma collection '{collection_name}' initialized.")
    return chroma_collection

def search_chroma(query_text, collection_name=None, n_results=5, product_id=None):
    """Search products from Chroma with embeddings."""
    if collection_name is None:
        collection_name = os.environ.get("CHROMA_COLLECTION", "gmg_test_parent_products_voyage")

    chroma_client = get_chroma_client()
    chroma_collection = get_chroma_collection(collection_name, chroma_client)
    
    # Compute the embedding manually using VoyageAI
    voyage_client = get_voyage_client()
    voyage_model = os.environ.get("VOYAGE_MODEL", "voyage-4")
    query_embeddings = voyage_client.embed([query_text], model=voyage_model).embeddings
    
    where_filter = None
    if product_id:
        where_filter = {"core__id": product_id}
    
    if where_filter:
        results = chroma_collection.query(query_embeddings=query_embeddings, n_results=n_results, where=where_filter)
    else:
        results = chroma_collection.query(query_embeddings=query_embeddings, n_results=n_results)
    
    print(f"Search results for query '{query_text}': {results}")
    return results