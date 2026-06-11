import os
import chromadb
from chromadb.utils import embedding_functions

def get_embedding_function():
    """Embedding function to use with Chroma, running locally without Ollama."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="mixedbread-ai/mxbai-embed-large-v1"
    )

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
    # We do NOT pass the embedding function here to avoid conflicts with the persisted 'ollama' metadata
    chroma_collection = chroma_client.get_collection(name=collection_name)
    print(f"Chroma collection '{collection_name}' initialized.")
    return chroma_collection

def search_chroma(query_text, collection_name="gmg_test_parent_products_mxbai", n_results=5, product_id=None):
    """Search products from Chroma with embeddings."""
    chroma_client = get_chroma_client()
    chroma_collection = get_chroma_collection(collection_name, chroma_client)
    
    # Compute the embedding manually
    emb_fn = get_embedding_function()
    query_embeddings = emb_fn([query_text])
    
    where_filter = None
    if product_id:
        where_filter = {"core__id": product_id}
    
    if where_filter:
        results = chroma_collection.query(query_embeddings=query_embeddings, n_results=n_results, where=where_filter)
    else:
        results = chroma_collection.query(query_embeddings=query_embeddings, n_results=n_results)
    
    print(f"Search results for query '{query_text}': {results}")
    return results