import os
import chromadb
from chromadb.utils import embedding_functions
from mongo_docs import MongoDBHandler
from dotenv import load_dotenv

# Load environment variables
# Since migration.py is in misc/scripts, we point to the root .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# 1. Connect to Chroma DB
def get_chroma_client():
    tenant = os.environ.get("CHROMA_TENANT", "default_tenant")
    database = os.environ.get("CHROMA_DATABASE", "default_database")
    api_key = os.environ.get("CHROMA_API_KEY", "")
    
    if api_key:
        return chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
    else:
        host = os.environ.get("CHROMA_HOST", "localhost")
        port = int(os.environ.get("CHROMA_PORT", 8000))
        return chromadb.HttpClient(host=host, port=port)

chroma_client = get_chroma_client()

# 2. Configure the VoyageAI embedding function
voyage_api_key = os.environ.get("VOYAGE_API_KEY")
voyage_model = os.environ.get("VOYAGE_MODEL", "voyage-4")

if not voyage_api_key:
    raise ValueError("VOYAGE_API_KEY must be set in .env to run the migration.")

voyage_ef = embedding_functions.VoyageAIEmbeddingFunction(
    api_key=voyage_api_key,
    model_name=voyage_model
)

def initialize_chroma_collection(collection_name, ef):
    """Initialize a Chroma collection with the given embedding function."""
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=ef)
    print(f"Chroma collection '{collection_name}' initialized with VoyageAI embedding function ({voyage_model}).")
    return chroma_collection

def migrate_products_to_chroma(collection_name=None):
    """Migrate products from MongoDB to Chroma with embeddings."""
    if collection_name is None:
        collection_name = os.environ.get("CHROMA_COLLECTION", "gmg_test_parent_products_voyage")

    uri = os.environ.get("MONGO_URI")
    db = os.environ.get("MONGO_DB", "product-sports-db")
    collection = os.environ.get("MONGO_COLLECTION", "product_model")
    
    if not uri:
        raise ValueError("MONGO_URI must be set in .env to run the migration.")

    query = {"grandParent": False, "brand": "NIKE"}
    handler = MongoDBHandler(uri, db, collection)
    _, _, _, total_count = handler.fetch_documents(query, page=1, page_size=1)
    print(f"Total documents matching query: {total_count}")
    
    page_size = 100
    total_pages = (total_count + page_size - 1) // page_size

    # Initialize Chroma collection
    chroma_collection = initialize_chroma_collection(collection_name, voyage_ef)

    for page in range(1, total_pages + 1):
        documents, metadatas, ids, _ = handler.fetch_documents(query, page=page, page_size=page_size)
        chroma_collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Page {page}/{total_pages} migrated to Chroma.")

if __name__ == "__main__":
    migrate_products_to_chroma()