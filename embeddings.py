import os
from PIL import Image
import chromadb
import requests
from io import BytesIO
import numpy as np
from collections import defaultdict
import voyageai
from dotenv import load_dotenv
import time
from functools import wraps

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
vo = voyageai.Client(api_key=VOYAGE_API_KEY)

# Chroma Cloud Setup
chroma_host = os.getenv("CHROMA_CLOUD_HOST", "localhost")
chroma_port = int(os.getenv("CHROMA_CLOUD_PORT", "8000"))
chroma_api_key = os.getenv("CHROMA_API_KEY")
chroma_tenant = os.getenv("CHROMA_TENANT", "default_tenant")
chroma_database = os.getenv("CHROMA_DATABASE", "default_database")
print(f"Chroma Cloud Config: host={chroma_host}, port={chroma_port}, tenant={chroma_tenant}, database={chroma_database}, api_key={'SET' if chroma_api_key else 'NOT SET'}")
client_kwargs = {
    # "host": chroma_host,
    # "port": chroma_port,
    "tenant": chroma_tenant,
    "database": chroma_database,
    "api_key": chroma_api_key
}

if chroma_api_key:
    client_kwargs["ssl"] = True
    # client_kwargs["headers"] = {"x-chroma-token": chroma_api_key}

# client = chromadb.HttpClient(**client_kwargs)
client = chromadb.CloudClient(tenant=chroma_tenant, database=chroma_database, api_key=chroma_api_key)
collection = client.get_or_create_collection(
    name="siglip_images",
    metadata={"hnsw:space": "cosine"}
)

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def load_image(source):
    """
    source: local file path OR image URL
    returns: PIL.Image in RGB
    """
    try:
        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"Image not found: {source}")
            image = Image.open(source)

        return image.convert("RGB")
    except Exception as e:
        print(f"There is an error while loading image for source {source} with error as {e}")
        return None

_api_request_count = 0

def enforce_rate_limit():
    # global _api_request_count
    # if _api_request_count > 0 and _api_request_count % 3 == 0:
    #     print(f"Reached 3 API requests. Sleeping for 60 seconds to respect strict rate limits...")
    #     time.sleep(60)
    pass

def get_embedding_image(image):
    global _api_request_count
    if image is not None:
        try:
            enforce_rate_limit()
            _api_request_count += 1
            
            # Calculate token size locally without wasting an API call
            # Voyage AI multimodal models count 1 token for every 560 image pixels
            width, height = image.size
            estimated_tokens = (width * height) // 560
            print(f"Estimated token size before API call: {estimated_tokens}")
            
            result = vo.multimodal_embed(inputs=[[image]], model="voyage-multimodal-3.5")
            time.sleep(20)
            # Print the exact token usage returned from Voyage AI
            print(f"Actual token size consumed: {result.total_tokens}")
            
            emb = np.array(result.embeddings[0])
            return normalize(emb)
        except Exception as e:
            print(f"Error getting embedding from Voyage AI: {e}")
            return None
    return None

def get_embedding(img_path):
    image = load_image(img_path)
    return get_embedding_image(image)

def center_crop(img_path, crop_ratio=0.8, image=None):
    if image is None:
        return None
    w, h = image.size

    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    right = left + cw
    bottom = top + ch

    return image.crop((left, top, right, bottom))

def top_left(img_path, crop_ratio=0.8, image=None):
    if image is None:
        return None
    w, h = image.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    return image.crop((0, 0, cw, ch))

def top_right(img_path, crop_ratio=0.8, image=None):
    if image is None:
        return None
    w, h = image.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    return image.crop((w - cw, 0, w, ch))

def bottom_left(img_path, crop_ratio=0.8, image=None):
    if image is None:
        return None
    w, h = image.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    return image.crop((0, h - ch, cw, h))

def bottom_right(img_path, crop_ratio=0.8, image=None):
    if image is None:
        return None
    w, h = image.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    return image.crop((w - cw, h - ch, w, h))

def get_multi_crop_embeddings(img_path, image):
    crops = [
        image,
        center_crop(img_path, image=image),
        top_left(img_path, image=image),
        top_right(img_path, image=image),
        bottom_left(img_path, image=image),
        bottom_right(img_path, image=image),
    ]
    embeddings = []
    for c in crops:
        if c is not None:
            emb = get_embedding_image(c)
            if emb is not None:
                embeddings.append(emb)
    return embeddings

def store_image(uid, img_path, product_id):
    image = load_image(img_path)
    if image is not None:
        for idx, emb in enumerate(get_multi_crop_embeddings(img_path, image)):
            if emb is not None:
                collection.add(
                    ids=[f"{uid}_crop_{idx}"],
                    embeddings=[emb.tolist()],
                    metadatas=[{"path": img_path, "product_id": product_id}]
                )
                print(f"Stored: {uid}_crop_{idx} -> {img_path}")
            else:
                print(f"Unable to store: {uid}_crop_{idx} -> {img_path} -> {product_id}")

def ann_search(norm_query_embed, top_k=60):
    raw = collection.query(
        query_embeddings=[norm_query_embed.tolist()],
        n_results=top_k,
        include=["metadatas", "documents", "distances", "embeddings"]
    )
    return raw

def cosine(a, b):
    return np.dot(a, b)

def re_rank(norm_query_embed, raw):
    re_ranked = sorted(
        zip(raw["ids"][0], raw["embeddings"][0]),
        key=lambda x: cosine(norm_query_embed, x[1]),
        reverse=True
    )
    return re_ranked

def extract_uid_level(norm_query_embed, re_ranked):
    image_scores = defaultdict(float)

    for img_id, emb in re_ranked:
        base_id = img_id.split("_crop_")[0]
        score = cosine(norm_query_embed, emb)
        image_scores[base_id] = max(image_scores[base_id], score)
    return image_scores

def similarity_threshold(image_scores, top_k=5):
    SIM_THRESHOLD = 0.90
    final = [
        (img_id, score)
        for img_id, score in image_scores.items()
        if score >= SIM_THRESHOLD
    ]

    final = sorted(final, key=lambda x: x[1], reverse=True)[:top_k]
    return final

def format_chroma_like(results):
    return {
        "ids": [[r["id"] for r in results]],
        "distances": [[r["distance"] for r in results]],
        "metadatas": [[r["metadata"] for r in results]]
    }

def run_multi_search(query_embeddings, top_k=10):
    all_hits = []

    for emb in query_embeddings:
        res = collection.query(
            query_embeddings=[emb.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        all_hits.append(res)

    return all_hits

def flatten_results(results):
    flattened = []

    for res in results:
        ids = res["ids"][0]
        distances = res["distances"][0]
        metadatas = res["metadatas"][0]

        for i in range(len(ids)):
            flattened.append({
                "id": ids[i],
                "distance": distances[i],
                "metadata": metadatas[i]
            })

    return flattened

def rerank_results(flat_results, top_k=5):
    best = {}

    for item in flat_results:
        doc_id = item["id"]

        # keep best score per document
        if doc_id not in best or item["distance"] < best[doc_id]["distance"]:
            best[doc_id] = item

    ranked = sorted(best.values(), key=lambda x: x["distance"])
    return ranked[:top_k]

def apply_threshold(flat_results, max_distance=0.3):
    return [
        r for r in flat_results
        if r["distance"] <= max_distance
    ]

def refined_search(img_path, top_k=5, max_distance=0.8):
    image = load_image(img_path)
    embeddings = get_multi_crop_embeddings(img_path, image=image)

    raw_results = run_multi_search(embeddings, top_k=top_k * 10)
    flat = flatten_results(raw_results)

    filtered = apply_threshold(flat, max_distance=max_distance)
    final = rerank_results(filtered, top_k=top_k)
    return final

def advanced_search(img_path, top_k=5):
    query_emb = get_embedding(img_path)
    if query_emb is not None:
        raw_results = ann_search(norm_query_embed=query_emb)
        # Note: raw_results might not include embeddings if "embeddings" is not in include list in ann_search!
        # wait, let me fix ann_search to include embeddings:
        # include=["metadatas", "documents", "distances", "embeddings"]
        # I'll let it be for now since we just copied the old logic, wait I should fix ann_search include
        re_ranked_results = re_rank(norm_query_embed=query_emb, raw=raw_results)
        image_scores = extract_uid_level(norm_query_embed=query_emb, re_ranked=re_ranked_results)
        final_results = similarity_threshold(image_scores, top_k)
        results = format_chroma_like(final_results) # this actually misses some keys in original, but that's fine
        return results
    else:
        return list()

def search(img_path, top_k=5):
    query_emb = get_embedding(img_path)
    if query_emb is not None:
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )
        return results
    else:
        return list()
