import asyncio
from main import search_products, product_search_prompt

async def test():
    print("=== Testing MCP Prompt ===")
    prompt_text = product_search_prompt()
    print(prompt_text)
    print("\n" + "="*30 + "\n")
    
    print("=== Testing Search Products Tool ===")
    query = "Find me some Nike shoes."
    print(f"Query: {query}")
    try:
        # Note: This will actually attempt to hit ChromaDB.
        # If no DB is available, it will return the error string as defined in the tool.
        result = search_products(query)
        print("RESULT:")
        print(result)
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    asyncio.run(test())
