import chromadb
from chromadb.config import Settings


# Connect to the ChromaDB HTTP server
client = chromadb.HttpClient(host="93.127.166.118", port=3333)

# List all collections
collections = client.list_collections()

print("Collections in ChromaDB:")
for col in collections:
    name = col.name if hasattr(col, 'name') else col['name']
    print(f"- {name}")
    # Get the collection object
    collection = client.get_collection(name)
    count = collection.count()
    print(f"  Number of items: {count}")
