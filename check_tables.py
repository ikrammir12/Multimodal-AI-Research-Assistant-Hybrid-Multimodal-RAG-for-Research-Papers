import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("paper_tables")

data = collection.get(include=["documents", "metadatas"])

print("Total table records:", len(data["ids"]))

for table_id, document, metadata in zip(
    data["ids"],
    data["documents"],
    data["metadatas"]
):
    print("\n" + "=" * 50)
    print("ID:", table_id)
    print("Metadata:", metadata)
    print("Content:", (document or "")[:400])