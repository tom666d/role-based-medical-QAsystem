import chromadb, json
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
collection = chroma_client.get_or_create_collection("medical_records")


def ingest_transcripts(json_path: str = 'transcripts.json') -> str:
    """Load transcripts.json and ingest all sections into ChromaDB."""
    with open(json_path) as f:
        records = json.load(f)

    count = 0
    for rec in records:
        pid = rec["patient_id"]
        for section in ["summary", "diagnosis", "prescription"]:
            text = rec.get(section, '').strip()
            if not text: continue
            embedding = embedder.encode(text).tolist()
            collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"patient_id": pid, "section": section}],
                ids=[f"{pid}_{section}"]
            )
            count += 1
    return f"Ingested {count} sections from {len(records)} patient records."


def query_all(question: str) -> str:
    """Doctor access: no filter — returns all matching records."""
    n   = collection.count()  # dynamically retrieve all available sections
    emb = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[emb], n_results=n)
    return _format(results)


def query_patient(question: str, patient_id: str) -> str:
    """Patient access: filter by patient_id only."""
    n   = collection.count()  # dynamically retrieve all available sections
    emb = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[emb],
        where={"patient_id": patient_id},  # ← access boundary
        n_results=n
    )
    return _format(results)


def query_prescriptions(question: str) -> str:
    """Pharmacist access: prescription sections only."""
    n   = collection.count()  # dynamically retrieve all available sections
    emb = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[emb],
        where={"section": "prescription"},  # ← access boundary
        n_results=n
    )
    return _format(results)


def _format(results: dict) -> str:
    docs  = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]
    if not docs: return "No relevant records found."
    lines = []
    for doc, meta in zip(docs, metas):
        pid = meta.get("patient_id", "?")
        sec = meta.get("section", "?")
        lines.append(f"[Patient {pid} | {sec.upper()}] {doc}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    print(ingest_transcripts())
    print(query_all("What prescriptions were given?"))
    print(query_patient("What is my diagnosis?", "P001"))
    print(query_prescriptions("What medications were prescribed?"))