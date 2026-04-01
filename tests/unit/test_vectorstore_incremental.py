from app.rag.chunking import Chunk
from app.rag.vectorstore.chroma_store import ChromaVectorStore


def test_vectorstore_memory_delete_by_source_ids() -> None:
    store = ChromaVectorStore(collection="t", persist_dir=".not_exists_test")
    chunks = [
        Chunk(chunk_id="a#c0", source_id="a", title="a", text="x"),
        Chunk(chunk_id="b#c0", source_id="b", title="b", text="y"),
    ]
    embs = [[1.0, 0.0], [0.0, 1.0]]
    store.upsert(chunks, embs)

    store.delete_by_source_ids(["a"])
    hits = store.query([1.0, 0.0], top_k=5)
    assert all(ch.source_id != "a" for ch, _ in hits)
