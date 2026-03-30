from app.rag.chunking import chunk_text


def test_chunk_text_basic() -> None:
    chunks = chunk_text("s1", "title", "a" * 1000, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert chunks[0].chunk_id == "s1#c0"
    assert all(c.source_id == "s1" for c in chunks)
