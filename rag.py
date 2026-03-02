"""
rag.py — Retrieval module for Q&A pairs.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME

QA_PAIRS_FILE = "./qa_pairs.json"

_model      = None
_collection = None
_qa_pairs   = None

# Custom chips that bypass semantic search and play specific audio segments directly.
CUSTOM_CHIPS = [
    {
        "question": "What is the meaning of your name?",
        "start_fmt": "08:54",
        "custom_segments": [
            [534.04, 556.04],
            [559.16, 560.60],
            [577.04, 643.04],
            [649.04, 661.04],
            [667.04, 678.04],
        ],
    },
    {
        "question": "Can you sing me your favorite song?",
        "start_fmt": "00:00",
        "custom_segments": [
            [0.0, 35.8],
            [149.0, 292.0],
        ],
    },
]


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_qa_pairs():
    global _qa_pairs
    if _qa_pairs is None:
        with open(QA_PAIRS_FILE) as f:
            _qa_pairs = json.load(f)
    return _qa_pairs


def get_all_questions(popular_order=None) -> list[dict]:
    """Return chip list: custom chips first, then featured, then the rest.

    popular_order: optional dict mapping question label → play count (descending).
    When provided, the 'rest' group is sorted by popularity.
    """
    pairs = _get_qa_pairs()

    custom = [
        {"id": f"custom_{i}", "question": c["question"], "start_fmt": c["start_fmt"],
         "custom_segments": c["custom_segments"]}
        for i, c in enumerate(CUSTOM_CHIPS)
    ]

    featured = [
        {"id": i, "question": p.get("visitor_question", p.get("label", p["question"])), "start_fmt": p["start_fmt"]}
        for i, p in enumerate(pairs) if p.get("featured")
    ]

    rest = [
        {"id": i, "question": p.get("visitor_question", p.get("label", p["question"])), "start_fmt": p["start_fmt"]}
        for i, p in enumerate(pairs) if not p.get("featured")
    ]

    if popular_order:
        rest.sort(key=lambda q: -popular_order.get(q["question"], 0))

    return custom + featured + rest


def retrieve(query: str):
    """
    Find the best matching Q&A pair for a user query.
    Returns the full pair dict or None.
    """
    model      = _get_model()
    collection = _get_collection()
    pairs      = _get_qa_pairs()

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["metadatas", "distances"],
    )

    if not results["ids"][0]:
        return None

    qa_id = results["metadatas"][0][0]["qa_id"]
    return pairs[qa_id]
