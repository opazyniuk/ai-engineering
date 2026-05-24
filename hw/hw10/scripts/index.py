"""
Індексація data/source.md у Qdrant collection 'chunks'.

Pipeline:
  1. читаємо data/source.md
  2. ріжемо на chunks ~500 токенів, overlap 50 (RecursiveCharacterTextSplitter)
  3. рахуємо токени через tiktoken (cl100k_base — тот енкодер, що у GPT-4)
  4. embedding batch через sentence-transformers (all-MiniLM-L6-v2, 384 dim)
  5. upsert у Qdrant з payload {text, section, token_count}

Запуск:
    python scripts/index.py
    python scripts/index.py --recreate   # повна переіндексація
"""
from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

# Дозволити імпорт app.* при запуску з кореня репо
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from qdrant_client.models import PointStruct  # noqa: E402

from app.config import settings  # noqa: E402
from app.embedder import embed  # noqa: E402
from app.vector_db import ensure_collection, get_client, recreate_collection  # noqa: E402


CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TIKTOKEN_ENC.encode(text))


def detect_section(text: str) -> str | None:
    """Витягти heading зі шматка (для красивого payload.section)."""
    for line in text.split("\n"):
        m = re.match(r"^#{1,3}\s+(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return None


def chunk_document(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens,
        # Йдемо від «крупних» до «дрібних» роздільників — зберігаємо смислову цілісність:
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/source.md", help="Path to source document")
    parser.add_argument("--recreate", action="store_true", help="Drop & recreate collection")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    src_path = Path(args.source)
    if not src_path.exists():
        print(f"✗ {src_path} not found. Run: python scripts/fetch_source.py", file=sys.stderr)
        return 1

    text = src_path.read_text(encoding="utf-8")
    total_tokens = count_tokens(text)
    print(f"→ Source: {src_path} ({len(text):,} chars · {total_tokens:,} tokens)")

    chunks = chunk_document(text)
    print(f"→ Split into {len(chunks)} chunks "
          f"(target {CHUNK_SIZE_TOKENS} tokens, overlap {CHUNK_OVERLAP_TOKENS})")

    # Збираємо payload-и до embedding-а, щоб embed-нути batch
    sections = [detect_section(c) for c in chunks]
    token_counts = [count_tokens(c) for c in chunks]
    print(f"  · avg tokens/chunk: {sum(token_counts) // len(token_counts)}")
    print(f"  · min/max: {min(token_counts)} / {max(token_counts)}")

    # Колекція
    coll = settings.qdrant_chunks_collection
    if args.recreate:
        recreate_collection(coll)
    else:
        ensure_collection(coll)

    # Batch embedding
    print(f"→ Embedding {len(chunks)} chunks (batch={args.batch_size}) ...")
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i : i + args.batch_size]
        vectors.extend(embed(batch))
        print(f"  · {min(i + args.batch_size, len(chunks))}/{len(chunks)}")

    # Upsert
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text": chunk,
                "section": section,
                "token_count": tok,
                "chunk_index": idx,
                "source": str(src_path),
            },
        )
        for idx, (chunk, vec, section, tok) in enumerate(
            zip(chunks, vectors, sections, token_counts, strict=True)
        )
    ]
    client = get_client()
    client.upsert(collection_name=coll, points=points)

    info = client.get_collection(coll)
    print(f"\n✓ Indexed {info.points_count} points into '{coll}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
