from __future__ import annotations

import argparse
import logging

from ..core.logging import configure_logging
from ..core.settings import settings
from ..services.index.bm25 import BM25IndexService
from ..services.index.embed import EmbeddingService
from ..services.ingest import EmailIngestService, MarkdownIngestService, PDFIngestService

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reindex knowledge sources (Markdown, PDFs, Emails)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to ingest")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for embedding requests"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum approximate tokens per chunk",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.12,
        help="Fractional overlap ratio between consecutive chunks",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for manual reindex operations."""

    configure_logging()
    args = parse_args()

    logger.info("Starting ingestion run with deterministic text-layer pipeline")
    embedding_service = EmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        timeout=settings.ollama_timeout_seconds,
        batch_size=args.batch_size,
        expected_dim=settings.vector_dim,
    )
    bm25_service = BM25IndexService(settings.bm25_index_path)
    markdown_service = MarkdownIngestService(
        source_dir=settings.knowledge_notes_dir,
        embedding_service=embedding_service,
        bm25_service=bm25_service,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap,
    )
    pdf_service = PDFIngestService(
        source_dir=settings.knowledge_pdfs_dir,
        embedding_service=embedding_service,
        bm25_service=bm25_service,
        max_tokens=args.max_tokens,
        overlap_tokens=max(1, int(args.max_tokens * args.overlap)),
    )
    email_service = EmailIngestService(
        source_dir=settings.knowledge_emails_dir,
        embedding_service=embedding_service,
        bm25_service=bm25_service,
        max_tokens=600,
        overlap_ratio=0.15,
    )
    try:
        markdown_service.ingest(limit=args.limit)
        pdf_service.ingest(limit=args.limit)
        email_service.ingest(limit=args.limit)
    finally:
        embedding_service.close()
        logger.info("Reindex completed.")


if __name__ == "__main__":
    main()

