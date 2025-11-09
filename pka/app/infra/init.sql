CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    path            TEXT UNIQUE NOT NULL,
    title           TEXT NOT NULL,
    type            VARCHAR(32) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    confidentiality_tag VARCHAR(32) NOT NULL DEFAULT 'private',
    sha256          CHAR(64) NOT NULL,
    size            BIGINT NOT NULL,
    meta            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    text            TEXT NOT NULL,
    start_line      INTEGER,
    end_line        INTEGER,
    page_no         INTEGER,
    token_count     INTEGER,
    embedding       vector(768),
    meta            JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT uq_chunks_document_ordinal UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS qa_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question        TEXT NOT NULL,
    mode            VARCHAR(32) NOT NULL,
    llm_version     VARCHAR(64) NOT NULL,
    prompt_version  VARCHAR(64) NOT NULL,
    template_hash   VARCHAR(64) NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    latency_ms      INTEGER,
    abstained       BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS qa_contexts (
    id              SERIAL PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES qa_runs(id) ON DELETE CASCADE,
    chunk_id        INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    rank            INTEGER NOT NULL,
    score_bm25      DOUBLE PRECISION,
    score_embed     DOUBLE PRECISION,
    score_rerank    DOUBLE PRECISION,
    rationale       TEXT
);

CREATE INDEX IF NOT EXISTS idx_qa_contexts_run_id ON qa_contexts (run_id);
CREATE INDEX IF NOT EXISTS idx_qa_contexts_chunk_id ON qa_contexts (chunk_id);

CREATE TABLE IF NOT EXISTS qa_answers (
    run_id          UUID PRIMARY KEY REFERENCES qa_runs(id) ON DELETE CASCADE,
    answer_json     JSONB NOT NULL
);

-- Ensure updated_at is maintained on documents.
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
BEFORE UPDATE ON documents
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();
