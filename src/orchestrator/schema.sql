-- src/orchestrator/schema.sql
CREATE TABLE IF NOT EXISTS orch_run (
    id            TEXT PRIMARY KEY,
    workflow_type TEXT NOT NULL,
    episode_id    BIGINT NOT NULL,
    callback_url  TEXT NOT NULL,
    dub_id        BIGINT,
    language      TEXT,
    audio_url     TEXT,
    bg_volume     DOUBLE PRECISION NOT NULL DEFAULT 0.15,
    status        TEXT NOT NULL DEFAULT 'running',
    current_step  TEXT NOT NULL DEFAULT '',
    attempts      INTEGER NOT NULL DEFAULT 0,
    segments_key  TEXT,
    source_lang   TEXT,
    speaker_keys  JSONB,
    r2_url        TEXT,
    duration_sec  DOUBLE PRECISION,
    segment_count INTEGER,
    nebius_job_id TEXT,
    step_deadline TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
