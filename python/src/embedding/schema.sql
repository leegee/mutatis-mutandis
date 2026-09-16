CREATE SCHEMA IF NOT EXISTS embedding;

CREATE TABLE IF NOT EXISTS embedding.models (
    model_id BIGSERIAL PRIMARY KEY,

    model_key TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT embedding_models_key_revision_uq
        UNIQUE (model_key, model_revision),

    CONSTRAINT embedding_models_dimension_ck
        CHECK (embedding_dimension > 0)
);

CREATE TABLE IF NOT EXISTS embedding.work (
    work_id BIGSERIAL PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES embedding.models(model_id),

    event_ids BIGINT[] NOT NULL,

    status TEXT NOT NULL DEFAULT 'pending',

    worker_id TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,

    claimed_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    lease_expires_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    observation_count INTEGER NOT NULL,
    completed_count INTEGER NOT NULL DEFAULT 0,

    last_error TEXT,

    CONSTRAINT embedding_work_status_ck
        CHECK (
            status IN (
                'pending',
                'claimed',
                'embedding',
                'completed',
                'failed'
            )
        ),

    CONSTRAINT embedding_work_observation_count_ck
        CHECK (observation_count > 0),

    CONSTRAINT embedding_work_completed_count_ck
        CHECK (
            completed_count >= 0
            AND completed_count <= observation_count
        ),

    CONSTRAINT embedding_work_event_ids_ck
        CHECK (cardinality(event_ids) = observation_count)
);

CREATE INDEX IF NOT EXISTS embedding_work_claim_idx
    ON embedding.work (status, lease_expires_at, work_id);

CREATE INDEX IF NOT EXISTS embedding_work_model_idx
    ON embedding.work (model_id, status, work_id);

CREATE TABLE IF NOT EXISTS embedding.inventory (
    event_id BIGINT NOT NULL
        REFERENCES events(event_id)
        ON DELETE CASCADE,

    model_id BIGINT NOT NULL
        REFERENCES embedding.models(model_id),

    embedding_key TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (event_id, model_id),

    CONSTRAINT embedding_inventory_key_uq
        UNIQUE (embedding_key)
);

CREATE INDEX IF NOT EXISTS embedding_inventory_model_event_idx
    ON embedding.inventory (model_id, event_id);
