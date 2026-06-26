"""SQL schema for the autism detection application.

Run this script against your PostgreSQL database (e.g. psql -f schema.sql).
"""

-- Enable the pgcrypto extension for UUID generation (optional but convenient)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1️⃣ Roles (optional)
CREATE TABLE roles (
    id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE
);

-- 2️⃣ Users
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    full_name     TEXT,
    role_id       UUID REFERENCES roles(id) ON DELETE SET NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3️⃣ Patients
CREATE TABLE patients (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    date_of_birth DATE,
    gender        TEXT,
    external_id   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_patients_user_id ON patients(user_id);

-- 4️⃣ Videos
CREATE TABLE videos (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    uploaded_by     UUID NOT NULL REFERENCES users(id) ON DELETE SET NULL,
    file_path       TEXT NOT NULL,
    filename        TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    mime_type       TEXT NOT NULL,
    duration_sec    NUMERIC(6,2),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_videos_patient_id ON videos(patient_id);
CREATE INDEX idx_videos_uploaded_by ON videos(uploaded_by);

-- 5️⃣ Predictions
CREATE TABLE predictions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id          UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    model_version     TEXT NOT NULL,
    asd_probability  NUMERIC(4,3) NOT NULL,
    classification    TEXT NOT NULL,
    inference_seconds NUMERIC(6,2),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_predictions_video_id ON predictions(video_id);

-- 6️⃣ PDF Reports
CREATE TABLE pdf_reports (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id    UUID NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    generated_by     UUID NOT NULL REFERENCES users(id) ON DELETE SET NULL,
    file_path        TEXT NOT NULL,
    filename         TEXT NOT NULL,
    file_size_bytes  BIGINT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_reports_prediction_id ON pdf_reports(prediction_id);
CREATE INDEX idx_reports_generated_by ON pdf_reports(generated_by);

-- 7️⃣ Audit Log (optional)
CREATE TABLE audit_log (
    id           BIGSERIAL PRIMARY KEY,
    user_id      UUID REFERENCES users(id) ON DELETE SET NULL,
    action       TEXT NOT NULL,
    target_type  TEXT NOT NULL,
    target_id    UUID,
    ip_address   INET,
    user_agent   TEXT,
    occurred_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
