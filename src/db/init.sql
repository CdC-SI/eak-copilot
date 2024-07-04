CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS pg_trgm;


CREATE TABLE sources (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL
);

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    language TEXT DEFAULT 'de',
    text TEXT NOT NULL,                     -- Text associated with the vector
    embedding vector(1536),                 -- A vector of dimension 1536
    url TEXT NOT NULL,                      -- URL associated with the vector
    source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT now(),   -- Timestamp when the record was created
    modified_at TIMESTAMP DEFAULT now()   -- Timestamp when the record was last modified
);

-- Erstelle eine Tabelle namens 'data' für die Verwaltung der Informationen
CREATE TABLE questions (
    id SERIAL PRIMARY KEY,
    language TEXT DEFAULT 'de',
    text TEXT NOT NULL,
    embedding vector(1536),
    answer_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    url TEXT NOT NULL,
    source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Erstelle eine Trigger-Funktion, um modified_at zu aktualisieren
CREATE OR REPLACE FUNCTION update_modified_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.modified_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Erstelle den Trigger, der die Trigger-Funktion auslöst
CREATE TRIGGER update_data_modified_at
BEFORE UPDATE ON questions
FOR EACH ROW
EXECUTE FUNCTION update_modified_at();