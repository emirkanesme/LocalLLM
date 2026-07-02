import json
import sqlite3

from config import DB_NAME


def _connect(db_name=DB_NAME):
    return sqlite3.connect(db_name)


def init_db(db_name=DB_NAME, quiet=False):
    """Create tables if they do not exist; migrate legacy schema if needed."""
    conn = _connect(db_name)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
    )
    if cursor.fetchone():
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(documents)")}
        if "term_freq" not in columns:
            cursor.execute("DROP TABLE documents")
            if not quiet:
                print("Eski şema algılandı — documents tablosu yeniden oluşturuluyor.")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            term_freq TEXT NOT NULL,
            source_name TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_name)"
    )

    conn.commit()
    conn.close()

    if not quiet:
        print(f"Veritabanı '{db_name}' hazır.")


def clear_documents(db_name=DB_NAME):
    """Remove all ingested chunks and index metadata (for full re-ingest)."""
    conn = _connect(db_name)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")
    cursor.execute("DELETE FROM metadata")
    conn.commit()
    conn.close()


def get_document_count(db_name=DB_NAME):
    conn = _connect(db_name)
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    return count


def set_metadata(key, value, db_name=DB_NAME):
    conn = _connect(db_name)
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        (key, json.dumps(value)),
    )
    conn.commit()
    conn.close()


def get_metadata(key, db_name=DB_NAME):
    conn = _connect(db_name)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?", (key,)
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def insert_document(content, term_freq, source_name, db_name=DB_NAME):
    conn = _connect(db_name)
    conn.execute(
        "INSERT INTO documents (content, term_freq, source_name) VALUES (?, ?, ?)",
        (content, json.dumps(term_freq), source_name),
    )
    conn.commit()
    conn.close()


def fetch_all_documents(db_name=DB_NAME):
    conn = _connect(db_name)
    rows = conn.execute(
        "SELECT content, term_freq, source_name FROM documents"
    ).fetchall()
    conn.close()
    return [
        {
            "content": row[0],
            "term_freq": json.loads(row[1]),
            "source": row[2],
        }
        for row in rows
    ]


if __name__ == "__main__":
    init_db()
