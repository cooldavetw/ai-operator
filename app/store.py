import fcntl
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    """Single service owner. SQLite transactions persist state and replay together."""

    def __init__(self, path: Path):
        self.path = path
        self._owner = None

    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.close()
            raise RuntimeError("Database already owned: use exactly one API worker and one replica")
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    response TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversations_expiry ON conversations(expires_at);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA secure_delete=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def purge(self, before: float):
        with self.connect() as db:
            db.execute("DELETE FROM conversations WHERE expires_at < ?", (before,))

    def close(self):
        if self._owner is not None:
            self._owner.close()
            self._owner = None

