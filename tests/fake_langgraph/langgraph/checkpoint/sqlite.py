import pickle


class SqliteSaver:
    """Persists pickled checkpoints in the given sqlite connection, so a second 'process' really can resume."""

    def __init__(self, conn):
        self.conn = conn

    def setup(self):
        self.conn.execute("CREATE TABLE IF NOT EXISTS fake_checkpoints (thread_id TEXT PRIMARY KEY, blob BLOB)")
        self.conn.commit()

    def get(self, tid):
        row = self.conn.execute("SELECT blob FROM fake_checkpoints WHERE thread_id=?", (tid,)).fetchone()
        return pickle.loads(row[0]) if row else None

    def put(self, tid, ck):
        self.conn.execute("INSERT OR REPLACE INTO fake_checkpoints VALUES (?, ?)", (tid, pickle.dumps(ck)))
        self.conn.commit()
