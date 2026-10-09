"""Postgres connection with the pgvector types registered."""

import psycopg
from pgvector.psycopg import register_vector

from sentinel1_rag.config import DATABASE_URL


def connect() -> psycopg.Connection:
    """Open an autocommit connection.

    In autocommit mode every `with conn.transaction():` block is a real transaction
    (BEGIN ... COMMIT) rather than a savepoint inside one long implicit transaction.
    """
    try:
        conn = psycopg.connect(DATABASE_URL, autocommit=True)
    except psycopg.OperationalError as exc:
        raise RuntimeError(f"cannot connect to Postgres; is it running? (make up)\n{exc}") from exc
    register_vector(conn)
    return conn
