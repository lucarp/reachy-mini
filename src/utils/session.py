"""Session management using SQLite storage."""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage conversation sessions with SQLite storage."""

    def __init__(self, db_path: str):
        """Initialize session manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at)
            """)

            conn.commit()

    def create_session(self, session_id: str) -> str:
        """Create a new session.

        Args:
            session_id: Unique session identifier

        Returns:
            Created session ID
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                (session_id,)
            )
            conn.commit()

        logger.info(f"Created session: {session_id}")
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to session history.

        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata dictionary
        """
        metadata_json = json.dumps(metadata) if metadata else None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, metadata_json)
            )

            conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )

            conn.commit()

        logger.debug(f"Added {role} message to session {session_id}")

    def get_history(
        self,
        session_id: str,
        max_messages: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier
            max_messages: Maximum number of messages to retrieve

        Returns:
            List of message dictionaries with role, content, and metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT role, content, metadata, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, max_messages)
            )

            messages = []
            for row in cursor.fetchall():
                msg = {
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"]
                }
                if row["metadata"]:
                    msg["metadata"] = json.loads(row["metadata"])
                messages.append(msg)

            # Reverse to get chronological order
            messages.reverse()

        return messages

    def clear_session(self, session_id: str):
        """Clear all messages from a session.

        Args:
            session_id: Session identifier
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()

        logger.info(f"Cleared session: {session_id}")

    def delete_session(self, session_id: str):
        """Delete a session and all its messages.

        Args:
            session_id: Session identifier
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

        logger.info(f"Deleted session: {session_id}")

    def cleanup_old_sessions(self, days: int = 7):
        """Delete sessions older than specified days.

        Args:
            days: Number of days after which to delete sessions
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT session_id FROM sessions WHERE updated_at < ?",
                (cutoff_date,)
            )
            old_sessions = [row[0] for row in cursor.fetchall()]

            if old_sessions:
                placeholders = ",".join("?" * len(old_sessions))
                conn.execute(
                    f"DELETE FROM messages WHERE session_id IN ({placeholders})",
                    old_sessions
                )
                conn.execute(
                    f"DELETE FROM sessions WHERE session_id IN ({placeholders})",
                    old_sessions
                )
                conn.commit()

                logger.info(f"Cleaned up {len(old_sessions)} old sessions")

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get list of all sessions.

        Returns:
            List of session dictionaries with id, created_at, updated_at
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT session_id, created_at, updated_at,
                       (SELECT COUNT(*) FROM messages WHERE session_id = s.session_id) as message_count
                FROM sessions s
                ORDER BY updated_at DESC
                """
            )

            return [dict(row) for row in cursor.fetchall()]
