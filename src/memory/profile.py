import os
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any
from src.config import settings

logger = logging.getLogger(__name__)

class ProfileMemoryDB:
    """
    Manages the local SQLite database for storing structured user profile facts
    and queued reminders.
    """
    def __init__(self) -> None:
        self.db_path: str = settings.DATABASE_PATH
        
        # Ensure the parent directory for the database exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enables access to columns by name
        return conn

    def _init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: Personal profile facts (e.g. preferences, name, bio)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profile_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table 2: Reminders
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    reminder_text TEXT NOT NULL,
                    trigger_time TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'pending', -- 'pending' | 'sent'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("SQLite database tables initialized successfully.")

    # --- Profile Facts Operations ---

    def add_chat_message(self, chat_id: str, role: str, content: str) -> None:
        """Saves a message in the chat history table (truncated if too long, and prunes old history)."""
        content_clean = content.strip()
        if len(content_clean) > 500:
            content_clean = content_clean[:500] + "... [truncated]"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Insert the new message
            cursor.execute(
                "INSERT INTO chat_history (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, role, content_clean)
            )
            # 2. Prune history to keep only the latest 30 messages per chat_id
            cursor.execute(
                """
                DELETE FROM chat_history 
                WHERE chat_id = ? AND id NOT IN (
                    SELECT id FROM chat_history 
                    WHERE chat_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT 30
                )
                """,
                (chat_id, chat_id)
            )
            conn.commit()
            logger.info(f"Added and pruned chat message for {chat_id} ({role})")

    def add_fact(self, fact: str) -> int:
        """
        Saves a new profile fact.
        Returns the row ID of the inserted record.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO profile_facts (fact) VALUES (?)",
                (fact.strip(),)
            )
            conn.commit()
            last_id = cursor.lastrowid or 0
            logger.info(f"Added new profile fact (ID: {last_id})")
            return last_id

    def get_all_facts(self) -> List[Dict[str, Any]]:
        """
        Retrieves all stored facts alongside their SQLite primary key IDs.
        Used for context injection and conflict resolution.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, fact FROM profile_facts ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [{"id": row["id"], "fact": row["fact"]} for row in rows]

    def delete_fact(self, fact_id: int) -> bool:
        """Deletes a fact by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM profile_facts WHERE id = ?", (fact_id,))
            conn.commit()
            return cursor.rowcount > 0

    # --- Reminders Operations ---

    def add_reminder(self, chat_id: str, reminder_text: str, trigger_time: datetime) -> int:
        """
        Queues a new reminder.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO reminders (chat_id, reminder_text, trigger_time, status)
                VALUES (?, ?, ?, 'pending')
                """,
                (chat_id, reminder_text.strip(), trigger_time.isoformat())
            )
            conn.commit()
            last_id = cursor.lastrowid or 0
            logger.info(f"Scheduled reminder {last_id} for chat {chat_id} at {trigger_time}")
            return last_id

    def get_pending_reminders(self) -> List[Dict[str, Any]]:
        """
        Retrieves all reminders that are 'pending' and whose trigger_time is
        in the past relative to the current UTC timestamp.
        """
        now_str = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, chat_id, reminder_text, trigger_time 
                FROM reminders 
                WHERE status = 'pending' AND trigger_time <= ?
                """,
                (now_str,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def mark_reminder_sent(self, reminder_id: int) -> bool:
        """Marks a reminder as sent to prevent re-sending."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE reminders SET status = 'sent' WHERE id = ?",
                (reminder_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_chat_history(self, chat_id: str, limit: int = 15) -> List[Dict[str, str]]:
        """Retrieves the latest chat history for a given chat_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content FROM chat_history 
                WHERE chat_id = ? 
                ORDER BY created_at ASC 
                LIMIT ?
                """,
                (chat_id, limit)
            )
            rows = cursor.fetchall()
            return [{"role": row["role"], "content": row["content"]} for row in rows]
