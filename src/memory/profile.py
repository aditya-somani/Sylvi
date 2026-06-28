import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import psycopg
from psycopg.rows import dict_row
from src.config import settings

logger = logging.getLogger(__name__)

class ProfileMemoryDB:
    """
    Manages the PostgreSQL database (e.g. hosted on Supabase) for storing 
    structured user profile facts, chat history, and queued reminders.
    """
    def __init__(self) -> None:
        self.db_url: str = settings.DATABASE_URL
        if not self.db_url:
            raise ValueError(
                "DATABASE_URL environment variable is empty. "
                "Please configure a valid PostgreSQL Connection URI."
            )
        self._init_db()

    def _get_connection(self) -> psycopg.Connection:
        """Returns a connection to the PostgreSQL database."""
        # Using dict_row to match the old sqlite3.Row dict-like access patterns
        return psycopg.connect(self.db_url, row_factory=dict_row)

    def _init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cursor:
                    # Table 1: Personal profile facts
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS profile_facts (
                            id SERIAL PRIMARY KEY,
                            chat_id TEXT DEFAULT 'system',
                            fact TEXT NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    # Ensure chat_id column exists for backward compatibility with existing databases
                    cursor.execute("ALTER TABLE profile_facts ADD COLUMN IF NOT EXISTS chat_id TEXT DEFAULT 'system'")
                    
                    # Table 2: Reminders
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS reminders (
                            id SERIAL PRIMARY KEY,
                            chat_id TEXT NOT NULL,
                            reminder_text TEXT NOT NULL,
                            trigger_time TIMESTAMP WITH TIME ZONE NOT NULL,
                            status TEXT DEFAULT 'pending', -- 'pending' | 'sent'
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Table 3: Chat history table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS chat_history (
                            id SERIAL PRIMARY KEY,
                            chat_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            content TEXT NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.commit()
                    logger.info("PostgreSQL database tables verified/initialized successfully.")
        finally:
            conn.close()

    # --- Profile Facts Operations ---

    def add_chat_message(self, chat_id: str, role: str, content: str) -> None:
        """Saves a message in the chat history table (truncated if too long, and prunes old history)."""
        content_clean = content.strip()
        if len(content_clean) > 500:
            content_clean = content_clean[:500] + "... [truncated]"

        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cursor:
                    # 1. Insert the new message
                    cursor.execute(
                        "INSERT INTO chat_history (chat_id, role, content) VALUES (%s, %s, %s)",
                        (chat_id, role, content_clean)
                    )
                    # 2. Prune history to keep only the latest 30 messages per chat_id
                    cursor.execute(
                        """
                        DELETE FROM chat_history 
                        WHERE chat_id = %s AND id NOT IN (
                            SELECT id FROM chat_history 
                            WHERE chat_id = %s 
                            ORDER BY created_at DESC 
                            LIMIT 30
                        )
                        """,
                        (chat_id, chat_id)
                    )
                    conn.commit()
                    logger.info(f"Added and pruned chat message for {chat_id} ({role})")
        finally:
            conn.close()

    def add_fact(self, chat_id: str, fact: str) -> int:
        """
        Saves a new profile fact.
        Returns the row ID of the inserted record.
        """
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO profile_facts (chat_id, fact) VALUES (%s, %s) RETURNING id",
                        (chat_id, fact.strip())
                    )
                    last_id = cursor.fetchone()["id"]
                    conn.commit()
                    logger.info(f"Added new profile fact (ID: {last_id}) for chat {chat_id}")
                    return last_id
        finally:
            conn.close()

    def get_all_facts(self, chat_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all stored facts alongside their primary key IDs.
        Used for context injection and conflict resolution.
        """
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, fact FROM profile_facts WHERE chat_id = %s ORDER BY created_at DESC",
                        (chat_id,)
                    )
                    rows = cursor.fetchall()
                    return [{"id": row["id"], "fact": row["fact"]} for row in rows]
        finally:
            conn.close()

    def delete_fact(self, fact_id: int) -> bool:
        """Deletes a fact by ID."""
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM profile_facts WHERE id = %s", (fact_id,))
                    conn.commit()
                    return cursor.rowcount > 0
        finally:
            conn.close()

    # --- Reminders Operations ---

    def add_reminder(self, chat_id: str, reminder_text: str, trigger_time: datetime) -> int:
        """
        Queues a new reminder.
        """
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO reminders (chat_id, reminder_text, trigger_time, status)
                        VALUES (%s, %s, %s, 'pending') RETURNING id
                        """,
                        (chat_id, reminder_text.strip(), trigger_time)
                    )
                    last_id = cursor.fetchone()["id"]
                    conn.commit()
                    logger.info(f"Scheduled reminder {last_id} for chat {chat_id} at {trigger_time}")
                    return last_id
        finally:
            conn.close()

    def get_pending_reminders(self) -> List[Dict[str, Any]]:
        """
        Retrieves all reminders that are 'pending' and whose trigger_time is
        in the past relative to the current UTC timestamp.
        """
        # PostgreSQL handles native datetime objects directly
        from datetime import timezone
        now = datetime.now(timezone.utc)
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT id, chat_id, reminder_text, trigger_time 
                        FROM reminders 
                        WHERE status = 'pending' AND trigger_time <= %s
                        """,
                        (now,)
                    )
                    rows = cursor.fetchall()
                    # Convert trigger_time timestamp to string matching previous SQLite string-based output for API compatibility
                    results = []
                    for row in rows:
                        results.append({
                            "id": row["id"],
                            "chat_id": row["chat_id"],
                            "reminder_text": row["reminder_text"],
                            "trigger_time": row["trigger_time"].isoformat() if isinstance(row["trigger_time"], datetime) else str(row["trigger_time"])
                        })
                    return results
        finally:
            conn.close()

    def mark_reminder_sent(self, reminder_id: int) -> bool:
        """Marks a reminder as sent to prevent re-sending."""
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE reminders SET status = 'sent' WHERE id = %s",
                        (reminder_id,)
                    )
                    conn.commit()
                    return cursor.rowcount > 0
        finally:
            conn.close()

    def get_chat_history(self, chat_id: str, limit: int = 15) -> List[Dict[str, str]]:
        """Retrieves the latest chat history for a given chat_id."""
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT role, content FROM chat_history 
                        WHERE chat_id = %s 
                        ORDER BY created_at ASC 
                        LIMIT %s
                        """,
                        (chat_id, limit)
                    )
                    rows = cursor.fetchall()
                    return [{"role": row["role"], "content": row["content"]} for row in rows]
        finally:
            conn.close()

    def get_reminders_by_chat(self, chat_id: str) -> List[Dict[str, Any]]:
        """Retrieves all pending/future reminders for a specific chat_id."""
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT id, reminder_text, trigger_time, status 
                        FROM reminders 
                        WHERE chat_id = %s AND status = 'pending'
                        ORDER BY trigger_time ASC
                        """,
                        (chat_id,)
                    )
                    rows = cursor.fetchall()
                    results = []
                    for row in rows:
                        results.append({
                            "id": row["id"],
                            "reminder_text": row["reminder_text"],
                            "trigger_time": row["trigger_time"].isoformat() if isinstance(row["trigger_time"], datetime) else str(row["trigger_time"]),
                            "status": row["status"]
                        })
                    return results
        finally:
            conn.close()
