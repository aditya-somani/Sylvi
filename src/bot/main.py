import os
import logging
import asyncio
from typing import Any

from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

from src.config import settings
from src.memory.profile import ProfileMemoryDB
from src.bot.workers import ingestion_worker
from src.bot.handlers import (
    start_handler,
    help_handler,
    facts_handler,
    button_handler,
    voice_handler,
    photo_handler,
    text_and_link_handler,
)

logger = logging.getLogger(__name__)


async def reminder_worker(application) -> None:
    """
    Background worker that polls the database for due reminders
    and dispatches them to their corresponding chats.
    """
    logger.info("Background Reminder Worker started.")
    db = ProfileMemoryDB()
    while True:
        try:
            due_reminders = db.get_pending_reminders()
            for r in due_reminders:
                rid = r["id"]
                chat_id = r["chat_id"]
                text = r["reminder_text"]
                logger.info(f"Triggering due reminder {rid} for chat {chat_id}")
                try:
                    await application.bot.send_message(
                        chat_id=chat_id,
                        text=f"⏰ **Reminder**: {text}",
                        parse_mode="Markdown"
                    )
                    db.mark_reminder_sent(rid)
                    logger.info(f"Reminder {rid} sent and marked successfully.")
                except Exception as send_err:
                    logger.error(f"Failed to send reminder {rid} to chat {chat_id}: {str(send_err)}")
                    db.mark_reminder_sent(rid)
        except Exception as e:
            logger.exception("Error in reminder worker loop")
        
        await asyncio.sleep(10)


# --- Life-cycle Hooks ---

async def post_init(application) -> None:
    """Triggered after bot application starts. Initializes queue and worker thread."""
    os.makedirs("temp_downloads", exist_ok=True)
    
    # Create the asyncio.Queue
    ingestion_queue = asyncio.Queue()
    application.bot_data["ingestion_queue"] = ingestion_queue
    
    # Start the worker thread
    application.bot_data["worker_task"] = asyncio.create_task(
        ingestion_worker(ingestion_queue, application)
    )
    
    # Start the reminder task
    application.bot_data["reminder_task"] = asyncio.create_task(
        reminder_worker(application)
    )


async def post_shutdown(application) -> None:
    """Triggered on bot shutdown. Safely cancels background worker tasks."""
    worker_task = application.bot_data.get("worker_task")
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            logger.info("Background worker task successfully cancelled.")
            
    reminder_task = application.bot_data.get("reminder_task")
    if reminder_task:
        reminder_task.cancel()
        try:
            await reminder_task
        except asyncio.CancelledError:
            logger.info("Background reminder task successfully cancelled.")


# --- Application Builder Hook ---

def create_bot_app() -> Any:
    """Builds and wires up the Telegram Application."""
    if not settings.TELEGRAM_BOT_TOKEN or settings.TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
        raise ValueError("TELEGRAM_BOT_TOKEN settings is empty or using placeholder.")
        
    application = (
        ApplicationBuilder()
        .token(settings.TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    
    # Add Command Handlers
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("facts", facts_handler))
    
    # Add callback handler for button clicks (deletion)
    application.add_handler(CallbackQueryHandler(button_handler))
    
    # Add Media Handlers
    application.add_handler(MessageHandler(filters.VOICE, voice_handler))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    
    # Fallback to Text Handler (Queries or URLs)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_and_link_handler))
    
    return application


def main() -> None:
    """Entry point for running the bot app."""
    # Configure root logger format if run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Initializing Sylvi Bot...")
    app = create_bot_app()
    app.run_polling()


if __name__ == "__main__":
    main()
