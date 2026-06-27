import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

from src.config import settings
from src.memory.profile import ProfileMemoryDB

logger = logging.getLogger(__name__)

# --- Background Ingestion Worker ---

async def ingestion_worker(queue: asyncio.Queue, application) -> None:
    """
    Background worker consuming tasks from the async queue.
    Processes voice notes, photos, and links sequentially to prevent blocking the bot loop.
    """
    logger.info("Background Ingestion Worker started.")
    while True:
        task = await queue.get()
        chat_id = task["chat_id"]
        message_id = task["message_id"]
        input_type = task["input_type"]
        raw_content = task["raw_content"]
        metadata = task["metadata"]
        
        try:
            # Let user know we are actively working on it
            await application.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            
            # Lazy import graph to avoid dependency cycles / slow starts
            from src.ingestion.graph import ingestion_graph
            
            # Execute Ingestion Pipeline
            final_state = await ingestion_graph.ainvoke({
                "input_type": input_type,
                "raw_content": raw_content,
                "metadata": metadata
            })
            
            # Cleanup temporary file if it was a downloaded file
            if input_type in ["voice", "image"] and os.path.exists(raw_content):
                try:
                    os.remove(raw_content)
                    logger.info(f"Removed temp file: {raw_content}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to delete temp file {raw_content}: {str(cleanup_err)}")
            
            chunks_count = len(final_state.get("chunks", []))
            
            # Reply directly to the message that triggered the ingestion!
            await application.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"✅ **Memory Ingested Successfully!**\n\n"
                    f"• **Type**: `{input_type}`\n"
                    f"• **Information Chunks**: {chunks_count}\n\n"
                    f"I've indexed this content in your vector memory. You can ask me about it anytime!"
                ),
                reply_to_message_id=message_id,
                parse_mode="Markdown"
            )
            
        except Exception as e:
            logger.exception(f"Error processing background ingestion task for chat {chat_id}")
            
            # Notify user of failure as a reply to their message
            try:
                await application.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"❌ **Ingestion Failed**\n\n"
                        f"I had trouble digesting that {input_type} memory.\n"
                        f"**Error**: {str(e)}"
                    ),
                    reply_to_message_id=message_id,
                    parse_mode="Markdown"
                )
            except Exception as notify_err:
                logger.error(f"Failed to send failure notification to user: {str(notify_err)}")
                
        finally:
            queue.task_done()


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

async def post_shutdown(application) -> None:
    """Triggered on bot shutdown. Safely cancels background worker tasks."""
    worker_task = application.bot_data.get("worker_task")
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            logger.info("Background worker task successfully cancelled.")


# --- Command Handlers ---

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcomes the user and explains bot capabilities."""
    welcome_text = (
        "👋 **Hello! I am Sylvi, your personal memory copilot.**\n\n"
        "I am designed to act as an extension of your digital mind. You can send me "
        "almost anything, and I'll keep it safe in my memory for you:\n\n"
        "📥 **How to Feed My Memory:**\n"
        "• Send **links/URLs** — I'll scrape, summarize, and index them.\n"
        "• Send **voice notes** — I'll transcribe and store the content.\n"
        "• Send **images/photos** — I'll analyze and caption what I see.\n"
        "• Type raw text — If it has links, I'll ingest them; otherwise, I'll reply using what I know!\n\n"
        "⚙️ **Available Commands:**\n"
        "• `/facts` — View all your SQLite-stored profile facts.\n"
        "• `/help` — Show this guide again."
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provides usage instructions."""
    await start_handler(update, context)


async def facts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Queries and renders stored profile facts, providing interactive buttons for deletion."""
    db = ProfileMemoryDB()
    facts = db.get_all_facts()
    
    if not facts:
        await update.message.reply_text(
            "🧠 **Your profile memory is currently empty.**\n"
            "Tell me facts about yourself (e.g. 'I prefere Python', 'My name is Adi') "
            "and I will save them automatically!",
            parse_mode="Markdown"
        )
        return
        
    response = "🧠 **Your Stored Profile Facts:**\n\n"
    keyboard_buttons = []
    
    for fact in facts:
        fid = fact["id"]
        text = fact["fact"]
        response += f"• **#{fid}**: {text}\n"
        
        # Abbreviate button text if it is too long to prevent Telegram formatting issues
        btn_text = text[:25] + "..." if len(text) > 25 else text
        keyboard_buttons.append([
            InlineKeyboardButton(f"🗑️ Delete #{fid}: {btn_text}", callback_data=f"delete_fact:{fid}")
        ])
        
    reply_markup = InlineKeyboardMarkup(keyboard_buttons)
    await update.message.reply_text(
        response,
        parse_mode="Markdown",
        reply_markup=reply_markup
    )


# --- Callback Query Handler ---

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles callback signals from inline buttons (e.g. fact deletion)."""
    query = update.callback_query
    await query.answer()
    
    data = query.data or ""
    if data.startswith("delete_fact:"):
        fact_id = int(data.split(":")[1])
        db = ProfileMemoryDB()
        
        success = db.delete_fact(fact_id)
        if success:
            await query.edit_message_text(
                text=f"🗑️ **Fact #{fact_id} deleted successfully!**",
                parse_mode="Markdown"
            )
        else:
            await query.edit_message_text(
                text=f"❌ **Deletion Failed**: Fact #{fact_id} could not be found.",
                parse_mode="Markdown"
            )


# --- Media Ingestion Queue Handlers ---

async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Downloads voice file and places it in the background worker queue."""
    voice = update.message.voice
    if not voice:
        return
        
    await update.message.reply_text(
        "🎙️ **Voice note received.**\nTranscribing and indexing in the background...",
        reply_to_message_id=update.message.message_id,
        parse_mode="Markdown"
    )
    
    # Download file using telegram API
    file = await context.bot.get_file(voice.file_id)
    file_path = os.path.join("temp_downloads", f"{voice.file_id}.ogg")
    await file.download_to_drive(file_path)
    
    # Push to queue
    queue = context.application.bot_data["ingestion_queue"]
    await queue.put({
        "chat_id": update.effective_chat.id,
        "message_id": update.message.message_id,
        "input_type": "voice",
        "raw_content": file_path,
        "metadata": {
            "source": "telegram",
            "message_id": str(update.message.message_id)
        }
    })


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Downloads image file and places it in the background worker queue."""
    if not update.message.photo:
        return
        
    # Get highest resolution image size
    photo = update.message.photo[-1]
    
    await update.message.reply_text(
        "🖼️ **Image received.**\nAnalyzing, captioning and indexing in the background...",
        reply_to_message_id=update.message.message_id,
        parse_mode="Markdown"
    )
    
    # Download file
    file = await context.bot.get_file(photo.file_id)
    file_path = os.path.join("temp_downloads", f"{photo.file_id}.jpg")
    await file.download_to_drive(file_path)
    
    # Push to queue
    queue = context.application.bot_data["ingestion_queue"]
    await queue.put({
        "chat_id": update.effective_chat.id,
        "message_id": update.message.message_id,
        "input_type": "image",
        "raw_content": file_path,
        "metadata": {
            "source": "telegram",
            "message_id": str(update.message.message_id)
        }
    })


async def text_and_link_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Checks incoming text for web links:
    - If links are found, places them into the background scraping queue.
    - Otherwise, runs the synchronous query RAG pipeline and replies directly.
    """
    text = update.message.text or ""
    
    # 1. Parse for URLs
    urls = []
    if update.message.entities:
        for entity in update.message.entities:
            if entity.type == "url":
                url = text[entity.offset : entity.offset + entity.length]
                urls.append(url)
            elif entity.type == "text_link":
                urls.append(entity.url)
                
    if urls:
        queue = context.application.bot_data["ingestion_queue"]
        for url in urls:
            await queue.put({
                "chat_id": update.effective_chat.id,
                "message_id": update.message.message_id,
                "input_type": "link",
                "raw_content": url,
                "metadata": {
                    "source": "telegram",
                    "message_id": str(update.message.message_id),
                    "source_url": url
                }
            })
        await update.message.reply_text(
            f"🔗 **Detected {len(urls)} link(s).**\nScraping and building memory index in the background...",
            reply_to_message_id=update.message.message_id,
            parse_mode="Markdown"
        )
        return

    # 2. Process conversational RAG Query
    # Send interactive chat action typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    # Lazy import graph to avoid circular dependency
    from src.query.graph import query_graph
    
    try:
        final_state = await query_graph.ainvoke({
            "query": text,
            "chat_id": str(update.effective_chat.id),
            "current_time": datetime.utcnow().isoformat()
        })
        
        answer = final_state.get("answer") or "Sorry, I couldn't process your request."
        
        await update.message.reply_text(
            answer,
            reply_to_message_id=update.message.message_id
        )
    except Exception as e:
        logger.exception("Error during Query Graph execution")
        await update.message.reply_text(
            f"❌ **Query Failed**\n\nSorry, I encountered an error while retrieving facts:\n`{str(e)}`",
            reply_to_message_id=update.message.message_id,
            parse_mode="Markdown"
        )


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
