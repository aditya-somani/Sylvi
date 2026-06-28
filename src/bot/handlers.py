import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.memory.profile import ProfileMemoryDB
from src.bot.helpers import run_conversational_query

logger = logging.getLogger(__name__)


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
        "• `/facts` — View all your PostgreSQL-stored profile facts.\n"
        "• `/help` — Show this guide again."
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provides usage instructions."""
    await start_handler(update, context)


async def facts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Queries and renders stored profile facts, providing interactive buttons for deletion."""
    chat_id = str(update.effective_chat.id)
    db = ProfileMemoryDB()
    facts = db.get_all_facts(chat_id)
    
    if not facts:
        await update.message.reply_text(
            "🧠 **Your profile memory is currently empty.**\n"
            "Tell me facts about yourself (e.g. 'I prefer Python', 'My name is Adi') "
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


# --- Media Message Handlers ---

async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Downloads voice file and places it in the background worker queue."""
    voice = update.message.voice
    if not voice:
        return
        
    # Send single clean status message and capture its ID
    processing_msg = await update.message.reply_text(
        "⏳ Processing...",
        reply_to_message_id=update.message.message_id
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
        "processing_msg_id": processing_msg.message_id,
        "metadata": {
            "source": "telegram",
            "message_id": str(update.message.message_id),
            "chat_id": str(update.effective_chat.id)
        }
    })


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Downloads image file and places it in the background worker queue with captions."""
    if not update.message.photo:
        return
        
    # Get highest resolution image size
    photo = update.message.photo[-1]
    caption = update.message.caption or ""
    
    # Download file
    file = await context.bot.get_file(photo.file_id)
    file_path = os.path.join("temp_downloads", f"{photo.file_id}.jpg")
    await file.download_to_drive(file_path)
    
    silent = bool(caption.strip())
    processing_msg_id = None
    
    if not silent:
        # Send status message only if there's no caption (pure ingestion)
        processing_msg = await update.message.reply_text(
            "⏳ Processing...",
            reply_to_message_id=update.message.message_id
        )
        processing_msg_id = processing_msg.message_id
        
    # Push to queue
    queue = context.application.bot_data["ingestion_queue"]
    await queue.put({
        "chat_id": update.effective_chat.id,
        "message_id": update.message.message_id,
        "input_type": "image",
        "raw_content": file_path,
        "processing_msg_id": processing_msg_id,
        "metadata": {
            "source": "telegram",
            "message_id": str(update.message.message_id),
            "chat_id": str(update.effective_chat.id),
            "caption": caption,
            "silent": silent,
            "original_text": f"[Photo] {caption}".strip() if caption else "[Photo]"
        }
    })
    
    # If there is a caption, run the conversational flow immediately
    if silent:
        await run_conversational_query(update, context, query_text=caption, original_msg_text=f"[Photo] {caption}".strip())


async def text_and_link_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Checks incoming text for web links:
    - If links are found, places them into the background scraping queue.
    - If the user replied to a media/link message, extracts context.
    - Otherwise, runs the synchronous query RAG pipeline, manages chat history, and replies directly.
    """
    text = update.message.text or ""
    chat_id = str(update.effective_chat.id)
    
    # 1. Check for replies first (Context Ingestion via Reply)
    reply_to = update.message.reply_to_message
    if reply_to:
        # Check if replied-to message has a URL
        rep_urls = []
        if reply_to.entities:
            for entity in reply_to.entities:
                if entity.type == "url":
                    url = reply_to.text[entity.offset : entity.offset + entity.length]
                    rep_urls.append(url)
                elif entity.type == "text_link":
                    rep_urls.append(entity.url)
                    
        if rep_urls:
            silent = bool(text.strip())
            processing_msg_id = None
            if not silent:
                processing_msg = await update.message.reply_text(
                    "⏳ Processing...",
                    reply_to_message_id=update.message.message_id
                )
                processing_msg_id = processing_msg.message_id
                
            queue = context.application.bot_data["ingestion_queue"]
            for url in rep_urls:
                await queue.put({
                    "chat_id": update.effective_chat.id,
                    "message_id": update.message.message_id,
                    "input_type": "link",
                    "raw_content": url,
                    "processing_msg_id": processing_msg_id,
                    "metadata": {
                        "source": "telegram",
                        "message_id": str(update.message.message_id),
                        "source_url": url,
                        "user_annotation": text,
                        "silent": silent,
                        "chat_id": chat_id,
                        "original_text": f"[Reply to link] {text}".strip()
                    }
                })
            if silent:
                await run_conversational_query(update, context, query_text=text, original_msg_text=text)
            return
            
        # Check if replied-to message has a photo
        if reply_to.photo:
            photo = reply_to.photo[-1]
            silent = bool(text.strip())
            processing_msg_id = None
            if not silent:
                processing_msg = await update.message.reply_text(
                    "⏳ Processing...",
                    reply_to_message_id=update.message.message_id
                )
                processing_msg_id = processing_msg.message_id
            
            file = await context.bot.get_file(photo.file_id)
            file_path = os.path.join("temp_downloads", f"{photo.file_id}.jpg")
            await file.download_to_drive(file_path)
            
            queue = context.application.bot_data["ingestion_queue"]
            await queue.put({
                "chat_id": update.effective_chat.id,
                "message_id": update.message.message_id,
                "input_type": "image",
                "raw_content": file_path,
                "processing_msg_id": processing_msg_id,
                "metadata": {
                    "source": "telegram",
                    "message_id": str(update.message.message_id),
                    "caption": text,
                    "silent": silent,
                    "chat_id": chat_id,
                    "original_text": f"[Photo Reply] {text}".strip()
                }
            })
            if silent:
                await run_conversational_query(update, context, query_text=text, original_msg_text=text)
            return

    # 2. Parse for URLs in the text itself
    urls = []
    if update.message.entities:
        for entity in update.message.entities:
            if entity.type == "url":
                url = text[entity.offset : entity.offset + entity.length]
                urls.append(url)
            elif entity.type == "text_link":
                urls.append(entity.url)
                
    if urls:
        # Strip URLs to obtain pure user text annotation
        user_annot = text
        for url in urls:
            user_annot = user_annot.replace(url, "")
        user_annot = user_annot.strip()
        
        silent = bool(user_annot)
        processing_msg_id = None
        if not silent:
            processing_msg = await update.message.reply_text(
                "⏳ Processing...",
                reply_to_message_id=update.message.message_id
            )
            processing_msg_id = processing_msg.message_id
            
        queue = context.application.bot_data["ingestion_queue"]
        for url in urls:
            await queue.put({
                "chat_id": update.effective_chat.id,
                "message_id": update.message.message_id,
                "input_type": "link",
                "raw_content": url,
                "processing_msg_id": processing_msg_id,
                "metadata": {
                    "source": "telegram",
                    "message_id": str(update.message.message_id),
                    "source_url": url,
                    "user_annotation": user_annot,
                    "silent": silent,
                    "chat_id": str(update.effective_chat.id),
                    "original_text": text
                }
            })
        if silent:
            await run_conversational_query(update, context, query_text=text, original_msg_text=text)
        return

    # 3. Process conversational RAG Query
    await run_conversational_query(update, context, query_text=text, original_msg_text=text)
