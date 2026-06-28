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
from src.prompts import INGESTION_CONFIRM_SYSTEM_PROMPT

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
        metadata = task["metadata"] or {}
        processing_msg_id = task.get("processing_msg_id")
        
        silent = metadata.get("silent", False)
        
        try:
            # Let user know we are actively working on it (only if not silent)
            if not silent:
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
            
            # 1. Determine user text for history logging or voice query execution
            user_text = ""
            if input_type == "link":
                user_text = metadata.get("user_annotation") or ""
            elif input_type == "image":
                user_text = metadata.get("caption") or ""
            elif input_type == "voice":
                # For voice, final_state["processed_text"] has the transcription
                proc_text = final_state.get("processed_text", "")
                if "\n" in proc_text:
                    user_text = proc_text.split("\n", 1)[1].strip()
                else:
                    user_text = proc_text
                    
            db = ProfileMemoryDB()
            
            # 2. If it's a voice note, we run conversational query node if there's any text!
            # Since a voice note is processed asynchronously in the background worker,
            # we run its query node here and edit the processing_msg placeholder.
            if input_type == "voice" and user_text.strip():
                # Run the query graph asynchronously to find the intent
                from src.query.graph import query_graph
                from datetime import timezone, timedelta
                ist = timezone(timedelta(hours=5, minutes=30))
                current_time_ist = datetime.now(ist).isoformat()
                
                chat_history = db.get_chat_history(str(chat_id), limit=15)
                
                query_state = await query_graph.ainvoke({
                    "query": user_text,
                    "chat_id": str(chat_id),
                    "current_time": current_time_ist,
                    "chat_history": chat_history,
                    "status_callback": None
                })
                
                success_text = query_state.get("answer") or "Sorry, I had trouble formulating a response."
                
                # Add to chat history
                db.add_chat_message(str(chat_id), "user", f"[Voice Note] {user_text}")
                db.add_chat_message(str(chat_id), "assistant", success_text)
                
                # Trigger background fact extraction silently
                context_list = []
                pinecone_ctx = query_state.get("pinecone_context") or []
                for match in pinecone_ctx:
                    meta = match.get("metadata") or {}
                    text = meta.get("text", "")
                    if text:
                        context_list.append(text)
                context_str = "\n".join(context_list)
                
                asyncio.create_task(
                    extract_and_save_facts_background(
                        chat_id=str(chat_id),
                        query=f"[Voice Note] {user_text}",
                        answer=success_text,
                        context_str=context_str
                    )
                )
                
            else:
                # Generate friendly personalized confirmation response using LLM (original behavior)
                success_text = "✅ Ingested successfully. I've indexed this content in your memory."
                try:
                    processed_text = final_state.get("processed_text", "")
                    if processed_text:
                        from src.services.llm import LLMService
                        llm_service = LLMService()
                        confirm_prompt = f"Input Type: {input_type}\nIngested Content:\n{processed_text}"
                        response = llm_service.generate_groq(
                            prompt=confirm_prompt,
                            system_instruction=INGESTION_CONFIRM_SYSTEM_PROMPT,
                            temperature=0.5
                        )
                        cleaned_resp = str(response).strip()
                        if cleaned_resp:
                            if (cleaned_resp.startswith('"') and cleaned_resp.endswith('"')) or (cleaned_resp.startswith("'") and cleaned_resp.endswith("'")):
                                cleaned_resp = cleaned_resp[1:-1].strip()
                            success_text = cleaned_resp
                except Exception as confirm_err:
                    logger.warning(f"Failed to generate dynamic confirmation response: {str(confirm_err)}")
                
                # If this background ingestion was not silent (i.e. pure ingestion without foreground conversation),
                # save it to chat history so future messages have context about this ingestion!
                if not silent:
                    user_msg_for_history = metadata.get("original_text") or raw_content
                    db.add_chat_message(str(chat_id), "user", user_msg_for_history)
                    db.add_chat_message(str(chat_id), "assistant", success_text)
            
            # Send message/edit placeholder only if NOT silent
            if not silent and processing_msg_id:
                try:
                    await application.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=processing_msg_id,
                        text=success_text,
                        parse_mode="Markdown"
                    )
                except Exception as edit_err:
                    logger.warning(f"Failed to edit success message: {str(edit_err)}")
                    await application.bot.send_message(
                        chat_id=chat_id,
                        text=success_text,
                        reply_to_message_id=message_id,
                        parse_mode="Markdown"
                    )
            elif not silent:
                await application.bot.send_message(
                    chat_id=chat_id,
                    text=success_text,
                    reply_to_message_id=message_id,
                    parse_mode="Markdown"
                )
                
        except Exception as e:
            logger.exception(f"Error processing background ingestion task for chat {chat_id}")
            if not silent:
                # Notify failure only if not silent
                if processing_msg_id:
                    try:
                        await application.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=processing_msg_id,
                            text="❌ Ingestion Failed.",
                            parse_mode="Markdown"
                        )
                    except Exception as edit_err:
                        logger.warning(f"Failed to edit failure message status: {str(edit_err)}")
                try:
                    await application.bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"❌ **Ingestion Failed**\n\n"
                            f"I had trouble digesting that memory.\n"
                            f"**Error**: {str(e)}"
                        ),
                        reply_to_message_id=message_id,
                        parse_mode="Markdown"
                    )
                except Exception as notify_err:
                    logger.error(f"Failed to send failure notification to user: {str(notify_err)}")
        finally:
            queue.task_done()


async def reminder_worker(application) -> None:
    """
    Background worker that polls the database for due reminders
    and dispatches them to their corresponding chats.
    """
    logger.info("Background Reminder Worker started.")
    # Instantiate DB here to ensure worker runs successfully on a separate connection
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
                    # Mark as sent anyway to avoid infinite loop retrying invalid/blocked chats
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
    chat_id = str(update.effective_chat.id)
    db = ProfileMemoryDB()
    facts = db.get_all_facts(chat_id)
    
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

async def extract_and_save_facts_background(chat_id: str, query: str, answer: str, context_str: str) -> None:
    """Extracts and saves new profile facts in the background using LLM."""
    try:
        from src.services.llm import LLMService
        from pydantic import BaseModel
        from typing import List
        
        class FactsExtractor(BaseModel):
            facts: List[str]
            
        db = ProfileMemoryDB()
        existing_facts = db.get_all_facts(chat_id)
        existing_str = "\n".join(f"- {f['fact']}" for f in existing_facts) if existing_facts else "None"
        
        llm_service = LLMService()
        
        prompt = (
            f"Existing Facts:\n{existing_str}\n\n"
            f"Context Documents:\n{context_str}\n\n"
            f"User Message: {query}\n"
            f"Assistant Response: {answer}"
        )
        
        system_instruction = (
            "You are a profile memory manager. Your task is to analyze the user message, "
            "assistant response, and context documents to extract any personal facts, preferences, "
            "location, names, or favorite things that the user has shared about themselves.\n\n"
            "Formulate each fact as a simple, standalone declarative sentence (e.g. 'User's favorite anime is One Piece', 'User is based in Jawad, MP, India').\n"
            "IMPORTANT: Do not extract facts that are already present in the 'Existing Facts' list.\n"
            "Ensure the facts are accurate, start with 'User' or 'User's', and are directly supported by the text. Do not hallucinate or assume facts.\n"
            "If no new facts are shared, return an empty list."
        )
        
        extracted = llm_service.generate_structured_groq(
            prompt=prompt,
            schema=FactsExtractor,
            system_instruction=system_instruction,
            temperature=0.0
        )
        
        if extracted.facts:
            for fact in extracted.facts:
                db.add_fact(chat_id, fact)
                logger.info(f"Extracted and saved new fact: '{fact}' for chat {chat_id}")
    except Exception as e:
        logger.error(f"Failed in background fact extraction: {e}")


async def run_conversational_query(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query_text: str,
    original_msg_text: str = None
) -> None:
    """Helper to run the conversational RAG query graph and manage chat history."""
    chat_id = str(update.effective_chat.id)
    db = ProfileMemoryDB()
    reply_to = update.message.reply_to_message
    
    # Load recent chat history context from database
    chat_history = db.get_chat_history(chat_id, limit=15)
    
    # Formulate query context if replying to a bot response
    if reply_to:
        ref_text = reply_to.text or reply_to.caption or "media file"
        if reply_to.from_user and reply_to.from_user.is_bot:
            query_text = f"[Context: Replying to Sylvi's message: \"{ref_text}\"]\n\nUser Query: {query_text}"
        else:
            query_text = f"[Context: Replying to message: \"{ref_text}\"]\n\nUser Query: {query_text}"
            
    try:
        from datetime import timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        current_time_ist = datetime.now(ist).isoformat()
        
        # Send a placeholder message to give immediate visual typing/thinking feedback
        processing_msg = await update.message.reply_text(
            "⏳ Thinking...",
            reply_to_message_id=update.message.message_id,
            parse_mode="Markdown"
        )
        
        async def update_status(status_text: str):
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=processing_msg.message_id,
                    text=status_text,
                    parse_mode="Markdown"
                )
            except Exception as edit_err:
                logger.debug(f"Status update failed: {edit_err}")

        # Lazy import graph to avoid circular dependency
        from src.query.graph import query_graph

        final_state = await query_graph.ainvoke({
            "query": query_text,
            "chat_id": chat_id,
            "current_time": current_time_ist,
            "chat_history": chat_history,
            "status_callback": update_status
        })
        
        answer = final_state.get("answer") or "Sorry, I couldn't formulate a response."
        
        # Add user query and assistant response to SQLite chat history
        log_text = original_msg_text if original_msg_text else query_text
        db.add_chat_message(chat_id, "user", log_text)
        db.add_chat_message(chat_id, "assistant", answer)
        
        # Trigger background fact extraction silently
        context_list = []
        pinecone_ctx = final_state.get("pinecone_context") or []
        for match in pinecone_ctx:
            meta = match.get("metadata") or {}
            text = meta.get("text", "")
            if text:
                context_list.append(text)
        context_str = "\n".join(context_list)
        
        asyncio.create_task(
            extract_and_save_facts_background(
                chat_id=chat_id,
                query=log_text,
                answer=answer,
                context_str=context_str
            )
        )
        
        # Update the placeholder with the final answer
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_msg.message_id,
                text=answer,
                parse_mode="Markdown"
            )
        except Exception as md_err:
            logger.warning(f"Markdown edit failed, falling back to raw text: {md_err}")
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_msg.message_id,
                text=answer
            )
            
    except Exception as e:
        logger.exception("Error during Query Graph execution")
        if 'processing_msg' in locals():
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=processing_msg.message_id,
                    text=f"❌ **Query Failed**\n\nSorry, I encountered an error while processing your request:\n`{str(e)}`",
                    parse_mode="Markdown"
                )
                return
            except Exception:
                pass
                
        await update.message.reply_text(
            f"❌ **Query Failed**\n\nSorry, I encountered an error while processing your request:\n`{str(e)}`",
            reply_to_message_id=update.message.message_id,
            parse_mode="Markdown"
        )


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
