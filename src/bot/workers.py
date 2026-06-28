import os
import logging
import asyncio
from datetime import datetime

from telegram.constants import ChatAction

from src.memory.profile import ProfileMemoryDB
from src.prompts import INGESTION_CONFIRM_SYSTEM_PROMPT
from src.bot.helpers import extract_and_save_facts_background

logger = logging.getLogger(__name__)


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
                            text=f"❌ **Ingestion Failed**\n\nSorry, I encountered an error:\n`{str(e)}`",
                            parse_mode="Markdown"
                        )
                    except Exception:
                        pass
                else:
                    await application.bot.send_message(
                        chat_id=chat_id,
                        text=f"❌ **Ingestion Failed**\n\nSorry, I encountered an error:\n`{str(e)}`",
                        reply_to_message_id=message_id,
                        parse_mode="Markdown"
                    )
        finally:
            queue.task_done()
