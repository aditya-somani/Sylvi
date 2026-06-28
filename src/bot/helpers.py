import logging
import asyncio
from datetime import datetime
from typing import List

from telegram import Update
from telegram.ext import ContextTypes

from src.memory.profile import ProfileMemoryDB

logger = logging.getLogger(__name__)


async def extract_and_save_facts_background(chat_id: str, query: str, answer: str, context_str: str) -> None:
    """Extracts and saves new profile facts in the background using LLM."""
    try:
        from src.services.llm import LLMService
        from pydantic import BaseModel
        
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
        
        # Add user query and assistant response to PostgreSQL chat history
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
