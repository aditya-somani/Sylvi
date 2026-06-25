import asyncio
import logging
from datetime import datetime
from src.query.graph import query_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestQuery")

async def run_scenario(query: str, description: str):
    logger.info(f"\n========================================\nSCENARIO: {description}\nQuery: '{query}'")
    
    test_input = {
        "query": query,
        "chat_id": "test_user_999",
        "current_time": datetime.utcnow().isoformat()
    }
    
    try:
        final_state = await query_graph.ainvoke(test_input)
        logger.info(f"Routed Intent: {final_state.get('intent')}")
        logger.info(f"Response: {final_state.get('answer')}")
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")

async def run_all_tests():
    # Scenario 1: Chit-chat (should route directly, bypassing vector/db, completely offline-safe!)
    await run_scenario(
        query="Hello! How are you doing today?",
        description="Casual Chit-Chat Routing (Expected: Fast greeting, no database lookup)"
    )
    
    # Scenario 2: Reminder scheduling (should parse time relative to current_time, save to DB, bypass Pinecone!)
    await run_scenario(
        query="Remind me in 5 minutes to submit the report.",
        description="Relative Reminder Scheduling (Expected: Save to SQLite, no Pinecone lookup)"
    )

    # Scenario 3: Profile Query (should load profile facts, bypass Pinecone!)
    await run_scenario(
        query="What do you know about my editor preferences?",
        description="Profile Memory Query (Expected: SQLite lookup, no Pinecone lookup)"
    )

    # Scenario 4: RAG Retrieval Search (should trigger both Pinecone and SQLite)
    await run_scenario(
        query="Search my notes for anything about LangGraph optimization.",
        description="Full Vector RAG Search (Expected: Pinecone lookup, will trigger auth check)"
    )

if __name__ == "__main__":
    asyncio.run(run_all_tests())
