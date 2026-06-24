import asyncio
import logging
from src.ingestion.graph import ingestion_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestIngestion")

async def run_test():
    # Construct a test state input
    test_input = {
        "input_type": "text",
        "raw_content": (
            "Sylvi is a multimodal, stateful memory copilot designed to act as an extension of "
            "your digital mind. It leverages LangGraph for orchestration and Pinecone for vector database storage."
        ),
        "metadata": {
            "source": "telegram_test",
            "message_id": "12345"
        }
    }
    
    logger.info("Executing Ingestion Graph...")
    try:
        # Run the graph and collect final state
        # In LangGraph v1+, we invoke the graph with input values
        final_state = await ingestion_graph.ainvoke(test_input)
        
        logger.info("Graph execution completed successfully!")
        logger.info(f"Processed Text: {final_state.get('processed_text')}")
        logger.info(f"Chunks generated: {len(final_state.get('chunks', []))}")
        logger.info(f"Telemetry Latencies (ms): {final_state.get('latency_ms')}")
        
    except Exception as e:
        logger.error(f"Graph execution failed: {str(e)}")
        logger.warning(
            "Note: If you have placeholder keys in your '.env', "
            "this failure is expected when calling Google Gemini or Pinecone APIs."
        )

if __name__ == "__main__":
    asyncio.run(run_test())
