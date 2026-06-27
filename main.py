import logging
from dotenv import load_dotenv
from src.bot import create_bot_app

# Load environment variables from .env on startup
load_dotenv()

# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("Sylvi")

def main() -> None:
    """Entry point of the Sylvi Telegram Bot application."""
    logger.info("Initializing Sylvi personal memory copilot bot...")
    try:
        app = create_bot_app()
        logger.info("Sylvi bot application created. Starting polling...")
        app.run_polling()
    except Exception as e:
        logger.critical(f"Failed to start bot application: {str(e)}")

if __name__ == "__main__":
    main()
