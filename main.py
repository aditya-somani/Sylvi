import os
import logging
import threading
from typing import Any
from http.server import BaseHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv
from src.bot import create_bot_app

# Load environment variables on startup
load_dotenv()

# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("Sylvi")

# --- Background Health Check Server ---

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        """Endpoint handler for HF/Uptime health requests."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format: str, *args: Any) -> None:
        # Silent overrides to prevent stdout clutter
        pass

def run_health_server(port: int) -> None:
    """Runs a simple HTTP health check server."""
    logger.info(f"Starting background health server on port {port}...")
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    try:
        server.serve_forever()
    except Exception as e:
        logger.error(f"Health server error: {str(e)}")


def main() -> None:
    """Entry point of the Sylvi Telegram Bot application."""
    # 1. Spawn the health-check server thread to satisfy hosting environments
    port = int(os.environ.get("PORT", 7860))
    health_thread = threading.Thread(target=run_health_server, args=(port,), daemon=True)
    health_thread.start()

    # 2. Boot the polling Telegram client
    logger.info("Initializing Sylvi personal memory copilot bot...")
    try:
        app = create_bot_app()
        logger.info("Sylvi bot application created. Starting polling...")
        app.run_polling()
    except Exception as e:
        logger.critical(f"Failed to start bot application: {str(e)}")

if __name__ == "__main__":
    main()
