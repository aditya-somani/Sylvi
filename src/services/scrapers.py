import logging
import httpx
from bs4 import BeautifulSoup
from readability import Document

logger = logging.getLogger(__name__)

class WebScraperService:
    """
    Scrapes and cleans webpages to retrieve core content.
    Uses readability-lxml to isolate the article content and strip headers/footers/ads.
    """
    def __init__(self, timeout_seconds: int = 10):
        self.timeout = timeout_seconds

    async def scrape_url(self, url: str) -> str:
        """
        Asynchronously fetches a URL and parses its core readable content.
        
        Returns:
            A clean string containing the core article text.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        
        try:
            logger.info(f"Fetching URL: {url}")
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                html_content = response.text
                
            # Use readability-lxml to extract the main content segment
            # Document parses the DOM, identifies the "readable" article element,
            # and discards navigation, headers, footers, and sidebars.
            doc = Document(html_content)
            readable_html = doc.summary()  # Returns clean HTML block of the article body
            title = doc.title()
            
            # Use BeautifulSoup to convert the cleaned HTML chunk to raw plain text
            soup = BeautifulSoup(readable_html, "lxml")
            clean_text = soup.get_text(separator="\n")
            
            # Remove excessive whitespace/newlines
            lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
            final_content = f"Title: {title}\n\n" + "\n".join(lines)
            
            return final_content
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred while scraping {url}: {str(e)}")
            raise ValueError(f"Failed to access URL: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to scrape URL {url}: {str(e)}")
            raise ValueError(f"Error reading website content: {str(e)}")
