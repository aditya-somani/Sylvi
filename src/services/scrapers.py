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
            content_text_only = "\n".join(lines)
            
            # Check for SPA/JavaScript-required boilerplates or empty content
            lowercase_content = content_text_only.lower()
            js_indicators = [
                "javascript must be enabled",
                "enable javascript",
                "enable js",
                "you need to enable javascript",
                "requires javascript",
                "please enable javascript"
            ]
            is_boilerplate = any(indicator in lowercase_content for indicator in js_indicators)
            
            if len(content_text_only.strip()) < 150 or is_boilerplate:
                return f"[Webpage content could not be scraped due to system limitations (requires JavaScript/SPA)]\nURL: {url}"
                
            final_content = f"Title: {title}\n\n" + content_text_only
            return final_content
            
        except Exception as e:
            logger.warning(f"Failed to scrape URL {url}: {str(e)}")
            return f"[Webpage content could not be scraped due to system limitations (failed to fetch URL: {str(e)})]\nURL: {url}"
