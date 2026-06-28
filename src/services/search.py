import logging
import httpx
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class WebSearchService:
    """
    Service that scrapes DuckDuckGo HTML search page keylessly.
    Exposes a synchronous search method.
    """
    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout = timeout_seconds
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    def search(self, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        """
        Synchronously scrapes DuckDuckGo for the given query and extracts result snippets.
        
        Returns:
            A list of dicts with keys: 'title', 'link', 'snippet'.
        """
        clean_query = query.strip()
        if not clean_query:
            return []
            
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(clean_query)}"
        logger.info(f"Running keyless DDG Web Search: {clean_query}")
        
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(url, headers=self.headers)
                if response.status_code != 200:
                    logger.warning(f"DDG Search failed with HTTP {response.status_code}")
                    return []
                    
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            # Extract snippets and search result structures
            for snippet_node in soup.find_all("a", class_="result__snippet")[:max_results]:
                parent = snippet_node.find_parent("div", class_="result__body")
                if not parent:
                    continue
                
                title_node = parent.find("a", class_="result__url")
                title = title_node.text.strip() if title_node else "No Title"
                
                # Retrieve direct URL from raw DDG redirection URL if possible
                link = ""
                if title_node and "href" in title_node.attrs:
                    raw_href = title_node["href"]
                    # Extract target URL from redirect parameter if present
                    if "/l/?uddg=" in raw_href:
                        parsed_url = urllib.parse.urlparse(raw_href)
                        query_params = urllib.parse.parse_qs(parsed_url.query)
                        if "uddg" in query_params:
                            link = query_params["uddg"][0]
                    if not link:
                        # Fallback to appending host if needed
                        if raw_href.startswith("//"):
                            link = f"https:{raw_href}"
                        elif raw_href.startswith("/"):
                            link = f"https://duckduckgo.com{raw_href}"
                        else:
                            link = raw_href
                            
                snippet = snippet_node.text.strip()
                
                results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet
                })
                
            logger.info(f"DDG Web Search returned {len(results)} matches.")
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute DDG search for '{clean_query}': {str(e)}")
            return []
