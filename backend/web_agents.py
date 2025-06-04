"""
Web scraping and search agents module for gathering data from various sources.
"""

import os
import logging
import time
import random
import asyncio
import requests
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import googleapiclient.discovery
from tavily import TavilyClient
from serpapi.google_search import GoogleSearch
from exa_py import Exa
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """Handles web scraping with error handling and anti-bot detection measures."""
    
    def __init__(self):
        """Initialize the web scraper with necessary configurations."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })
        # Initialize base Chrome options
        self.base_chrome_options = [
            '--headless',
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-blink-features=AutomationControlled',
            '--disable-extensions',
            '--disable-software-rasterizer',
            '--disable-notifications',
            '--disable-popup-blocking',
            '--disable-infobars',
            '--disable-features=IsolateOrigins,site-per-process',
            '--disable-site-isolation-trials',
            '--disable-web-security',
            '--allow-running-insecure-content',
            '--window-size=1920,1080',
            '--start-maximized',
            '--ignore-certificate-errors',
            '--ignore-ssl-errors',
            '--ignore-certificate-errors-spki-list',
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
    def _get_random_delay(self) -> float:
        """Generate a random delay between requests."""
        return random.uniform(3, 7)
    
    def scrape_url(self, url: str, use_selenium: bool = True) -> Optional[str]:
        """
        Scrape content from a URL with error handling and anti-bot measures.
        
        Args:
            url: The URL to scrape
            use_selenium: Whether to use Selenium (default: True)
            
        Returns:
            The scraped content or None if failed
        """
        try:
            time.sleep(self._get_random_delay())
            return self._scrape_with_selenium(url)
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def _scrape_with_selenium(self, url: str) -> Optional[str]:
        """Scrape content using Selenium for JavaScript-heavy pages."""
        driver = None
        try:
            # Create new Chrome options
            options = Options()
            
            # Add base options
            for option in self.base_chrome_options:
                options.add_argument(option)
            
            # Add random user agent
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0'
            ]
            selected_ua = random.choice(user_agents)
            options.add_argument(f'user-agent={selected_ua}')
            
            # Add experimental options
            options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
            options.add_experimental_option('useAutomationExtension', False)
            
            # Add preferences
            prefs = {
                'profile.default_content_setting_values': {
                    'notifications': 2,
                    'images': 2,  # Disable images for faster loading
                    'javascript': 1,
                    'cookies': 1
                },
                'profile.managed_default_content_settings': {
                    'javascript': 1
                },
                'profile.cookie_controls_mode': 0,
                'credentials_enable_service': False,
                'profile.password_manager_enabled': False
            }
            options.add_experimental_option('prefs', prefs)
            
            # Initialize the driver with the configured options
            driver = webdriver.Chrome(options=options)
            
            # Set window size and position
            driver.set_window_size(1920, 1080)
            driver.set_window_position(0, 0)
            
            # Add CDP commands to modify browser fingerprint
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                'userAgent': selected_ua,
                'platform': 'Windows',
                'acceptLanguage': 'en-US,en;q=0.9'
            })
            
            # Modify navigator properties
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5]
                    });
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en']
                    });
                    window.chrome = {
                        runtime: {}
                    };
                '''
            })
            
            # Load the page with retry mechanism
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Add headers to the request
                    driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {
                        'headers': {
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.5',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'DNT': '1',
                            'Connection': 'keep-alive',
                            'Upgrade-Insecure-Requests': '1',
                            'Sec-Fetch-Dest': 'document',
                            'Sec-Fetch-Mode': 'navigate',
                            'Sec-Fetch-Site': 'none',
                            'Sec-Fetch-User': '?1',
                            'Cache-Control': 'max-age=0',
                            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
                            'Sec-Ch-Ua-Mobile': '?0',
                            'Sec-Ch-Ua-Platform': '"Windows"'
                        }
                    })
                    
                    # Add random delay before loading
                    time.sleep(random.uniform(1, 3))
                    
                    driver.get(url)
                    
                    # Wait for page to load with shorter timeout
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                    
                    # Simulate human-like behavior
                    self._simulate_human_behavior(driver)
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                    time.sleep(random.uniform(2, 4))  # Longer delay between retries
            
            # Get the page source
            content = driver.page_source
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
                element.decompose()
            
            # Get main content - optimized selectors
            main_content = None
            content_selectors = [
                'main', 'article', 'div[role="main"]',
                'div.content', 'div.main', 'div.article',
                'div[class*="content"]', 'div[class*="article"]'
            ]
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                text = ' '.join(main_content.stripped_strings)
            else:
                text = ' '.join(soup.body.stripped_strings)
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Selenium scraping failed for {url}: {str(e)}")
            return None
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def _simulate_human_behavior(self, driver):
        """Simulate human-like behavior to avoid bot detection."""
        try:
            # Get page height
            total_height = int(driver.execute_script("return document.body.scrollHeight"))
            
            # Random initial scroll
            initial_scroll = random.randint(100, 300)
            driver.execute_script(f"window.scrollTo(0, {initial_scroll});")
            time.sleep(random.uniform(0.5, 1.5))
            
            # Scroll down in chunks with random pauses
            current_position = initial_scroll
            while current_position < total_height:
                # Random scroll amount
                scroll_amount = random.randint(100, 300)
                current_position += scroll_amount
                
                # Smooth scroll
                driver.execute_script(f"window.scrollTo({{top: {current_position}, behavior: 'smooth'}});")
                
                # Random pause
                time.sleep(random.uniform(0.3, 1.0))
                
                # Occasionally scroll back up a bit
                if random.random() < 0.1:
                    current_position -= random.randint(50, 150)
                    driver.execute_script(f"window.scrollTo({{top: {current_position}, behavior: 'smooth'}});")
                    time.sleep(random.uniform(0.2, 0.5))
            
            # Scroll back to top
            driver.execute_script("window.scrollTo({top: 0, behavior: 'smooth'});")
            time.sleep(random.uniform(0.5, 1.0))
            
        except Exception as e:
            logger.warning(f"Error in human behavior simulation: {str(e)}")

    def _is_bot_detected(self, response: requests.Response) -> bool:
        """Check if the website has detected us as a bot."""
        bot_indicators = [
            'captcha',
            'robot',
            'bot',
            'automated',
            'security check',
            'unusual traffic',
            'verify you are human',
            'please wait while we verify',
            'access denied',
            'blocked'
        ]
        return any(indicator in response.text.lower() for indicator in bot_indicators)

class SearchAgent:
    """Handles various search APIs with parallel execution."""
    
    def __init__(self):
        """Initialize search agent with Tavily and arXiv APIs."""
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        
        # Verify we have the necessary keys
        if not self.tavily_api_key:
            logger.warning("Tavily API key not found. Tavily search will not be available.")
        
        # Initialize session for HTTP requests
        self.session = None
        
    async def _search_tavily(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        if not self.tavily_api_key:
            return []
            
        try:
            # Use direct API call for better control
            url = "https://api.tavily.com/search"
            headers = {
                "content-type": "application/json",
                "Authorization": f"Bearer {self.tavily_api_key}"
            }
            data = {
                "query": query,
                "search_depth": "advanced",
                "max_results": num_results,
                "include_domains": ["arxiv.org", "scholar.google.com", "ieee.org", "acm.org"],
                "include_answer": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._process_tavily_results(result.get('results', []))
                    else:
                        logger.warning(f"Tavily API returned status code {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            return []
    
    async def _search_arxiv(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using arXiv API."""
        try:
            # Format query for API
            formatted_query = query.replace(' ', '+')
            
            # Construct the arXiv API URL
            base_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{formatted_query}',
                'start': 0,
                'max_results': num_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            # Make the request
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            # Parse the XML response
            from xml.etree import ElementTree
            root = ElementTree.fromstring(response.content)
            
            # Extract results
            processed_results = []
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                try:
                    title = entry.find('.//{http://www.w3.org/2005/Atom}title').text
                    abstract = entry.find('.//{http://www.w3.org/2005/Atom}summary').text
                    url = entry.find('.//{http://www.w3.org/2005/Atom}id').text
                    
                    # Clean up the text
                    title = title.strip()
                    abstract = abstract.strip()
                    
                    processed_results.append({
                        'title': title,
                        'url': url,
                        'snippet': abstract[:200] + '...' if len(abstract) > 200 else abstract,
                        'source': 'arxiv'
                    })
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Error processing arXiv entry: {str(e)}")
                    continue
            
            return processed_results
            
        except Exception as e:
            logger.error(f"arXiv search failed: {str(e)}")
            return []
    
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform parallel search using Tavily and arXiv APIs.
        
        Args:
            query: The search query
            num_results: Number of results to return per API
            
        Returns:
            Combined list of unique search results
        """
        # Run searches in parallel
        tasks = [
            self._search_tavily(query, num_results),
            self._search_arxiv(query, num_results)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and deduplicate results
        combined_results = []
        seen_urls = set()
        
        for api_results in results:
            if isinstance(api_results, Exception):
                logger.error(f"Search task failed: {str(api_results)}")
                continue
                
            for result in api_results:
                url = result['url']
                if url not in seen_urls:
                    seen_urls.add(url)
                    combined_results.append(result)
        
        return combined_results
    
    def _process_tavily_results(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Process Tavily search results."""
        processed_results = []
        for result in results:
            processed_results.append({
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'snippet': result.get('description', ''),
                'source': 'tavily'
            })
        return processed_results

class WebDataGatherer:
    """Combines web scraping and search capabilities for comprehensive data gathering."""
    
    def __init__(self):
        """Initialize the web data gatherer with scraper and search agent."""
        self.scraper = WebScraper()
        self.search_agent = SearchAgent()
    
    async def gather_data(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Gather data from web sources using parallel search and scraping.
        
        Args:
            query: The search query
            max_results: Maximum number of results to gather per API
            
        Returns:
            List of gathered data with content
        """
        # Get search results from all APIs in parallel
        search_results = await self.search_agent.search(query, num_results=max_results)
        
        # Scrape content from all URLs in parallel
        async def scrape_url(result):
            content = self.scraper.scrape_url(result['url'])
            if content:
                return {
                    'title': result['title'],
                    'url': result['url'],
                    'snippet': result['snippet'],
                    'content': content,
                    'source': result['source']
                }
            return None
        
        # Create scraping tasks
        scraping_tasks = [scrape_url(result) for result in search_results]
        gathered_data = await asyncio.gather(*scraping_tasks)
        
        # Filter out None results
        return [data for data in gathered_data if data is not None] 