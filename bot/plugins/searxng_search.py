import os
import logging
from typing import Dict, List, Optional
import httpx
from urllib.parse import urlparse

from .plugin import Plugin

logger = logging.getLogger(__name__)


class SearxngSearchPlugin(Plugin):
    """
    A plugin to search the web using self-hosted SearXNG instance.
    Supports authentication via X-API-KEY header and multiple search types.
    """

    def __init__(self):
        self.searxng_url = os.getenv('SEARXNG_URL', 'http://localhost:8888')
        self.api_key = os.getenv('SEARXNG_API_KEY', '')
        self.timeout = float(os.getenv('SEARXNG_TIMEOUT', '10'))
        self.max_results = int(os.getenv('SEARXNG_MAX_RESULTS', '10'))
        self.engines = os.getenv('SEARXNG_ENGINES', 'google,bing,duckduckgo')
        
        # Ensure URL doesn't have trailing slash for consistency
        self.searxng_url = self.searxng_url.rstrip('/')

    def get_source_name(self) -> str:
        return 'SearXNG'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'web_search',
                    'description': 'Execute a web search using SearXNG for the given query and return a list of results with images, news, and related searches',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'the user query to search for'
                            },
                            'num_results': {
                                'type': 'integer',
                                'description': 'number of results to return (default: 10, max: 50)',
                                'minimum': 1,
                                'maximum': 50,
                                'default': 10
                            },
                            'search_type': {
                                'type': 'string',
                                'description': 'type of search: general, images, videos, news (default: general)',
                                'enum': ['general', 'images', 'videos', 'news'],
                                'default': 'general'
                            },
                        },
                        'required': ['query'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    async def _search(self, query: str, num_results: int = None, search_type: str = 'general') -> Dict:
        """
        Execute search against SearXNG instance.
        
        Args:
            query: Search query string
            num_results: Number of results to return (default: self.max_results)
            search_type: Type of search - 'general', 'images', 'videos', 'news' (default: 'general')
            
        Returns:
            Dictionary with search results
        """
        if num_results is None:
            num_results = self.max_results
        else:
            num_results = min(num_results, 50)  # Cap at 50
        
        # Validate search type
        if search_type not in ['general', 'images', 'videos', 'news']:
            search_type = 'general'
        
        try:
            # Prepare request headers
            headers = {}
            
            if self.api_key:
                headers['X-API-Key'] = self.api_key
            
            # Map search type to SearXNG category
            category_map = {
                'general': 'general',
                'images': 'images',
                'videos': 'videos',
                'news': 'news',
            }
            category = category_map.get(search_type, 'general')
            
            # Prepare search parameters
            params = {
                'q': query,
                'format': 'json',
                'pageno': 1,
                'categories': category,
                'language': 'all',
                'engines': self.engines,
            }
            
            search_url = f'{self.searxng_url}/search'
            
            logger.debug(f'Searching SearXNG ({search_type}): {search_url} with query: {query}')
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(search_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
            
            return data
            
        except httpx.TimeoutException:
            logger.error(f'SearXNG search timeout for query: {query}')
            return {'error': 'Search request timed out. Please try again.'}
        except httpx.HTTPStatusError as e:
            logger.error(f'SearXNG HTTP error {e.status_code}: {e}')
            if e.status_code == 401:
                return {'error': 'Authentication failed. Check SEARXNG_API_KEY.'}
            elif e.status_code == 403:
                return {'error': 'Access forbidden. Check API permissions.'}
            else:
                return {'error': f'Search service error: {e.status_code}'}
        except httpx.RequestError as e:
            logger.error(f'SearXNG request error: {e}')
            return {'error': 'Failed to connect to search service. Check SEARXNG_URL.'}
        except Exception as e:
            logger.error(f'Unexpected error in SearXNG search: {e}')
            return {'error': f'Unexpected error: {str(e)}'}

    def _get_attribution(self, url: str) -> str:
        """Extract domain/attribution from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc or ''
        except Exception:
            return ''

    def _is_news_result(self, result: Dict) -> bool:
        """Identify if a result is a news article based on URL and title patterns."""
        url = (result.get('url') or '').lower()
        title = (result.get('title') or '').lower()
        
        news_keywords = [
            'breaking news', 'latest news', 'top stories', 'news today',
            'developing story', 'trending news', 'news', 'story'
        ]
        
        has_news_keywords = any(keyword in title for keyword in news_keywords)
        has_news_path = any(path in url for path in ['/news/', '/world/', '/politics/', '/breaking/'])
        
        return has_news_keywords or has_news_path

    def _format_results(self, data: Dict, search_type: str = 'general') -> Dict:
        """
        Format SearXNG JSON response into structured result format.
        
        Args:
            data: Raw SearXNG API response
            search_type: Type of search performed
            
        Returns:
            Dictionary with organized results (organic, images, news, related_searches)
        """
        formatted = {
            'organic': [],
            'images': [],
            'news': [],
            'related_searches': [],
        }
        
        if 'error' in data:
            return formatted
        
        results_list = data.get('results', [])
        
        # Process organic results
        for idx, result in enumerate(results_list):
            if not result.get('url'):
                continue
            
            attribution = self._get_attribution(result.get('url', ''))
            
            organic_result = {
                'position': idx + 1,
                'title': result.get('title', 'No title'),
                'link': result.get('url', ''),
                'snippet': result.get('content', '')[:300],  # Limit snippet to 300 chars
                'attribution': attribution,
                'date': result.get('publishedDate', ''),
            }
            
            formatted['organic'].append(organic_result)
            
            # Identify and separate news results
            if self._is_news_result(result):
                news_result = {
                    'position': len(formatted['news']) + 1,
                    'title': result.get('title', ''),
                    'link': result.get('url', ''),
                    'snippet': result.get('content', '')[:300],
                    'source': attribution,
                    'date': result.get('publishedDate', ''),
                }
                formatted['news'].append(news_result)
            
            # Extract image results
            if result.get('img_src'):
                image_result = {
                    'position': len(formatted['images']) + 1,
                    'title': result.get('title', ''),
                    'imageUrl': result.get('img_src', ''),
                    'link': result.get('url', ''),
                    'source': attribution,
                    'domain': attribution,
                }
                formatted['images'].append(image_result)
        
        # Process related/suggestion searches
        if data.get('suggestions'):
            formatted['related_searches'] = [
                {'query': suggestion}
                for suggestion in data.get('suggestions', [])[:5]
            ]
        
        # Cap results
        formatted['images'] = formatted['images'][:6]
        formatted['news'] = formatted['news'][:5]
        
        return formatted

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        """
        Execute web search and return results.
        
        Args:
            function_name: Name of function to execute (should be 'web_search')
            helper: OpenAI helper instance
            **kwargs: Function arguments including 'query', optional 'num_results' and 'search_type'
            
        Returns:
            Dictionary with search results or error message
        """
        query = kwargs.get('query', '').strip()
        num_results = kwargs.get('num_results', self.max_results)
        search_type = kwargs.get('search_type', 'general')
        
        if not query:
            return {'result': 'Empty query provided'}
        
        if len(query) > 500:
            return {'result': 'Query too long (max 500 characters)'}
        
        # Execute search
        data = await self._search(query, num_results, search_type)
        
        if 'error' in data:
            return {'result': data['error']}
        
        # Format results
        formatted_results = self._format_results(data, search_type)
        
        # Return based on search type
        if search_type == 'images':
            return {'result': formatted_results['images']}
        elif search_type == 'news':
            return {'result': formatted_results['news']}
        else:
            # For general search, return all types
            return {
                'result': {
                    'organic': formatted_results['organic'],
                    'images': formatted_results['images'],
                    'news': formatted_results['news'],
                    'related_searches': formatted_results['related_searches'],
                }
            }
