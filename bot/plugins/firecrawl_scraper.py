import asyncio
import logging
import os
from typing import Dict, List, Optional

import httpx

from .plugin import Plugin

logger = logging.getLogger(__name__)


class FirecrawlScraperPlugin(Plugin):
    """
    A plugin to scrape web pages using self-hosted Firecrawl API.
    Supports batch scraping with parallel requests and flexible configuration.
    """

    def __init__(self):
        self.api_url = os.getenv('FIRECRAWL_API_URL', 'http://localhost:7000')
        self.api_key = os.getenv('FIRECRAWL_API_KEY', '')
        self.timeout = float(os.getenv('FIRECRAWL_TIMEOUT', '10'))
        self.max_concurrent = int(os.getenv('FIRECRAWL_MAX_CONCURRENT', '5'))
        self.extract_markdown = os.getenv('FIRECRAWL_EXTRACT_MARKDOWN', 'true').lower() == 'true'

        # Ensure URL doesn't have trailing slash and has /v1/scrape endpoint
        self.api_url = self.api_url.rstrip('/')
        if not self.api_url.endswith('/v1/scrape'):
            self.api_url = f'{self.api_url}/v1/scrape'

    def get_source_name(self) -> str:
        return 'Firecrawl'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'scrape_content',
                    'description': 'Scrape content from multiple URLs in parallel using Firecrawl. Use after web_search to fetch full page content, or independently to get page text/markdown. Returns markdown, HTML, and metadata for each URL.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'urls': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'List of URLs to scrape (max 20)',
                                'maxItems': 20,
                            },
                            'extract_markdown': {
                                'type': 'boolean',
                                'description': 'Extract markdown format (default: true)',
                                'default': True,
                            },
                            'only_main_content': {
                                'type': 'boolean',
                                'description': 'Extract only main content, ignore sidebars etc (default: true)',
                                'default': True,
                            },
                            'include_tags': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'HTML tags to include (optional)',
                            },
                            'exclude_tags': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'HTML tags to exclude (optional)',
                            },
                        },
                        'required': ['urls', 'extract_markdown', 'only_main_content', 'include_tags', 'exclude_tags'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    def _omit_undefined(self, obj: Dict) -> Dict:
        """Remove undefined/None values from dictionary."""
        return {k: v for k, v in obj.items() if v is not None}

    def _extract_metadata(self, response_data: Optional[Dict]) -> Dict:
        """Extract metadata from Firecrawl response."""
        if not response_data or 'metadata' not in response_data:
            return {}

        metadata = response_data.get('metadata', {})
        return {
            'title': metadata.get('title', ''),
            'description': metadata.get('description', ''),
            'language': metadata.get('language', ''),
            'source_url': metadata.get('sourceURL', ''),
            'status_code': metadata.get('statusCode'),
        }

    async def _scrape_single_url(
        self,
        client: httpx.AsyncClient,
        url: str,
        extract_markdown: bool = True,
        only_main_content: bool = True,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
    ) -> Dict:
        """
        Scrape a single URL using Firecrawl API.

        Args:
            client: httpx AsyncClient instance
            url: URL to scrape
            extract_markdown: Whether to extract markdown format
            only_main_content: Extract only main content
            include_tags: HTML tags to include
            exclude_tags: HTML tags to exclude

        Returns:
            Dictionary with scrape result
        """
        try:
            # Prepare payload
            formats = ['markdown', 'html'] if extract_markdown else ['html']

            payload = self._omit_undefined(
                {
                    'url': url,
                    'formats': formats,
                    'onlyMainContent': only_main_content,
                    'includeTags': include_tags,
                    'excludeTags': exclude_tags,
                    'timeout': int(self.timeout * 1000),  # Convert to milliseconds
                }
            )

            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
            }

            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'

            logger.debug(f'Scraping URL: {url}')

            response = await client.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            response.raise_for_status()
            data = response.json()

            if not data.get('success'):
                return {
                    'url': url,
                    'success': False,
                    'error': data.get('error', 'Unknown error from Firecrawl'),
                    'markdown': None,
                    'html': None,
                    'metadata': {},
                }

            response_data = data.get('data', {})

            return {
                'url': url,
                'success': True,
                'markdown': response_data.get('markdown') if extract_markdown else None,
                'html': response_data.get('html'),
                'metadata': self._extract_metadata(response_data),
                'error': None,
            }

        except httpx.TimeoutException:
            logger.error(f'Timeout scraping {url}')
            return {
                'url': url,
                'success': False,
                'error': f'Request timeout after {self.timeout}s',
                'markdown': None,
                'html': None,
                'metadata': {},
            }
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            logger.error(f'HTTP error {status_code} scraping {url}: {e}')
            error_msg = f'HTTP {status_code}'
            if status_code == 401:
                error_msg = 'Authentication failed. Check FIRECRAWL_API_KEY.'
            elif status_code == 403:
                error_msg = 'Access forbidden. Check API permissions.'
            elif status_code == 404:
                error_msg = 'URL not found or Firecrawl API not available.'

            return {
                'url': url,
                'success': False,
                'error': error_msg,
                'markdown': None,
                'html': None,
                'metadata': {},
            }
        except httpx.RequestError as e:
            logger.error(f'Request error scraping {url}: {e}')
            return {
                'url': url,
                'success': False,
                'error': f'Connection failed: {str(e)}',
                'markdown': None,
                'html': None,
                'metadata': {},
            }
        except Exception as e:
            logger.error(f'Unexpected error scraping {url}: {e}')
            return {
                'url': url,
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'markdown': None,
                'html': None,
                'metadata': {},
            }

    async def _scrape_batch(
        self,
        urls: List[str],
        extract_markdown: bool = True,
        only_main_content: bool = True,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Scrape multiple URLs in parallel with concurrency limit.

        Args:
            urls: List of URLs to scrape
            extract_markdown: Whether to extract markdown format
            only_main_content: Extract only main content
            include_tags: HTML tags to include
            exclude_tags: HTML tags to exclude

        Returns:
            List of scrape results
        """
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def scrape_with_limit(client: httpx.AsyncClient, url: str) -> Dict:
            async with semaphore:
                return await self._scrape_single_url(
                    client,
                    url,
                    extract_markdown=extract_markdown,
                    only_main_content=only_main_content,
                    include_tags=include_tags,
                    exclude_tags=exclude_tags,
                )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                tasks = [scrape_with_limit(client, url) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=False)

            return results

        except Exception as e:
            logger.error(f'Error in batch scraping: {e}')
            return [
                {
                    'url': url,
                    'success': False,
                    'error': f'Batch scraping failed: {str(e)}',
                    'markdown': None,
                    'html': None,
                    'metadata': {},
                }
                for url in urls
            ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        """
        Execute scraping of multiple URLs.

        Args:
            function_name: Name of function to execute (should be 'scrape_content')
            helper: OpenAI helper instance
            **kwargs: Function arguments

        Returns:
            Dictionary with scraping results
        """
        urls = kwargs.get('urls', [])
        extract_markdown = kwargs.get('extract_markdown', self.extract_markdown)
        only_main_content = kwargs.get('only_main_content', True)
        include_tags = kwargs.get('include_tags')
        exclude_tags = kwargs.get('exclude_tags')

        # Validate input
        if not urls:
            return {'result': 'No URLs provided'}

        if not isinstance(urls, list):
            return {'result': 'URLs must be a list'}

        if len(urls) > 20:
            return {'result': 'Maximum 20 URLs per request'}

        # Filter valid URLs
        valid_urls = [url for url in urls if isinstance(url, str) and url.strip()]
        if not valid_urls:
            return {'result': 'No valid URLs provided'}

        logger.info(f'Scraping {len(valid_urls)} URLs with max {self.max_concurrent} concurrent requests')

        # Execute batch scraping
        results = await self._scrape_batch(
            valid_urls,
            extract_markdown=extract_markdown,
            only_main_content=only_main_content,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )

        return {'result': results}
