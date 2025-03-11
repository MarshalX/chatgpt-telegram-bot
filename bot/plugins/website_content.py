from typing import Dict

import httpx
import readability

from .plugin import Plugin


class WebsiteContentPlugin(Plugin):
    """
    A plugin to query text from a website
    """

    def get_source_name(self) -> str:
        return 'Website Content'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'website_content',
                'description': 'Get and clean up the main body text and title for an URL',
                'parameters': {
                    'type': 'object',
                    'properties': {'url': {'type': 'string', 'description': 'URL address'}},
                    'required': ['url'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        url = kwargs.get('url')
        if not url:
            return {'error': 'URL not provided'}

        async with httpx.AsyncClient() as client:
            response = await client.get(url)

        response.raise_for_status()
        doc = readability.Document(response.content)

        return {
            'title': doc.title(),
            'summary': doc.summary(),
        }
