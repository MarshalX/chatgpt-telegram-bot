import os
from typing import Dict, List

from googleapiclient.discovery import build

from .plugin import Plugin


class GoogleWebSearchPlugin(Plugin):
    """
    A plugin to search the web for a given query, using Google
    """

    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.cse_id = os.getenv('GOOGLE_CSE_ID')

    def get_source_name(self) -> str:
        return 'Google'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'web_search',
                    'description': 'Execute a web search for the given query and return a list of results',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string', 'description': 'the user query'},
                        },
                        'required': ['query'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        service = build('customsearch', 'v1', developerKey=self.api_key)
        results = service.cse().list(q=kwargs['query'], cx=self.cse_id).execute()

        items = results.get('items', [])

        if not items:
            return {'result': 'No good Google Search Result was found'}

        def to_metadata(result: Dict) -> Dict[str, str]:
            return {
                'snippet': result['snippet'],
                'title': result['title'],
                'link': result['link'],
            }

        return {'result': [to_metadata(item) for item in items]}
