import os
from itertools import islice
from typing import Dict, List

from duckduckgo_search import DDGS

from .plugin import Plugin


class DDGWebSearchPlugin(Plugin):
    """
    A plugin to search the web for a given query, using DuckDuckGo
    """

    def __init__(self):
        self.safesearch = os.getenv('DUCKDUCKGO_SAFESEARCH', 'moderate')

    def get_source_name(self) -> str:
        return 'DuckDuckGo'

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
                            'region': {
                                'type': 'string',
                                'enum': [
                                    'pl-pl',
                                    'ru-ru',
                                    'uk-en',
                                    'us-en',
                                    'wt-wt',
                                ],
                                'description': 'The region to use for the search. Infer this from the language used for the'
                                'query. Default to `wt-wt` if not specified',
                            },
                        },
                        'required': ['query', 'region'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(
                kwargs['query'],
                region=kwargs.get('region', 'wt-wt'),
                safesearch=self.safesearch,
            )
            results = list(islice(ddgs_gen, 3))

            if results is None or len(results) == 0:
                return {'result': 'No good DuckDuckGo Search Result was found'}

            def to_metadata(result: Dict) -> Dict[str, str]:
                return {
                    'snippet': result['body'],
                    'title': result['title'],
                    'link': result['href'],
                }

            return {'result': [to_metadata(result) for result in results]}
