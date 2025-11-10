import os
from itertools import islice
from typing import Dict, List

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from .plugin import Plugin


class DDGSPlugin(Plugin):
    """
    A plugin to search the web for a given query, using DDGS
    """

    def __init__(self):
        self.safesearch = os.getenv('DDGS_SAFESEARCH', 'moderate')
        self.max_results = 5

    def get_source_name(self) -> str:
        return 'DDGS'

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
                                ],
                                'description': 'The region to use for the search. Infer this from the language used for the'
                                'query. Default to `pl-pl` if not specified',
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
        try:
            results = DDGS().text(
                kwargs['query'],
                region=kwargs.get('region', 'pl-pl'),
                max_results=self.max_results,
                safesearch=self.safesearch,
            )
            results = list(islice(results, self.max_results))

            if results is None or len(results) == 0:
                return {'result': 'No good DDGS results were found'}

            def to_metadata(result: Dict) -> Dict[str, str]:
                return {
                    'snippet': result['body'],
                    'title': result['title'],
                    'link': result['href'],
                }

            return {'result': [to_metadata(result) for result in results]}

        except DDGSException as e:
            return {'error': f'Error during DDGS search: {str(e)}'}
