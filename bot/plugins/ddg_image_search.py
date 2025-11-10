import os
import random
from itertools import islice
from typing import Dict, List

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from .plugin import Plugin


class DDGImageSearchPlugin(Plugin):
    """
    A plugin to search images and GIFs for a given query, using DuckDuckGo
    """

    def __init__(self):
        self.safesearch = os.getenv('DDGS_SAFESEARCH', 'moderate')
        self.max_results = 10

    def get_source_name(self) -> str:
        return 'DuckDuckGo Images'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'search_images',
                    'description': 'Search image or GIFs for a given query',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The query to search for',
                            },
                            'type': {
                                'type': 'string',
                                'enum': ['photo', 'gif'],
                                'description': 'The type of image to search for. Default to `photo` if not specified',
                            },
                            'count': {
                                'type': 'integer',
                                'description': 'The number of images to return. Default to 10 if not specified',
                            },
                        },
                        'required': ['query', 'type', 'count'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            image_type = kwargs.get('type', 'photo')
            results = DDGS().images(
                query=kwargs['query'],
                safesearch=self.safesearch,
                max_results=kwargs.get('count', self.max_results),
                type_image=image_type,
            )

            results = list(islice(results, self.max_results))
            if not results or len(results) == 0:
                return {'result': 'No results found'}

            # Shuffle the results to avoid always returning the same image
            random.shuffle(results)

            if image_type == 'gif':
                return {
                    'direct_result': {
                        'kind': 'gif',
                        'document': results[0]['image'],
                    }
                }

            return {
                'direct_result': {
                    'kind': 'album',
                    'photos': [result['image'] for result in results],
                }
            }
        except DDGSException as e:
            return {'error': f'Error during DDGS image search: {str(e)}'}
