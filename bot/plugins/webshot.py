import random
import string
from typing import Dict, List

import httpx

from .plugin import Plugin


class WebshotPlugin(Plugin):
    """
    A plugin to screenshot a website
    """

    def get_source_name(self) -> str:
        return 'WebShot'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'screenshot_website',
                    'description': 'Show screenshot/image of a website from a given url or domain name.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'url': {
                                'type': 'string',
                                'description': 'Website url or domain name. Correctly formatted url is required. Example: https://www.google.com',
                            }
                        },
                        'required': ['url'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    def generate_random_string(self, length):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        image_url = f'https://image.thum.io/get/maxAge/12/width/720/{kwargs["url"]}'

        async with httpx.AsyncClient() as client:
            await client.get(image_url)

            # download the actual image
            response = await client.get(image_url, timeout=30)

        response.raise_for_status()

        return {
            'direct_result': {
                'kind': 'photo',
                'value': response.content,
            }
        }
