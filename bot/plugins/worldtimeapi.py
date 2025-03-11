import os
from datetime import datetime
from typing import Dict

import httpx

from .plugin import Plugin


class WorldTimeApiPlugin(Plugin):
    """
    A plugin to get the current time from a given timezone, using WorldTimeAPI
    """

    def __init__(self):
        default_timezone = os.getenv('WORLDTIME_DEFAULT_TIMEZONE')
        if not default_timezone:
            raise ValueError('WORLDTIME_DEFAULT_TIMEZONE environment variable must be set to use WorldTimeApiPlugin')
        self.default_timezone = default_timezone

    def get_source_name(self) -> str:
        return 'WorldTimeAPI'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'worldtimeapi',
                'description': 'Get the current time from a given timezone',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'timezone': {
                            'type': 'string',
                            'description': 'The timezone identifier (e.g: `Europe/Rome`). Infer this from the location.'
                            f'Use {self.default_timezone} if not specified.',
                        }
                    },
                    'required': ['timezone'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        timezone = kwargs.get('timezone', self.default_timezone)
        url = f'https://worldtimeapi.org/api/timezone/{timezone}'

        async with httpx.AsyncClient() as client:
            response = await client.get(url)

        response.raise_for_status()
        wtr = response.json().get('datetime')

        wtr_obj = datetime.strptime(wtr, '%Y-%m-%dT%H:%M:%S.%f%z')
        time_24hr = wtr_obj.strftime('%H:%M:%S')
        time_12hr = wtr_obj.strftime('%I:%M:%S %p')

        return {'24hr': time_24hr, '12hr': time_12hr}
