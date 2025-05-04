from typing import Dict, List

import httpx

from .plugin import Plugin


class IpLocationPlugin(Plugin):
    """
    A plugin to get geolocation and other information for a given IP address
    """

    def get_source_name(self) -> str:
        return 'IP.FM'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'iplocation',
                    'description': 'Get information for an IP address using the IP.FM API.',
                    'parameters': {
                        'type': 'object',
                        'properties': {'ip': {'type': 'string', 'description': 'IP Address'}},
                        'required': ['ip'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        ip = kwargs.get('ip')
        if not ip:
            return {'Error': 'IP address not provided'}

        url = f'https://api.ip.fm/?ip={ip}'

        async with httpx.AsyncClient() as client:
            response = (await client.get(url)).json()

        country = response.get('data', {}).get('country', 'None')
        subdivisions = response.get('data', {}).get('subdivisions', 'None')
        city = response.get('data', {}).get('city', 'None')
        location = ', '.join(filter(None, [country, subdivisions, city])) or 'None'

        asn = response.get('data', {}).get('asn', 'None')
        as_name = response.get('data', {}).get('as_name', 'None')
        as_domain = response.get('data', {}).get('as_domain', 'None')
        return {
            'Location': location,
            'ASN': asn,
            'AS Name': as_name,
            'AS Domain': as_domain,
        }
