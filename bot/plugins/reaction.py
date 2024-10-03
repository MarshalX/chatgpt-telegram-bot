from typing import Dict

from telegram.constants import ReactionEmoji

from .plugin import Plugin


class ReactionPlugin(Plugin):
    """
    A plugin to respond with a specified emoji reaction
    """

    _emojis = [emoji.value for emoji in ReactionEmoji]
    _emojis_set = set(_emojis)

    def get_source_name(self) -> str:
        return 'Reaction'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'react_with_emoji',
                'description': 'Set emoji as a reaction to the reply message',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'reaction': {
                            'type': 'string',
                            'description': 'Emoji reaction to respond with',
                            'enum': self._emojis,
                        }
                    },
                    'required': ['reaction'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        reaction = kwargs.get('reaction')
        if reaction not in self._emojis_set:
            # fallback if model does not respect enum or required field
            reaction = ReactionEmoji.THUMBS_UP

        return {
            'direct_result': {
                'kind': 'reaction',
                'value': reaction,
            }
        }
