from typing import Dict

from .plugin import Plugin


class AutoTextToSpeech(Plugin):
    """
    A plugin to convert text to speech using Openai Speech API
    """

    def get_source_name(self) -> str:
        return 'TTS'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'translate_text_to_speech',
                'description': 'Translate text to speech using OpenAI API',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'The text to translate to speech',
                        },
                    },
                    'required': ['text'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        data, _ = await helper.generate_speech(text=kwargs['text'])
        return {'direct_result': {'kind': 'voice', 'value': data}}
