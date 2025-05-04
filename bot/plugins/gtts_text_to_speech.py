import io
from typing import Dict, List

from gtts import gTTS

from .plugin import Plugin


class GTTSTextToSpeech(Plugin):
    """
    A plugin to convert text to speech using Google Translate's Text to Speech API
    """

    def get_source_name(self) -> str:
        return 'gTTS'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'google_translate_text_to_speech',
                    'description': "Translate text to speech using Google Translate's Text to Speech API",
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'text': {
                                'type': 'string',
                                'description': 'The text to translate to speech',
                            },
                            'lang': {
                                'type': 'string',
                                'description': 'The language of the text to translate to speech.'
                                'Infer this from the language of the text.',
                            },
                        },
                        'required': ['text', 'lang'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        tts = gTTS(kwargs['text'], lang=kwargs.get('lang', 'en'))
        file_obj = io.BytesIO()
        tts.write_to_fp(file_obj)
        file_obj.seek(0)
        return {'direct_result': {'kind': 'voice', 'value': file_obj}}
