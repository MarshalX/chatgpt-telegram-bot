import io
from typing import Dict, List

from pytube import YouTube

from .plugin import Plugin


class YouTubeAudioExtractorPlugin(Plugin):
    """
    A plugin to extract audio from a YouTube video
    """

    def get_source_name(self) -> str:
        return 'YouTube Audio Extractor'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'extract_youtube_audio',
                    'description': 'Extract audio from a YouTube video',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'youtube_link': {
                                'type': 'string',
                                'description': 'YouTube video link to extract audio from',
                            }
                        },
                        'required': ['youtube_link'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        link = kwargs['youtube_link']
        video = YouTube(link)
        audio = video.streams.filter(only_audio=True).first()
        file_obj = io.BytesIO()
        audio.stream_to_buffer(file_obj)
        return {'direct_result': {'kind': 'file', 'document': file_obj}}
