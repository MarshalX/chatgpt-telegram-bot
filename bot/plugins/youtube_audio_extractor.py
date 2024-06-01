import io
import logging
from typing import Dict

from pytube import YouTube

from .plugin import Plugin


class YouTubeAudioExtractorPlugin(Plugin):
    """
    A plugin to extract audio from a YouTube video
    """

    def get_source_name(self) -> str:
        return 'YouTube Audio Extractor'

    def get_spec(self) -> [Dict]:
        return [
            {
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
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        link = kwargs['youtube_link']
        try:
            video = YouTube(link)
            audio = video.streams.filter(only_audio=True, file_extension='mp4').first()
            file_obj = io.BytesIO()
            audio.stream_to_buffer(file_obj)
            return {'direct_result': {'kind': 'file', 'value': file_obj}}
        except Exception as e:
            logging.warning(f'Failed to extract audio from YouTube video: {str(e)}')
            return {'result': 'Failed to extract audio'}
