from typing import Dict

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

from .plugin import Plugin


class YoutubeTranscriptPlugin(Plugin):
    """
    A plugin to query text from YouTube video transcripts
    """

    def get_source_name(self) -> str:
        return 'YouTube Transcript'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'youtube_video_transcript',
                'description': 'Get the transcript of a YouTube video for a given YouTube video ID',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'video_id': {
                            'type': 'string',
                            'description': 'YouTube video ID. For example, for the video https://youtu.be/dQw4w9WgXcQ, the video ID is dQw4w9WgXcQ',
                        }
                    },
                    'required': ['video_id'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            video_id = kwargs.get('video_id')
            if not video_id:
                return {'result': 'Video ID not provided'}

            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'ru', 'pl'])
            json_transcript = JSONFormatter().format_transcript(transcript)

            return {
                'transcript': json_transcript,
            }
        except Exception as e:
            return {'error': 'An unexpected error occurred: ' + str(e)}
