import os
from typing import Dict, List, Optional, Set

from telegram.constants import ReactionEmoji

from .plugin import Plugin


class TelegramToolkitPlugin(Plugin):
    """
    A plugin providing direct access to Telegram API features.

    By default all functions are enabled. To restrict which functions are exposed,
    either pass ``enabled_functions`` explicitly or set the ``TELEGRAM_TOOLKIT_FUNCTIONS``
    environment variable to a comma-separated list of function names, e.g.:

        TELEGRAM_TOOLKIT_FUNCTIONS=set_reaction,send_photo,send_album
    """

    ALL_FUNCTIONS = {
        'set_reaction',
        'send_photo',
        'send_album',
        'send_document',
        'send_voice',
        'send_dice',
        'send_poll',
        'send_location',
        'send_venue',
        'send_contact',
        'send_video',
        'send_video_note',
        'send_audio',
    }

    _emojis = [emoji.value for emoji in ReactionEmoji]
    _emojis_set = set(_emojis)

    def __init__(self, enabled_functions: Optional[Set[str]] = None):
        self._enabled = self.ALL_FUNCTIONS
        if enabled_functions is not None:
            self._enabled = enabled_functions & self.ALL_FUNCTIONS
        else:
            env = os.getenv('TELEGRAM_TOOLKIT_FUNCTIONS', '').strip()
            if env:
                self._enabled = {f.strip() for f in env.split(',') if f.strip()} & self.ALL_FUNCTIONS

    def get_source_name(self) -> str:
        return 'Telegram Toolkit'

    def get_spec(self) -> List[Dict]:
        all_specs = [
            {
                'type': 'function',
                'function': {
                    'name': 'set_reaction',
                    'description': (
                        "React to the user's message with a single emoji. "
                        'Use this instead of a text reply when a brief acknowledgement is more appropriate '
                        'than a full response — e.g. 👍 to confirm, 👀 to say "noted", ❤️ to express appreciation, '
                        '🔥 for excitement. Do not use this together with a text reply for the same message.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'reaction': {
                                'type': 'string',
                                'description': (
                                    'A single Telegram reaction emoji. '
                                    'Pick the emoji that best matches the intent (e.g. 👍, ❤️, 😂, 🔥, 👀, 🎉). '
                                    'Must be one of the allowed values.'
                                ),
                                'enum': self._emojis,
                            }
                        },
                        'required': ['reaction'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_photo',
                    'description': (
                        "Send a single image as a reply to the user's message. "
                        'Use this when you have exactly one photo or image to share. '
                        'For two or more images, use send_album instead.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'photo': {
                                'type': 'string',
                                'description': (
                                    'The image to send. Accepted formats: '
                                    'public URL (e.g. https://example.com/photo.jpg), '
                                    'base64 data URI (e.g. data:image/jpeg;base64,...), '
                                    'or Telegram file ID (e.g. AgACAgIAAxk...).'
                                ),
                            },
                            'caption': {
                                'type': 'string',
                                'description': (
                                    'Optional text displayed below the photo in Telegram. '
                                    'Use "" if no caption is needed.'
                                ),
                            },
                        },
                        'required': ['photo', 'caption'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_album',
                    'description': (
                        "Send 2–10 images as a grouped media album in reply to the user's message. "
                        'Use this when you have multiple photos to share at once. '
                        'For a single image, use send_photo instead.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'photos': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': (
                                    'Array of 2–10 images to group into an album. '
                                    'Each item can be a public URL (https://...), '
                                    'a base64 data URI (data:image/jpeg;base64,...), '
                                    'or a Telegram file ID. Items beyond the 10th are ignored.'
                                ),
                            },
                            'caption': {
                                'type': 'string',
                                'description': (
                                    'Optional text displayed below the album in Telegram (appears under the first photo). '
                                    'Use "" if no caption is needed.'
                                ),
                            },
                        },
                        'required': ['photos', 'caption'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_document',
                    'description': (
                        "Send a file or document as a reply to the user's message. "
                        'Use this for downloadable files (PDF, ZIP, CSV, GIF, etc.). '
                        'For images displayed inline, use send_photo. '
                        'For playable video, use send_video.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'document': {
                                'type': 'string',
                                'description': (
                                    'The file to send. Accepted formats: '
                                    'public URL (e.g. https://example.com/report.pdf), '
                                    'base64 data URI (e.g. data:application/pdf;base64,...), '
                                    'or Telegram file ID.'
                                ),
                            },
                            'caption': {
                                'type': 'string',
                                'description': (
                                    'Optional text displayed below the document in Telegram. '
                                    'Use "" if no caption is needed.'
                                ),
                            },
                        },
                        'required': ['document', 'caption'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_voice',
                    'description': (
                        'Send a voice message that appears as a waveform audio note in Telegram. '
                        'Use this for spoken audio (OGG/Opus preferred). '
                        'For music or audio files with track metadata, use send_audio instead.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'voice': {
                                'type': 'string',
                                'description': (
                                    'The voice audio to send. Accepted formats: '
                                    'public URL (e.g. https://example.com/message.ogg), '
                                    'base64 data URI (e.g. data:audio/ogg;base64,...), '
                                    'or Telegram file ID.'
                                ),
                            },
                            'caption': {
                                'type': 'string',
                                'description': (
                                    'Optional text displayed below the voice message in Telegram. '
                                    'Use "" if no caption is needed.'
                                ),
                            },
                        },
                        'required': ['voice', 'caption'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_dice',
                    'description': (
                        'Send an animated dice that rolls to a random value in the chat. '
                        'Use this for games, random picks, or fun interactions when the user wants to roll dice, '
                        'shoot a basketball, spin a slot machine, etc.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'emoji': {
                                'type': 'string',
                                'enum': ['🎲', '🎯', '🏀', '⚽', '🎳', '🎰'],
                                'description': (
                                    'The animated dice type to send. '
                                    'Values: 🎲 and 🎯 and 🎳 roll 1–6; '
                                    '🏀 and ⚽ score 1–5; '
                                    '🎰 spins 1–64. '
                                    'Default to 🎲 unless the user specifies a different type.'
                                ),
                            }
                        },
                        'required': ['emoji'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_poll',
                    'description': (
                        'Send an interactive poll that chat members can vote on. '
                        'Use this when the user asks to create a vote, survey, or poll.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'question': {
                                'type': 'string',
                                'description': 'The poll question shown at the top (e.g. "Which framework do you prefer?").',
                            },
                            'options': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'List of 2–10 answer choices shown to voters (e.g. ["React", "Vue", "Angular"]).',
                            },
                            'is_anonymous': {
                                'type': 'boolean',
                                'description': (
                                    "Whether voters' identities are hidden. "
                                    'Set to true to keep votes anonymous, false to show who voted for what. '
                                    'Default is false.'
                                ),
                            },
                            'allows_multiple_answers': {
                                'type': 'boolean',
                                'description': (
                                    'Whether voters can select more than one option. '
                                    'Set to true for multi-choice, false for single-choice polls. '
                                    'Default is false.'
                                ),
                            },
                        },
                        'required': ['question', 'options', 'is_anonymous', 'allows_multiple_answers'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_location',
                    'description': (
                        'Send a map pin with geographic coordinates as a reply. '
                        'Use this when the user asks to share a location or point on a map without a venue name. '
                        'If you also have a name and address for the place, use send_venue instead.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'latitude': {
                                'type': 'number',
                                'description': 'Geographic latitude in decimal degrees (e.g. 48.8566 for Paris).',
                            },
                            'longitude': {
                                'type': 'number',
                                'description': 'Geographic longitude in decimal degrees (e.g. 2.3522 for Paris).',
                            },
                        },
                        'required': ['latitude', 'longitude'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_venue',
                    'description': (
                        'Send a named venue with a map pin, title, and address as a reply. '
                        'Use this when sharing a business, landmark, or any named place '
                        '(e.g. a restaurant, hotel, airport). '
                        'If you only have coordinates with no name, use send_location instead.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'latitude': {
                                'type': 'number',
                                'description': 'Geographic latitude of the venue in decimal degrees.',
                            },
                            'longitude': {
                                'type': 'number',
                                'description': 'Geographic longitude of the venue in decimal degrees.',
                            },
                            'title': {
                                'type': 'string',
                                'description': 'Name of the venue (e.g. "Eiffel Tower", "JFK Airport", "Le Bernardin").',
                            },
                            'address': {
                                'type': 'string',
                                'description': 'Street address of the venue (e.g. "Champ de Mars, 5 Av. Anatole France, 75007 Paris").',
                            },
                        },
                        'required': ['latitude', 'longitude', 'title', 'address'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_contact',
                    'description': (
                        'Send a contact card with a phone number and name as a reply. '
                        "Use this when the user asks to share someone's contact details."
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'phone_number': {
                                'type': 'string',
                                'description': 'Phone number in international format (e.g. "+12025551234").',
                            },
                            'first_name': {
                                'type': 'string',
                                'description': "Contact's first name.",
                            },
                            'last_name': {
                                'type': 'string',
                                'description': 'Contact\'s last name. Use "" if not available.',
                            },
                        },
                        'required': ['phone_number', 'first_name', 'last_name'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_video',
                    'description': (
                        'Send a video file that plays inline in Telegram as a reply. '
                        'Use this for regular video clips. '
                        'For short round personal video messages, use send_video_note instead.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'video': {
                                'type': 'string',
                                'description': (
                                    'The video to send. Accepted formats: '
                                    'public URL (e.g. https://example.com/clip.mp4), '
                                    'base64 data URI (e.g. data:video/mp4;base64,...), '
                                    'or Telegram file ID.'
                                ),
                            },
                            'caption': {
                                'type': 'string',
                                'description': (
                                    'Optional text displayed below the video in Telegram. '
                                    'Use "" if no caption is needed.'
                                ),
                            },
                        },
                        'required': ['video', 'caption'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_video_note',
                    'description': (
                        'Send a short round video note (up to 60 seconds) that appears as a circle in Telegram. '
                        'Use this for brief personal-style video messages. '
                        'For regular video clips, use send_video instead.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'video_note': {
                                'type': 'string',
                                'description': (
                                    'The round video to send. Accepted formats: '
                                    'public URL (e.g. https://example.com/note.mp4), '
                                    'base64 data URI (e.g. data:video/mp4;base64,...), '
                                    'or Telegram file ID. '
                                    'Must be square (1:1 aspect ratio) and up to 60 seconds.'
                                ),
                            }
                        },
                        'required': ['video_note'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'send_audio',
                    'description': (
                        "Send a music or audio track that appears in Telegram's audio player with title and artist metadata. "
                        'Use this for songs or audio files. '
                        'For spoken voice messages, use send_voice instead.'
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'audio': {
                                'type': 'string',
                                'description': (
                                    'The audio file to send. Accepted formats: '
                                    'public URL (e.g. https://example.com/song.mp3), '
                                    'base64 data URI (e.g. data:audio/mpeg;base64,...), '
                                    'or Telegram file ID.'
                                ),
                            },
                            'caption': {
                                'type': 'string',
                                'description': (
                                    'Optional text displayed below the audio player in Telegram. '
                                    'Use "" if no caption is needed.'
                                ),
                            },
                        },
                        'required': ['audio', 'caption'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
        ]
        return [s for s in all_specs if s['function']['name'] in self._enabled]

    async def _handle_set_reaction(self, **kwargs) -> Dict:
        reaction = kwargs.get('reaction')
        if reaction not in self._emojis_set:
            # fallback if model does not respect enum or required field
            reaction = ReactionEmoji.THUMBS_UP

        return {
            'direct_result': {
                'kind': 'reaction',
                'reaction': reaction,
            }
        }

    @staticmethod
    async def _handle_send_photo(**kwargs) -> Dict:
        photo = kwargs.get('photo')
        caption = kwargs.get('caption', '')

        return {'direct_result': {'kind': 'photo', 'photo': photo, 'caption': caption}}

    @staticmethod
    async def _handle_send_album(**kwargs) -> Dict:
        photos = kwargs.get('photos', [])
        caption = kwargs.get('caption', '')

        # Limit to 10 photos
        photos = photos[:10]

        return {'direct_result': {'kind': 'album', 'photos': photos, 'caption': caption}}

    @staticmethod
    async def _handle_send_document(**kwargs) -> Dict:
        document = kwargs.get('document')
        caption = kwargs.get('caption', '')

        return {'direct_result': {'kind': 'document', 'document': document, 'caption': caption}}

    @staticmethod
    async def _handle_send_voice(**kwargs) -> Dict:
        voice = kwargs.get('voice')
        caption = kwargs.get('caption', '')

        return {'direct_result': {'kind': 'voice', 'voice': voice, 'caption': caption}}

    @staticmethod
    async def _handle_send_dice(**kwargs) -> Dict:
        return {
            'direct_result': {
                'kind': 'dice',
                'emoji': kwargs.get('emoji', '🎲'),
            }
        }

    @staticmethod
    async def _handle_send_poll(**kwargs) -> Dict:
        question = kwargs.get('question', '?')
        options = kwargs.get('options', [])
        is_anonymous = kwargs.get('is_anonymous', False)
        allows_multiple_answers = kwargs.get('allows_multiple_answers', False)

        return {
            'direct_result': {
                'kind': 'poll',
                'question': question,
                'options': options,
                'is_anonymous': is_anonymous,
                'allows_multiple_answers': allows_multiple_answers,
            }
        }

    @staticmethod
    async def _handle_send_location(**kwargs) -> Dict:
        latitude = kwargs.get('latitude')
        longitude = kwargs.get('longitude')

        return {'direct_result': {'kind': 'location', 'latitude': latitude, 'longitude': longitude}}

    @staticmethod
    async def _handle_send_venue(**kwargs) -> Dict:
        latitude = kwargs.get('latitude')
        longitude = kwargs.get('longitude')
        title = kwargs.get('title')
        address = kwargs.get('address')

        return {
            'direct_result': {
                'kind': 'venue',
                'latitude': latitude,
                'longitude': longitude,
                'title': title,
                'address': address,
            }
        }

    @staticmethod
    async def _handle_send_contact(**kwargs) -> Dict:
        phone_number = kwargs.get('phone_number')
        first_name = kwargs.get('first_name')
        last_name = kwargs.get('last_name', '')

        return {
            'direct_result': {
                'kind': 'contact',
                'phone_number': phone_number,
                'first_name': first_name,
                'last_name': last_name,
            }
        }

    @staticmethod
    async def _handle_send_video(**kwargs) -> Dict:
        video = kwargs.get('video')
        caption = kwargs.get('caption', '')

        return {'direct_result': {'kind': 'video', 'video': video, 'caption': caption}}

    @staticmethod
    async def _handle_send_video_note(**kwargs) -> Dict:
        video_note = kwargs.get('video_note')

        return {'direct_result': {'kind': 'video_note', 'video_note': video_note}}

    @staticmethod
    async def _handle_send_audio(**kwargs) -> Dict:
        audio = kwargs.get('audio')
        caption = kwargs.get('caption', '')

        return {'direct_result': {'kind': 'audio', 'audio': audio, 'caption': caption}}

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        handlers = {
            'set_reaction': self._handle_set_reaction,
            'send_photo': self._handle_send_photo,
            'send_album': self._handle_send_album,
            'send_document': self._handle_send_document,
            'send_voice': self._handle_send_voice,
            'send_dice': self._handle_send_dice,
            'send_poll': self._handle_send_poll,
            'send_location': self._handle_send_location,
            'send_venue': self._handle_send_venue,
            'send_contact': self._handle_send_contact,
            'send_video': self._handle_send_video,
            'send_video_note': self._handle_send_video_note,
            'send_audio': self._handle_send_audio,
        }

        if function_name not in handlers:
            return {'error': f'Unknown function: {function_name}'}

        handler = handlers.get(function_name)
        return await handler(**kwargs)
