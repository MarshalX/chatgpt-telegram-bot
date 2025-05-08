from typing import Dict, List

from telegram.constants import ReactionEmoji

from .plugin import Plugin


class TelegramToolkitPlugin(Plugin):
    """
    A plugin providing direct access to Telegram API features
    """

    _emojis = [emoji.value for emoji in ReactionEmoji]
    _emojis_set = set(_emojis)

    def get_source_name(self) -> str:
        return 'Telegram Toolkit'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'set_reaction',
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
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'reply_photo',
                    'description': 'Send photo as a reply to the message',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'photo': {
                                'type': 'string',
                                'description': 'URI to the photo, bytes, or file ID',
                            },
                            'caption': {
                                'type': 'string',
                                'description': 'Photo caption (could be empty string)',
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
                    'name': 'reply_album',
                    'description': 'Send multiple photos as an album in reply to the message',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'photos': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'Array of URIs, bytes, or file IDs of photos (max 10)',
                            },
                            'caption': {
                                'type': 'string',
                                'description': 'Album caption (could be empty string)',
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
                    'name': 'reply_document',
                    'description': 'Send document/file/gif as a reply to the message',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'document': {
                                'type': 'string',
                                'description': 'URL, bytes, or file ID of the document',
                            },
                            'caption': {
                                'type': 'string',
                                'description': 'Document caption (could be empty string)',
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
                    'name': 'reply_voice',
                    'description': 'Send voice message as a reply',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'voice': {
                                'type': 'string',
                                'description': 'URL, bytes, or file ID of the voice message',
                            },
                            'caption': {
                                'type': 'string',
                                'description': 'Voice message caption (could be empty string)',
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
                    'name': 'reply_dice',
                    'description': 'Send a dice in the chat, with a random number between 1 and 6',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'emoji': {
                                'type': 'string',
                                'enum': ['🎲', '🎯', '🏀', '⚽', '🎳', '🎰'],
                                'description': 'Emoji on which the dice throw animation is based.'
                                'Dice can have values 1-6 for "🎲", "🎯" and "🎳", values 1-5 for "🏀" '
                                'and "⚽", and values 1-64 for "🎰". Defaults to "🎲".',
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
                    'name': 'reply_poll',
                    'description': 'Send a poll as a reply',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'question': {
                                'type': 'string',
                                'description': 'Poll question',
                            },
                            'options': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'List of poll options',
                            },
                            'is_anonymous': {
                                'type': 'boolean',
                                'description': 'Whether the poll is anonymous. Default is False',
                            },
                            'allows_multiple_answers': {
                                'type': 'boolean',
                                'description': 'Whether the poll allows multiple answers. Default is False',
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
                    'name': 'reply_location',
                    'description': 'Send a location as a reply',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'latitude': {
                                'type': 'number',
                                'description': 'Latitude',
                            },
                            'longitude': {
                                'type': 'number',
                                'description': 'Longitude',
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
                    'name': 'reply_venue',
                    'description': 'Send a venue as a reply',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'latitude': {
                                'type': 'number',
                                'description': 'Latitude',
                            },
                            'longitude': {
                                'type': 'number',
                                'description': 'Longitude',
                            },
                            'title': {
                                'type': 'string',
                                'description': 'Venue title',
                            },
                            'address': {
                                'type': 'string',
                                'description': 'Venue address',
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
                    'name': 'reply_contact',
                    'description': 'Send a contact as a reply',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'phone_number': {
                                'type': 'string',
                                'description': 'Contact phone number',
                            },
                            'first_name': {
                                'type': 'string',
                                'description': 'Contact first name',
                            },
                            'last_name': {
                                'type': 'string',
                                'description': 'Contact last name',
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
                    'name': 'reply_video',
                    'description': 'Send a video as a reply',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'video': {
                                'type': 'string',
                                'description': 'URL, bytes, or file ID of the video',
                            },
                            'caption': {
                                'type': 'string',
                                'description': 'Video caption (could be empty string)',
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
                    'name': 'reply_video_note',
                    'description': 'Send a round short video (up to 60 sec) as a reply',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'video_note': {
                                'type': 'string',
                                'description': 'URL, bytes, or file ID of the video note',
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
                    'name': 'reply_audio',
                    'description': 'Send an audio file as a reply',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'audio': {
                                'type': 'string',
                                'description': 'URL, bytes, or file ID of the audio',
                            },
                            'caption': {
                                'type': 'string',
                                'description': 'Audio caption (could be empty string)',
                            },
                        },
                        'required': ['audio', 'caption'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            },
        ]

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
    async def _handle_reply_photo(**kwargs) -> Dict:
        photo = kwargs.get('photo')
        caption = kwargs.get('caption', '')

        return {'direct_result': {'kind': 'photo', 'photo': photo, 'caption': caption}}

    @staticmethod
    async def _handle_reply_album(**kwargs) -> Dict:
        photos = kwargs.get('photos', [])
        caption = kwargs.get('caption', '')

        # Limit to 10 photos
        photos = photos[:10]

        return {'direct_result': {'kind': 'album', 'photos': photos, 'caption': caption}}

    @staticmethod
    async def _handle_reply_document(**kwargs) -> Dict:
        document = kwargs.get('document')
        caption = kwargs.get('caption', '')

        return {'direct_result': {'kind': 'document', 'document': document, 'caption': caption}}

    @staticmethod
    async def _handle_reply_voice(**kwargs) -> Dict:
        voice = kwargs.get('voice')
        caption = kwargs.get('caption', '')

        return {'direct_result': {'kind': 'voice', 'voice': voice, 'caption': caption}}

    @staticmethod
    async def _handle_reply_dice(**kwargs) -> Dict:
        return {
            'direct_result': {
                'kind': 'dice',
                'emoji': kwargs.get('emoji', '🎲'),
            }
        }

    @staticmethod
    async def _handle_reply_poll(**kwargs) -> Dict:
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
    async def _handle_reply_location(**kwargs) -> Dict:
        latitude = kwargs.get('latitude')
        longitude = kwargs.get('longitude')

        return {'direct_result': {'kind': 'location', 'latitude': latitude, 'longitude': longitude}}

    @staticmethod
    async def _handle_reply_venue(**kwargs) -> Dict:
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
    async def _handle_reply_contact(**kwargs) -> Dict:
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
    async def _handle_reply_video(**kwargs) -> Dict:
        video = kwargs.get('video')
        caption = kwargs.get('caption', '')

        return {'direct_result': {'kind': 'video', 'video': video, 'caption': caption}}

    @staticmethod
    async def _handle_reply_video_note(**kwargs) -> Dict:
        video_note = kwargs.get('video_note')

        return {'direct_result': {'kind': 'video_note', 'video_note': video_note}}

    @staticmethod
    async def _handle_reply_audio(**kwargs) -> Dict:
        audio = kwargs.get('audio')
        caption = kwargs.get('caption', '')

        return {'direct_result': {'kind': 'audio', 'audio': audio, 'caption': caption}}

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        handlers = {
            'set_reaction': self._handle_set_reaction,
            'reply_photo': self._handle_reply_photo,
            'reply_album': self._handle_reply_album,
            'reply_document': self._handle_reply_document,
            'reply_voice': self._handle_reply_voice,
            'reply_dice': self._handle_reply_dice,
            'reply_poll': self._handle_reply_poll,
            'reply_location': self._handle_reply_location,
            'reply_venue': self._handle_reply_venue,
            'reply_contact': self._handle_reply_contact,
            'reply_video': self._handle_reply_video,
            'reply_video_note': self._handle_reply_video_note,
            'reply_audio': self._handle_reply_audio,
        }

        if function_name not in handlers:
            return {'error': f'Unknown function: {function_name}'}

        # Get the handler function or None if not found
        handler = handlers.get(function_name)
        return await handler(**kwargs)
