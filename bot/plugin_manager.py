import json
from typing import Dict

from plugins.auto_tts import AutoTextToSpeech
from plugins.ddg_image_search import DDGImageSearchPlugin
from plugins.dice import DicePlugin
from plugins.google_web_search import GoogleWebSearchPlugin
from plugins.gtts_text_to_speech import GTTSTextToSpeech
from plugins.reaction import ReactionPlugin
from plugins.spotify import SpotifyPlugin
from plugins.weather import WeatherPlugin
from plugins.website_content import WebsiteContentPlugin
from plugins.wolfram_alpha import WolframAlphaPlugin
from plugins.worldtimeapi import WorldTimeApiPlugin
from plugins.youtube_audio_extractor import YouTubeAudioExtractorPlugin
from plugins.youtube_transcript import YoutubeTranscriptPlugin


class PluginManager:
    """
    A class to manage the plugins and call the correct functions
    """

    def __init__(self, config):
        plugin_mapping = {
            'reaction': ReactionPlugin,
            'wolfram': WolframAlphaPlugin,
            'weather': WeatherPlugin,
            # 'ddg_web_search': DDGWebSearchPlugin,
            # 'ddg_translate': DDGTranslatePlugin,
            'google_web_search': GoogleWebSearchPlugin,
            'ddg_image_search': DDGImageSearchPlugin,
            'spotify': SpotifyPlugin,
            'worldtimeapi': WorldTimeApiPlugin,
            'youtube_audio_extractor': YouTubeAudioExtractorPlugin,
            'dice': DicePlugin,
            'gtts_text_to_speech': GTTSTextToSpeech,
            'auto_tts': AutoTextToSpeech,
            # 'whois': WhoisPlugin,
            # 'webshot': WebshotPlugin,
            # 'iplocation': IpLocationPlugin,
            'website_content': WebsiteContentPlugin,
            'youtube_transcript': YoutubeTranscriptPlugin,
        }

        enabled_plugins = config.get('plugins', [])
        if 'all' in enabled_plugins:
            enabled_plugins = list(plugin_mapping.keys())

        self.plugins = [plugin_mapping[plugin]() for plugin in enabled_plugins if plugin in plugin_mapping]

    def get_functions_specs(self):
        """
        Return the list of function specs that can be called by the model
        """
        return [spec for specs in map(lambda plugin: plugin.get_spec(), self.plugins) for spec in specs]

    async def call_function(self, function_name, helper, arguments) -> Dict:
        """
        Call a function based on the name and parameters provided
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return {'error': f'Function {function_name} not found'}

        return await plugin.execute(function_name, helper, **json.loads(arguments))

    def get_plugin_source_name(self, function_name) -> str:
        """
        Return the source name of the plugin
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return ''
        return plugin.get_source_name()

    def __get_plugin_by_function_name(self, function_name):
        return next(
            (
                plugin
                for plugin in self.plugins
                if function_name in map(lambda spec: spec.get('name'), plugin.get_spec())
            ),
            None,
        )
