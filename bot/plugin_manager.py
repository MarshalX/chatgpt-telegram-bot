import json
import logging
from typing import TYPE_CHECKING, Dict

from plugins.auto_tts import AutoTextToSpeech
from plugins.code_execution import CodeExecutionPlugin
from plugins.ddg_image_search import DDGImageSearchPlugin
from plugins.dux_distributed_global_search import DDGSPlugin
from plugins.gtts_text_to_speech import GTTSTextToSpeech
from plugins.searxng_search import SearxngSearchPlugin
from plugins.sequential_thinking import SequentialThinkingPlugin
from plugins.telegram_direct import TelegramToolkitPlugin
from plugins.weather import WeatherPlugin
from plugins.website_content import WebsiteContentPlugin
from plugins.wolfram_alpha import WolframAlphaPlugin
from plugins.worldtimeapi import WorldTimeApiPlugin
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

if TYPE_CHECKING:
    from openai_helper import OpenAIHelper


class PluginManager:
    """
    A class to manage the plugins and call the correct functions
    """

    def __init__(self, config):
        plugin_mapping = {
            'telegram': TelegramToolkitPlugin,
            'wolfram': WolframAlphaPlugin,
            'weather': WeatherPlugin,
            # 'ddg_web_search': DDGWebSearchPlugin,
            # 'ddg_translate': DDGTranslatePlugin,
            #'google_web_search': GoogleWebSearchPlugin,
            'ddg_image_search': DDGImageSearchPlugin,
            'dux_distributed_global_search': DDGSPlugin,
            'searxng': SearxngSearchPlugin,
            # 'spotify': SpotifyPlugin,
            'worldtimeapi': WorldTimeApiPlugin,
            # 'youtube_audio_extractor': YouTubeAudioExtractorPlugin,
            'gtts_text_to_speech': GTTSTextToSpeech,
            'auto_tts': AutoTextToSpeech,
            # 'whois': WhoisPlugin,
            # 'webshot': WebshotPlugin,
            # 'iplocation': IpLocationPlugin,
            'website_content': WebsiteContentPlugin,
            # 'youtube_transcript': YoutubeTranscriptPlugin,
            'code': CodeExecutionPlugin,
            'thinking': SequentialThinkingPlugin,
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

    @retry(
        reraise=True,
        retry=retry_if_exception_type(BaseException),
        wait=wait_exponential_jitter(),
        stop=stop_after_attempt(3),
    )
    async def call_function(self, chat_id: str, function_name: str, helper: 'OpenAIHelper', arguments: str) -> Dict:
        try:
            return await self.__call_function(chat_id, function_name, helper, arguments)
        except Exception as e:
            logging.error(f'Error calling function {function_name}:', exc_info=e)
            return {'error': f'Error calling function {function_name}'}

    async def __call_function(self, chat_id: str, function_name: str, helper: 'OpenAIHelper', arguments: str) -> Dict:
        """
        Call a function based on the name and parameters provided
        """
        plugin = self.__get_plugin_by_function_name(function_name)
        if not plugin:
            return {'error': f'Function {function_name} not found'}

        if not getattr(plugin, '_bootstrap', True):
            logging.info(f'Bootstrapping plugin {plugin.get_source_name()}')
            await plugin.bootstrap()

        return await plugin.execute(function_name, helper, **json.loads(arguments), chat_id=chat_id)

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
                if function_name in map(lambda spec: spec.get('function', {}).get('name'), plugin.get_spec())
            ),
            None,
        )
