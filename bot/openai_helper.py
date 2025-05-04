from __future__ import annotations

import asyncio
import base64
import datetime
import io
import json
import logging
import os
from typing import TYPE_CHECKING, Optional, TypedDict, Union

import httpx
import openai
import tiktoken
from openai._utils import async_maybe_transform
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam
from openai.types.images_response import Usage
from PIL import Image
from plugin_manager import PluginManager
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from utils import decode_image, encode_image, is_direct_result

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

# Models can be found here: https://platform.openai.com/docs/models/overview
# Models gpt-3.5-turbo-0613 and gpt-3.5-turbo-16k-0613 will be deprecated on June 13, 2024
GPT_3_MODELS = ('gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613')
GPT_3_16K_MODELS = (
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-16k-0613',
    'gpt-3.5-turbo-1106',
    'gpt-3.5-turbo-0125',
)
GPT_4_MODELS = ('gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-turbo-preview')
GPT_4_32K_MODELS = ('gpt-4-32k', 'gpt-4-32k-0314', 'gpt-4-32k-0613')
GPT_4_VISION_MODELS = (
    'gpt-4-vision-preview',
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
)
GPT_4_128K_MODELS = (
    'gpt-4-1106-preview',
    'gpt-4-0125-preview',
    'gpt-4-turbo-preview',
    'gpt-4-turbo',
    'gpt-4-turbo-2024-04-09',
)
GPT_4O_MODELS = (
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4o-2024-08-06',
    'gpt-4o-2024-05-13',
    'gpt-4o-mini-2024-07-18',
    'gpt-4o-search-preview',
    'gpt-4o-mini-search-preview',
)
GPT_41_MODELS = (
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'gpt-4.1-2025-04-14',
    'gpt-4.1-mini-2025-04-14',
    'gpt-4.1-nano-2025-04-14',
)
GPT_SEARCH_MODELS = (
    'gpt-4o-search-preview',
    'gpt-4o-mini-search-preview',
)
GPT_ALL_MODELS = (
    GPT_3_MODELS
    + GPT_3_16K_MODELS
    + GPT_4_MODELS
    + GPT_4_32K_MODELS
    + GPT_4_VISION_MODELS
    + GPT_4_128K_MODELS
    + GPT_4O_MODELS
    + GPT_41_MODELS
)


def default_max_output_tokens(model: str) -> int:
    """
    Gets the default number of max OUTPUT tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    base = 1024
    if model in GPT_3_MODELS:
        return base
    elif model in GPT_4_MODELS:
        return base * 2
    elif model in GPT_3_16K_MODELS:
        return base * 4
    elif model in GPT_4_32K_MODELS:
        return base * 8
    elif model in GPT_4_128K_MODELS:
        return base * 8
    elif model in GPT_4O_MODELS:
        return base * 16
    elif model in GPT_41_MODELS:
        return base * 32


def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    # Deprecated models
    if model in ('gpt-3.5-turbo-0301', 'gpt-4-0314', 'gpt-4-32k-0314'):
        return False
    # Stable models will be updated to support functions on June 27, 2023
    if model in (
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-1106',
        'gpt-4',
        'gpt-4-32k',
        'gpt-4-1106-preview',
        'gpt-4-0125-preview',
        'gpt-4-turbo-preview',
    ):
        return datetime.date.today() > datetime.date(2023, 6, 27)
    # Models gpt-3.5-turbo-0613 and gpt-3.5-turbo-16k-0613 will be deprecated on June 13, 2024
    if model in ('gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613'):
        return datetime.date.today() < datetime.date(2024, 6, 13)
    if model == 'gpt-4-vision-preview':
        return False
    if model in GPT_SEARCH_MODELS:
        return False
    return True


_MODELS_COST = {
    # tuples of input price, output price per 1M in $
    # ref: https://openai.com/api/pricing/
    'gpt-4o-2024-05-13': (5, 15),
    'gpt-4o-2024-08-06': (2.5, 10),
    'gpt-4o': (2.5, 10),
    'gpt-4o-mini': (0.15, 0.6),
    'gpt-4o-mini-2024-07-18': (0.15, 0.6),
    'gpt-4o-search-preview': (2.5, 10),
    'gpt-4o-mini-search-preview': (0.15, 0.6),
    'gpt-4.1': (2, 8),
    'gpt-4.1-mini': (0.4, 1.6),
    'gpt-4.1-nano': (0.1, 0.4),
    'gpt-image-1': (5, 40, 10),  # text input, image output, input image
}
_DEFAULT_MODEL_PRICE = (0, 0)


def get_model_cost(model: str, usage: Union[Usage, CompletionUsage]) -> float:
    input_price_per_m, output_price_per_m, *extra_prices = _MODELS_COST.get(model, _DEFAULT_MODEL_PRICE)

    input_price = input_price_per_m / 1_000_000
    output_price = output_price_per_m / 1_000_000

    extra_total = 0
    if isinstance(usage, Usage):
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        if usage.input_tokens_details:
            input_tokens = usage.input_tokens_details.text_tokens

            (input_image_price_per_m,) = extra_prices
            input_image_price = input_image_price_per_m / 1_000_000
            input_image_tokens = usage.input_tokens_details.image_tokens
            extra_total = input_image_price * input_image_tokens
    else:
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

    return (input_price * input_tokens) + (output_price * output_tokens) + extra_total


def get_formatted_price(cost: float) -> str:
    return f'¢{cost * 100:.2f}' if cost >= 1e-4 else ''


# Load translations
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)


def localized_text(key, bot_language):
    """
    Return translated text for a key in specified bot_language.
    Keys and translations can be found in the translations.json.
    """
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation available for bot_language code '{bot_language}' and key '{key}'")
        # Fallback to English if the translation is not available
        if key in translations['en']:
            return translations['en'][key]
        else:
            logging.warning(f"No english definition found for key '{key}' in translations.json")
            # return key as text
            return key


class OpenAIHelper:
    """
    ChatGPT helper class.
    """

    def __init__(self, config: dict, plugin_manager: PluginManager):
        """
        Initializes the OpenAI helper class with the given configuration.
        :param config: A dictionary containing the GPT configuration
        :param plugin_manager: The plugin manager
        """
        http_client = httpx.AsyncClient(proxies=config['proxy']) if 'proxy' in config else None
        self.client = openai.AsyncOpenAI(api_key=config['api_key'], http_client=http_client)

        self.db_pool = None

        self.config = config
        self.plugin_manager = plugin_manager
        self.conversations: dict[int:list] = {}  # {chat_id: history}
        self.conversations_costs: dict[int:float] = {}  # {chat_id: cost}
        self.conversations_vision: dict[int:bool] = {}  # {chat_id: is_vision}
        self.last_updated: dict[int:datetime] = {}  # {chat_id: last_update_timestamp}
        self.conversation_locks: dict[str : asyncio.Lock] = {}  # {chat_id: lock}

    class MessageDb(TypedDict, total=False):
        message: ChatCompletionMessageParam

    async def init_conv_in_db(self, chat_id: str) -> None:
        if not self.db_pool:
            return

        if chat_id.split('_')[0] not in self.config['allowed_chat_ids_to_track']:
            logging.debug(f'Chat ID {chat_id} is not allowed to be tracked')
            return

        async with self.db_pool.acquire() as conn:
            await conn.execute(f'DROP TABLE IF EXISTS "{chat_id}"')  # взлом жопы

            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{chat_id}" (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL,
                    name TEXT,
                    content TEXT NOT NULL
                )
                """
            )

        logging.debug(f'Chat ID {chat_id} is now being tracked')

    async def add_conv_in_db(self, chat_id: str, role: str, content: str, name: Optional[str] = None) -> None:
        if not self.db_pool:
            return

        if chat_id.split('_')[0] not in self.config['allowed_chat_ids_to_track']:
            logging.debug(f'Chat ID {chat_id} is not allowed to be tracked')
            return

        async with self.db_pool.acquire() as conn:
            prepared = await conn.prepare(f"""
                INSERT INTO "{chat_id}" (role, name, content)
                VALUES ($1, $2, $3)
                """)
            await prepared.fetchval(role, name, content)

        logging.debug(f'Added message to chat ID {chat_id} history')

    async def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages and tokens used
        """
        if chat_id not in self.conversations:
            await self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), self.__count_tokens(self.conversations[chat_id])

    async def get_chat_response(self, chat_id: str, query: str, user_id: Optional[str] = None) -> tuple[str, int]:
        """
        Gets a full response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :param user_id: The user ID for tracking
        :return: The answer from the model and the number of tokens used
        """
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query, user_id=user_id)
        if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
            response, plugins_used = await self.__handle_function_call(chat_id, response, user_id=user_id)
            if is_direct_result(response):
                return response, 0

        answer = ''

        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = choice.message.content.strip()
                if index == 0:
                    await self.__add_to_history(chat_id, role='assistant', content=content)
                answer += f'{index + 1}\u20e3\n'
                answer += content
                answer += '\n\n'
        else:
            message = response.choices[0].message
            answer = message.content.strip()

            if self.config['web_search_support_annotations'] and message.annotations:
                citations = []
                offset = 0  # Keep track of how much we've shifted the text
                for i, annotation in enumerate(message.annotations):
                    start = annotation.url_citation.start_index + offset
                    end = annotation.url_citation.end_index + offset

                    # Insert citation reference number
                    citation_text = f'\[{i}]'
                    answer = answer[:start] + citation_text + answer[end:]

                    # Update offset for next citation
                    offset += len(citation_text) - (end - start)

                    citations.append(f'- \[{i}] [{annotation.url_citation.title}]({annotation.url_citation.url})')
                if citations:
                    answer += '\n\n🌐 References:\n' + '\n'.join(citations)

            await self.__add_to_history(chat_id, role='assistant', content=answer)

        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            cost = get_model_cost(self.config['model'], response.usage)
            self.add_cost(chat_id, cost)

            price = get_formatted_price(self.get_cost(chat_id))
            answer += f'\n\n---\nID: {chat_id[-2:]} {price}'

            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        return answer, response.usage.total_tokens

    async def get_chat_response_stream(self, chat_id: str, query: str, user_id: Optional[str] = None):
        """
        Stream response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used, or 'not_finished'
        """
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query, stream=True, user_id=user_id)
        if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
            response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True, user_id=user_id)
            if is_direct_result(response):
                yield response, '0'
                return

        answer = ''
        last_chunk = None
        async for chunk in response:
            last_chunk = chunk
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'

        answer = answer.strip()
        await self.__add_to_history(chat_id, role='assistant', content=answer)

        usage = last_chunk.usage

        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            cost = get_model_cost(self.config['model'], usage)
            self.add_cost(chat_id, cost)

            price = get_formatted_price(self.get_cost(chat_id))
            answer += f'\n\n---\nID: {chat_id[-2:]} {price}'

            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        yield answer, usage.total_tokens

    def get_conversation_lock(self, chat_id: str) -> asyncio.Lock:
        """
        Gets or creates a lock for the given chat ID.
        :param chat_id: The chat ID
        :return: The lock for the conversation
        """
        if chat_id not in self.conversation_locks:
            self.conversation_locks[chat_id] = asyncio.Lock()
        return self.conversation_locks[chat_id]

    @retry(
        reraise=True,
        retry=retry_if_exception_type(BaseException),
        wait=wait_exponential_jitter(),
        stop=stop_after_attempt(5),
    )
    async def __common_get_chat_response(self, chat_id: str, query: str, stream=False, user_id: Optional[str] = None):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :param user_id: The user ID for tracking
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                await self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            await self.__add_to_history(chat_id, role='user', content=query)

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = token_count + self.config['max_output_tokens'] > self.__max_model_tokens()
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1], user_id)
                    logging.debug(f'Summary: {summary}')
                    await self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    await self.__add_to_history(chat_id, role='assistant', content=summary)
                    await self.__add_to_history(chat_id, role='user', content=query)
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    # FIXME update in DB
                    self.conversations[chat_id] = [self.conversations[chat_id][0]] + self.conversations[chat_id][
                        -self.config['max_history_size'] - 1 :
                    ]

            common_args = {
                'model': self.config['model']
                if not self.conversations_vision[chat_id]
                else self.config['vision_model'],
                'messages': self.conversations[chat_id],
                'max_tokens': self.config['max_output_tokens'],
                'stream': stream,
                'user': user_id,
            }

            if common_args['model'] not in GPT_SEARCH_MODELS:
                common_args.update(
                    {
                        'n': self.config['n_choices'],
                        'temperature': self.config['temperature'],
                        'presence_penalty': self.config['presence_penalty'],
                        'frequency_penalty': self.config['frequency_penalty'],
                    }
                )

            if stream:
                common_args['stream_options'] = {'include_usage': True}

            if common_args['model'] in GPT_SEARCH_MODELS:
                common_args['web_search_options'] = {
                    'search_context_size': self.config['web_search_context_size'],
                }

            if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
                functions = self.plugin_manager.get_functions_specs()
                if len(functions) > 0:
                    common_args['tools'] = functions
                    common_args['tool_choice'] = 'auto'
                    common_args['parallel_tool_calls'] = False

            return await self.client.chat.completions.create(**common_args)

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e

        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def __handle_function_call(
        self, chat_id, response, stream=False, times=0, plugins_used=(), user_id: Optional[str] = None
    ):
        # FIXME misses correct count of tokens on chained function calls
        # TODO add support of parallel_tool_calls
        final_tool_calls = {}

        if stream:
            async for chunk in response:
                if not chunk.choices:
                    break

                first_choice = chunk.choices[0]
                if not first_choice.delta.tool_calls:
                    break

                for tool_call in first_choice.delta.tool_calls:
                    index = tool_call.index

                    if index not in final_tool_calls:
                        final_tool_calls[index] = tool_call

                    final_tool_calls[index].function.arguments += tool_call.function.arguments

                if first_choice.finish_reason == 'tool_calls':
                    break
        else:
            tool_calls = (
                response.choices[0].message.tool_calls
                if response.choices and response.choices[0].message.tool_calls
                else []
            )
            for index, tool_call in enumerate(tool_calls):
                final_tool_calls[index] = tool_call

        if not final_tool_calls:
            return response, plugins_used

        if len(final_tool_calls) > 1:
            logging.warning(
                f'Got multiple tool calls: {final_tool_calls}. Which is not supported yet. Only the first one will be used.'
            )

        tool_call = final_tool_calls[0]
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        await self.__add_tool_call_to_history(chat_id, tool_call, arguments)

        logging.info(f'Calling function {function_name} with arguments {arguments}')
        function_response = await self.plugin_manager.call_function(function_name, self, arguments)

        if function_name not in plugins_used:
            plugins_used += (function_name,)

        if is_direct_result(function_response):
            await self.__add_tool_call_result_to_history(
                chat_id=chat_id,
                tool_call_id=tool_call.id,
                tool_name=function_name,
                result=json.dumps({'result': 'Done, the content has been sent to the user.'}),
            )
            return function_response, plugins_used

        await self.__add_tool_call_result_to_history(
            chat_id=chat_id,
            tool_call_id=tool_call.id,
            tool_name=function_name,
            result=json.dumps(function_response, default=str),
        )

        response = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=self.conversations[chat_id],
            tools=self.plugin_manager.get_functions_specs(),
            tool_choice='auto' if times < self.config['functions_max_consecutive_calls'] else 'none',
            parallel_tool_calls=False,
            stream=stream,
            stream_options={'include_usage': True} if stream else None,
            user=user_id,
        )
        return await self.__handle_function_call(chat_id, response, stream, times + 1, plugins_used, user_id)

    async def generate_image(
        self,
        prompt: str,
        image_to_edit: Optional[io.BytesIO] = None,
        quality: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> tuple[bytes, str, str]:
        """
        Generates an image from the given prompt using DALL·E or GPT model.
        :param prompt: The prompt to send to the model
        :param image_to_edit: The image to edit
        :param quality: The quality of the image
        :param user_id: The user ID for tracking
        :return: The image URL and the image size
        """
        bot_language = self.config['bot_language']
        image_model = self.config['image_model']

        gpt_image_quality_to_dall_e_quality = {
            'low': 'standard',
            'medium': 'standard',
            'high': 'hd',
        }

        generate_kwargs = {
            'prompt': prompt,
            'model': image_model,
            'quality': quality or self.config['image_quality'],
            'size': self.config['image_size'],
            'user': user_id,
        }

        if image_model.startswith('dall'):
            generate_kwargs.update(
                {
                    'quality': gpt_image_quality_to_dall_e_quality.get(quality, quality),
                    'style': self.config.get('image_style'),
                    'response_format': 'b64_json',
                }
            )
        else:  # GPT Image
            if image_to_edit:
                generate_kwargs['image'] = ('image_to_edit.jpeg', image_to_edit, 'image/jpeg')
            else:
                generate_kwargs.update(
                    {
                        'moderation': 'low',  # TODO: move to config
                    }
                )

        method = self.client.images.edit if image_to_edit else self.client.images.generate

        try:
            response = await method(**generate_kwargs)

            if len(response.data) == 0:
                logging.error(f'No response from GPT: {str(response)}')
                raise Exception(
                    f"⚠️ _{localized_text('error', bot_language)}._ " f"⚠️\n{localized_text('try_again', bot_language)}."
                )

            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)

            price = ''
            if response.usage:
                cost = get_model_cost(image_model, response.usage)
                price = get_formatted_price(cost)

            return image_bytes, self.config['image_size'], price
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def generate_speech(self, text: str) -> tuple[io.BytesIO, int]:
        """
        Generates an audio from the given text using a TTS model.
        :param text: The text to send to the model
        :return: The audio in bytes and the text size
        """
        bot_language = self.config['bot_language']
        try:
            response = await self.client.audio.speech.create(
                model=self.config['tts_model'],
                voice=self.config['tts_voice'],
                input=text,
                response_format='opus',
            )

            temp_file = io.BytesIO()
            temp_file.write(response.read())
            temp_file.seek(0)
            return temp_file, len(text)
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def transcribe(self, filename):
        # FIXME do not use filename; use fileobj instead
        """
        Transcribes the audio file using the Whisper model.
        """
        try:
            with open(filename, 'rb') as audio:  # noqa: ASYNC101
                prompt_text = self.config['whisper_prompt']
                result = await self.client.audio.transcriptions.create(
                    model='whisper-1', file=audio, prompt=prompt_text
                )
                return result.text
        except Exception as e:
            logging.exception(e)
            raise Exception(f"⚠️ _{localized_text('error', self.config['bot_language'])}._ ⚠️\n{str(e)}") from e

    @retry(
        reraise=True,
        retry=retry_if_exception_type(BaseException),
        wait=wait_exponential_jitter(),
        stop=stop_after_attempt(5),
    )
    async def __common_get_chat_response_vision(
        self, chat_id: int, content: list, stream=False, user_id: Optional[str] = None
    ):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                await self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            if self.config['enable_vision_follow_up_questions']:
                self.conversations_vision[chat_id] = True
                await self.__add_to_history(chat_id, role='user', content=content)
            else:
                query = None

                for message in content:
                    if message['type'] == 'text':
                        query = message['text']
                        break

                if query:
                    await self.__add_to_history(chat_id, role='user', content=query)

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = token_count + self.config['max_output_tokens'] > self.__max_model_tokens()
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    last = self.conversations[chat_id][-1]
                    summary = await self.__summarise(self.conversations[chat_id][:-1], user_id)
                    logging.debug(f'Summary: {summary}')
                    await self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    await self.__add_to_history(chat_id, role='assistant', content=summary)
                    await self.__add_to_history(chat_id, role=last['role'], content=last['content'])
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = [self.conversations[chat_id][0]] + self.conversations[chat_id][
                        -self.config['max_history_size'] - 1 :
                    ]

            message = {'role': 'user', 'content': content}

            common_args = {
                'model': self.config['vision_model'],
                'messages': self.conversations[chat_id][:-1] + [message],
                'temperature': self.config['temperature'],
                'n': 1,  # several choices is not implemented yet
                'max_tokens': self.config['vision_max_output_tokens'],
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream,
                'user': user_id,
            }

            if stream:
                common_args['stream_options'] = {'include_usage': True}

            # vision model does not yet support functions

            # if self.config['enable_functions']:
            #     functions = self.plugin_manager.get_functions_specs()
            #     if len(functions) > 0:
            #         common_args['functions'] = self.plugin_manager.get_functions_specs()
            #         common_args['function_call'] = 'auto'

            return await self.client.chat.completions.create(**common_args)

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e

        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def interpret_image(self, chat_id, fileobj, prompt=None, user_id: Optional[str] = None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [
            {'type': 'text', 'text': prompt},
            {
                'type': 'image_url',
                'image_url': {'url': image, 'detail': self.config['vision_detail']},
            },
        ]

        response = await self.__common_get_chat_response_vision(chat_id, content, user_id=user_id)

        # functions are not available for this model

        # if self.config['enable_functions']:
        #     response, plugins_used = await self.__handle_function_call(chat_id, response)
        #     if is_direct_result(response):
        #         return response, '0'

        answer = ''

        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = choice.message.content.strip()
                if index == 0:
                    await self.__add_to_history(chat_id, role='assistant', content=content)
                answer += f'{index + 1}\u20e3\n'
                answer += content
                answer += '\n\n'
        else:
            answer = response.choices[0].message.content.strip()
            await self.__add_to_history(chat_id, role='assistant', content=answer)

        bot_language = self.config['bot_language']
        # Plugins are not enabled either
        # show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        # plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += (
                "\n\n---\n"
                f"ID: {chat_id[-2:]} 💰 {str(response.usage.total_tokens)} {localized_text('stats_tokens', bot_language)}"
                f" ({str(response.usage.prompt_tokens)} {localized_text('prompt', bot_language)},"
                f" {str(response.usage.completion_tokens)} {localized_text('completion', bot_language)})"
            )
            # if show_plugins_used:
            #     answer += f"\n🔌 {', '.join(plugin_names)}"
        # elif show_plugins_used:
        #     answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        return answer, response.usage.total_tokens

    async def interpret_image_stream(self, chat_id, fileobj, prompt=None, user_id: Optional[str] = None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [
            {'type': 'text', 'text': prompt},
            {
                'type': 'image_url',
                'image_url': {'url': image, 'detail': self.config['vision_detail']},
            },
        ]

        response = await self.__common_get_chat_response_vision(chat_id, content, stream=True, user_id=user_id)

        # if self.config['enable_functions']:
        #     response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True)
        #     if is_direct_result(response):
        #         yield response, '0'
        #         return

        answer = ''
        last_chunk = None
        async for chunk in response:
            last_chunk = chunk
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'

        usage = last_chunk.usage
        answer = answer.strip()
        await self.__add_to_history(chat_id, role='assistant', content=answer)
        # tokens_used = str(self.__count_tokens(self.conversations[chat_id]))
        tokens_used = usage.total_tokens

        price = get_formatted_price(get_model_cost(self.config['model'], usage))

        # show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        # plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += f'\n\n---\nID: {chat_id[-2:]} {price}'
        #     if show_plugins_used:
        #         answer += f"\n🔌 {', '.join(plugin_names)}"
        # elif show_plugins_used:
        #     answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        yield answer, tokens_used

    async def reset_chat_history(self, chat_id, content=''):
        """
        Resets the conversation history.
        """
        if content == '':
            content = self.config['assistant_prompt']
        self.conversations[chat_id] = [{'role': 'system', 'content': content}]
        self.conversations_vision[chat_id] = False
        self.conversations_costs[chat_id] = 0

        await self.init_conv_in_db(chat_id)
        await self.add_conv_in_db(chat_id, 'system', content)

    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        :param chat_id: The chat ID
        :return: A boolean indicating whether the maximum conversation age has been reached
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = self.config['max_conversation_age_minutes']
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    async def __add_tool_call_to_history(
        self, chat_id, tool_call: Union['ChatCompletionMessageToolCall', 'ChoiceDeltaToolCall'], arguments: str
    ):
        """
        Adds a tool call to the conversation history
        """
        self.conversations[chat_id].append(
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'id': tool_call.id,
                        'type': 'function',
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': arguments,
                        },
                    }
                ],
            }
        )
        await self.add_conv_in_db(chat_id, 'assistant', arguments, tool_call.function.name)

    async def __add_tool_call_result_to_history(self, chat_id, tool_call_id, tool_name: str, result: str):
        """
        Adds a tool call result to the conversation history
        """
        self.conversations[chat_id].append({'role': 'tool', 'tool_call_id': tool_call_id, 'content': result})
        await self.add_conv_in_db(chat_id, 'tool', result, tool_name)

    async def __add_to_history(self, chat_id, role, content):
        """
        Adds a message to the conversation history.
        :param chat_id: The chat ID
        :param role: The role of the message sender
        :param content: The message content
        """
        self.conversations[chat_id].append({'role': role, 'content': content})

        data = await async_maybe_transform({'message': {'role': role, 'content': content}}, self.MessageDb)
        msg = data['message']

        if not isinstance(msg['content'], str):
            msg['content'] = json.dumps(msg['content'])

        await self.add_conv_in_db(chat_id, msg['role'], msg['content'])

    def add_cost(self, chat_id, cost):
        """
        Adds the cost to the conversation.
        """
        self.conversations_costs[chat_id] = cost + self.conversations_costs.get(chat_id, 0)

    def get_cost(self, chat_id):
        """
        Gets the cost of the conversation.
        """
        return self.conversations_costs.get(chat_id, 0)

    async def __summarise(self, conversation, user_id: Optional[str] = None) -> str:
        """
        Summarises the conversation history.
        :param conversation: The conversation history
        :return: The summary
        """
        messages = [
            {
                'role': 'assistant',
                'content': 'Summarize this conversation in 700 characters or less',
            },
            {'role': 'user', 'content': str(conversation)},
        ]
        response = await self.client.chat.completions.create(
            model=self.config['model'], messages=messages, temperature=0.4, user=user_id
        )
        return response.choices[0].message.content

    def __max_model_tokens(self):
        base = 4096
        if self.config['model'] in GPT_3_MODELS:
            return base
        if self.config['model'] in GPT_3_16K_MODELS:
            return base * 4
        if self.config['model'] in GPT_4_MODELS:
            return base * 2
        if self.config['model'] in GPT_4_32K_MODELS:
            return base * 8
        if self.config['model'] in GPT_4_128K_MODELS:
            return base * 31
        if self.config['model'] in GPT_4O_MODELS:
            return base * 31  # 128K
        if self.config['model'] in GPT_41_MODELS:
            return base * 240  # 1M
        raise NotImplementedError(f"Max tokens for model {self.config['model']} is not implemented yet.")

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def __count_tokens(self, messages) -> int:
        """
        Counts the number of tokens required to send the given messages.
        :param messages: the messages to send
        :return: the number of tokens required
        """
        model = self.config['model']
        try:
            if model.startswith('gpt-4.1'):
                # tiktoken is not updated to support 4.1 yet
                # remove when tiktoken will support it
                encoding = tiktoken.get_encoding('o200k_base')
            else:
                encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding('gpt-3.5-turbo')

        if model in GPT_3_MODELS + GPT_3_16K_MODELS:
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model in GPT_4_MODELS + GPT_4_32K_MODELS + GPT_4_128K_MODELS + GPT_4O_MODELS + GPT_41_MODELS:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == 'content':
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
                    else:
                        for message1 in value:
                            if message1['type'] == 'image_url':
                                image = decode_image(message1['image_url']['url'])
                                num_tokens += self.__count_tokens_vision(image)
                            else:
                                num_tokens += len(encoding.encode(message1['text']))
                else:
                    num_tokens += len(encoding.encode(json.dumps(value, default=str)))
                    if key == 'name':
                        num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    # no longer needed

    def __count_tokens_vision(self, image_bytes: bytes) -> int:
        """
        Counts the number of tokens for interpreting an image.
        :param image_bytes: image to interpret
        :return: the number of tokens required
        """
        image_file = io.BytesIO(image_bytes)
        image = Image.open(image_file)
        model = self.config['vision_model']
        if model not in GPT_4_VISION_MODELS:
            raise NotImplementedError(f"""count_tokens_vision() is not implemented for model {model}.""")

        w, h = image.size
        if w > h:
            w, h = h, w
        # this computation follows https://platform.openai.com/docs/guides/vision and https://openai.com/pricing#gpt-4-turbo
        base_tokens = 85
        detail = self.config['vision_detail']
        if detail == 'low':
            return base_tokens
        elif detail == 'high' or detail == 'auto':  # assuming worst cost for auto
            f = max(w / 768, h / 2048)
            if f > 1:
                w, h = int(w / f), int(h / f)
            tw, th = (w + 511) // 512, (h + 511) // 512
            tiles = tw * th
            num_tokens = base_tokens + tiles * 170
            return num_tokens
        else:
            raise NotImplementedError(f"""unknown parameter detail={detail} for model {model}.""")
