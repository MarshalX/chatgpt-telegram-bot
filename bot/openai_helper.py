from __future__ import annotations

import asyncio
import base64
import datetime
import io
import json
import logging
import os
from collections import Counter
from typing import TYPE_CHECKING, List, Optional, TypedDict, Union

import aiofiles
import httpx
import openai
from openai._utils import async_maybe_transform
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam
from openai.types.images_response import Usage
from plugin_manager import PluginManager
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from utils import is_direct_result

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

# Models can be found here: https://platform.openai.com/docs/models/overview
GPT_4_MODELS = ('gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-turbo-preview')
GPT_4_32K_MODELS = ('gpt-4-32k', 'gpt-4-32k-0314', 'gpt-4-32k-0613')
GPT_4_VISION_MODELS = (
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
GPT_5_MODELS = (
    'gpt-5.1',
    'gpt-5.1-2025-11-13',
    'gpt-5.1-codex',
    'gpt-5.1-codex-mini',
    'gpt-5.1-chat-latest',
    'gpt-5',
    'gpt-5-mini',
    'gpt-5-nano',
    'gpt-5-2025-08-07',
    'gpt-5-mini-2025-08-07',
    'gpt-5-nano-2025-08-07',
    'gpt-5-chat-latest',
)
GPT_SEARCH_MODELS = (
    'gpt-4o-search-preview',
    'gpt-4o-mini-search-preview',
)
GPT_ALL_MODELS = (
    GPT_4_MODELS
    + GPT_4_32K_MODELS
    + GPT_4_VISION_MODELS
    + GPT_4_128K_MODELS
    + GPT_4O_MODELS
    + GPT_41_MODELS
    + GPT_5_MODELS
    + GPT_SEARCH_MODELS
)


def default_max_output_tokens(model: str) -> int:
    """
    Gets the default number of max OUTPUT tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    base = 1024
    if model in GPT_4_MODELS:
        return base * 2
    elif model in GPT_4_32K_MODELS:
        return base * 8
    elif model in GPT_4_128K_MODELS:
        return base * 8
    elif model in GPT_4O_MODELS:
        return base * 16
    elif model in GPT_41_MODELS:
        return base * 32
    elif model in GPT_5_MODELS:
        return base * 32

    return base


def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    # Stable models will be updated to support functions on June 27, 2023
    if model in (
        'gpt-4',
        'gpt-4-32k',
        'gpt-4-1106-preview',
        'gpt-4-0125-preview',
        'gpt-4-turbo-preview',
    ):
        return datetime.date.today() > datetime.date(2023, 6, 27)
    if model in GPT_SEARCH_MODELS:
        return False
    return True


_MODELS_COST = {
    # tuples of input price, cached input price, output price per 1M in $
    # ref: https://openai.com/api/pricing/
    'gpt-4o': (2.5, 1.25, 10),
    'gpt-4o-mini': (0.15, 0.075, 0.6),
    'gpt-4o-search-preview': (2.5, 2.5, 10),
    'gpt-4o-mini-search-preview': (0.15, 2.5, 0.6),
    'gpt-4.1': (2, 0.5, 8),
    'gpt-4.1-mini': (0.4, 0.1, 1.6),
    'gpt-4.1-nano': (0.1, 0.025, 0.4),
    'gpt-image-1': (5, 5, 40, 10),  # text input, cached input price, image output, input image
    'gpt-5': (1.25, 0.125, 10),
    'gpt-5-mini': (0.25, 0.025, 2),
    'gpt-5-nano': (0.05, 0.005, 0.4),
    'gpt-5.1': (1.25, 0.125, 10),
    'gpt-5.1-codex': (1.25, 0.125, 10),
    'gpt-5.1-codex-mini': (0.25, 0.025, 2),
}
_DEFAULT_MODEL_PRICE = (0, 0, 0)


def get_model_cost(model: str, usage: Union[Usage, CompletionUsage]) -> float:
    input_price_per_m, cached_input_price_per_m, output_price_per_m, *extra_prices = _MODELS_COST.get(
        model, _DEFAULT_MODEL_PRICE
    )

    input_price = input_price_per_m / 1_000_000
    output_price = output_price_per_m / 1_000_000
    cached_input_price = cached_input_price_per_m / 1_000_000

    extra_total = 0
    if isinstance(usage, Usage):
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cached_input_tokens = 0

        if usage.input_tokens_details:
            input_tokens = usage.input_tokens_details.text_tokens

            (input_image_price_per_m,) = extra_prices
            input_image_price = input_image_price_per_m / 1_000_000
            input_image_tokens = usage.input_tokens_details.image_tokens
            extra_total = input_image_price * input_image_tokens
    else:
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        cached_input_tokens = 0
        if usage.prompt_tokens_details:
            cached_input_tokens = usage.prompt_tokens_details.cached_tokens
            input_tokens -= cached_input_tokens  # cached tokens are charged at the cached input price

    return (
        (input_price * input_tokens)
        + (cached_input_price * cached_input_tokens)
        + (output_price * output_tokens)
        + extra_total
    )


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
        http_client = httpx.AsyncClient(proxy=config['proxy']) if 'proxy' in config else None
        self.client = openai.AsyncOpenAI(api_key=config['api_key'], http_client=http_client)

        self.db_pool = None

        self.config = config
        self.plugin_manager = plugin_manager

        self.conversations: dict[str:list] = {}  # {chat_id: history}
        self.conversations_costs: dict[str:float] = {}  # {chat_id: cost}
        self.conversations_tokens: dict[str:int] = {}  # {chat_id: tokens}
        self.last_updated: dict[str:datetime] = {}  # {chat_id: last_update_timestamp}
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

    async def get_conversation_stats(self, chat_id: str) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages and tokens used
        """
        if chat_id not in self.conversations:
            await self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), self.get_tokens(chat_id)

    async def get_chat_response(
        self, chat_id: str, query: str, image: Optional[str] = None, user_id: Optional[str] = None
    ) -> tuple[str, int]:
        """
        Gets a full response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :param image: The image to send to the model
        :param user_id: The user ID for tracking
        :return: The answer from the model and the number of tokens used
        """
        plugins_used = []
        response = await self.__common_get_chat_response(chat_id, query, image=image, user_id=user_id)
        if self.config['enable_functions']:
            response, _, plugins_used = await self.__handle_function_call(chat_id, response, user_id=user_id)
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
        plugins_used_counter = Counter(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        plugin_names = [f'{name} x{count}' if count > 1 else name for name, count in plugins_used_counter.items()]
        if self.config['show_usage']:
            cost = get_model_cost(self.config['model'], response.usage)
            self.add_cost(chat_id, cost)
            self.set_tokens(chat_id, response.usage.total_tokens)

            price = get_formatted_price(self.get_cost(chat_id))
            answer += f'\n\n---\nID: {chat_id[-2:]} {price}'

            if show_plugins_used:
                answer += f'\n🔌 {", ".join(plugin_names)}'
        elif show_plugins_used:
            answer += f'\n\n---\n🔌 {", ".join(plugin_names)}'

        return answer, response.usage.total_tokens

    async def get_chat_response_stream(
        self, chat_id: str, query: str, image: Optional[str] = None, user_id: Optional[str] = None
    ):
        """
        Stream response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :param image: The image to send to the model
        :param user_id: The user ID for tracking
        :return: The answer from the model and the number of tokens used, or 'not_finished'
        """
        plugins_used = []
        response = await self.__common_get_chat_response(chat_id, query, stream=True, image=image, user_id=user_id)
        if self.config['enable_functions']:
            response, response_chunks, plugins_used = await self.__handle_function_call(
                chat_id, response, stream=True, user_id=user_id
            )
            if is_direct_result(response):
                yield response, '0'
                return

        answer = ''
        last_chunk = None
        for chunk in response_chunks:
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
        plugins_used_counter = Counter(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        plugin_names = [f'{name} x{count}' if count > 1 else name for name, count in plugins_used_counter.items()]
        if self.config['show_usage']:
            cost = get_model_cost(self.config['model'], usage)
            self.add_cost(chat_id, cost)
            self.set_tokens(chat_id, usage.total_tokens)

            price = get_formatted_price(self.get_cost(chat_id))
            answer += f'\n\n---\nID: {chat_id[-2:]} {price}'

            if show_plugins_used:
                answer += f'\n🔌 {", ".join(plugin_names)}'
        elif show_plugins_used:
            answer += f'\n\n---\n🔌 {", ".join(plugin_names)}'

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
    async def __common_get_chat_response(
        self, chat_id: str, query: str, stream=False, image: Optional[str] = None, user_id: Optional[str] = None
    ):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :param user_id: The user ID for tracking
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        model = self.config['vision_model'] if image else self.config['model']

        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                await self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            async def _add_to_history() -> None:
                if query:
                    await self.__add_to_history(chat_id, role='user', content=query)

                if image:
                    await self.__add_to_history(
                        chat_id,
                        role='user',
                        content=[
                            {
                                'type': 'image_url',
                                'image_url': {'url': image, 'detail': self.config['vision_detail']},
                            },
                        ],
                    )

            await _add_to_history()

            # Summarize the chat history if it's too long to avoid excessive token usage
            exceeded_max_tokens = (
                self.get_tokens(chat_id) + self.config['max_output_tokens'] > self.__max_model_tokens()
            )
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1], user_id)
                    logging.debug(f'Summary: {summary}')
                    await self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    await self.__add_to_history(chat_id, role='assistant', content=summary)
                    await _add_to_history()
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    # FIXME update in DB
                    self.conversations[chat_id] = [self.conversations[chat_id][0]] + self.conversations[chat_id][
                        -self.config['max_history_size'] - 1 :
                    ]

            max_tokens_arg_name = 'max_completion_tokens' if model in GPT_5_MODELS else 'max_tokens'
            common_args = {
                'model': model,
                'messages': self.conversations[chat_id],
                max_tokens_arg_name: self.config['max_output_tokens'],
                'stream': stream,
                'user': user_id,
            }

            if common_args['model'] in GPT_5_MODELS:
                common_args.update(
                    {
                        'reasoning_effort': self.config['reasoning_effort'],
                        'verbosity': self.config['verbosity'],
                    }
                )

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

            if self.config['enable_functions']:
                functions = self.plugin_manager.get_functions_specs()
                if len(functions) > 0:
                    common_args['tools'] = functions
                    common_args['tool_choice'] = 'auto'
                    common_args['parallel_tool_calls'] = True

            return await self.client.chat.completions.create(**common_args)

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            raise Exception(f'⚠️ <i>{localized_text("openai_invalid", bot_language)}.</i> ⚠️\n{str(e)}') from e

        except Exception as e:
            raise Exception(f'⚠️ <i>{localized_text("error", bot_language)}.</i> ⚠️\n{str(e)}') from e

    async def __call_functions_in_parallel(self, chat_id, final_tool_calls, times, cost, plugins_used):
        logging.info(
            f'[FUNC CALL][{times}] Calling functions in parallel: {list(f.function.name for f in final_tool_calls.values())}'
        )

        tasks = []
        for tool_call in final_tool_calls.values():
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            logging.info(f'[FUNC CALL][{times}] Calling "{function_name}" with arguments {arguments}')
            task = asyncio.create_task(self.plugin_manager.call_function(chat_id, function_name, self, arguments))
            tasks.append((tool_call, function_name, task))

        direct_responses = []
        for tool_call, function_name, task in tasks:
            function_response = await task
            function_response_json = json.dumps(function_response, default=str)

            plugins_used.append(function_name)

            price = get_formatted_price(cost)
            logging.info(
                f'[FUNC CALL][{times}] "{function_name}" costed {price} returned {function_response_json[:100]}'
            )

            if is_direct_result(function_response):
                logging.info(f'[FUNC CALL][{times}] "{function_name}" returned a direct result')
                await self.__add_tool_call_to_history(chat_id, tool_call)
                await self.__add_tool_call_result_to_history(
                    chat_id=chat_id,
                    tool_call_id=tool_call.id,
                    tool_name=function_name,
                    result=json.dumps({'result': 'Done, the content has been sent to the user.'}),
                )
                direct_responses.append(function_response)
                continue

            await self.__add_tool_call_to_history(chat_id, tool_call)
            await self.__add_tool_call_result_to_history(
                chat_id=chat_id, tool_call_id=tool_call.id, tool_name=function_name, result=function_response_json
            )

        return direct_responses, plugins_used

    async def __handle_function_call(
        self, chat_id, response, stream=False, times=0, plugins_used=None, user_id: Optional[str] = None
    ):
        if plugins_used is None:
            plugins_used = []

        final_tool_calls = {}
        response_chunks = []

        if stream:
            async for chunk in response:
                response_chunks.append(chunk)
                if not chunk.choices:
                    break

                first_choice = chunk.choices[0]
                if not first_choice.delta.tool_calls:
                    continue

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
            return response, response_chunks, plugins_used

        if stream:
            usage = None
            # this is a chained function call
            # we do not need this response anymore except for finding usage
            async for chunk in response:
                response_chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage
        else:
            usage = response.usage

        cost = 0
        if usage:
            # for chained function calls we process cost here (intermediate calls)
            # otherwise it will be calculated outside in the response handler (final or the only one call)
            cost = get_model_cost(self.config['model'], usage)
            self.add_cost(chat_id, cost)
            self.set_tokens(chat_id, usage.total_tokens)

        direct_responses, plugins_used = await self.__call_functions_in_parallel(
            chat_id, final_tool_calls, times, cost, plugins_used
        )
        if direct_responses:
            return direct_responses, response_chunks, plugins_used

        response = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=self.conversations[chat_id],
            tools=self.plugin_manager.get_functions_specs(),
            tool_choice='auto' if times < self.config['functions_max_consecutive_calls'] else 'none',
            parallel_tool_calls=True,
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
                    f'⚠️ <i>{localized_text("error", bot_language)}.</i> ⚠️\n{localized_text("try_again", bot_language)}.'
                )

            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)

            price = ''
            if response.usage:
                cost = get_model_cost(image_model, response.usage)
                price = get_formatted_price(cost)

            return image_bytes, self.config['image_size'], price
        except Exception as e:
            raise Exception(f'⚠️ <i>{localized_text("error", bot_language)}.</i> ⚠️\n{str(e)}') from e

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
            raise Exception(f'⚠️ <i>{localized_text("error", bot_language)}.</i> ⚠️\n{str(e)}') from e

    async def transcribe(self, filename):
        # FIXME do not use filename; use fileobj instead
        """
        Transcribes the audio file using the Whisper model.
        """
        try:
            async with aiofiles.open(filename, mode='rb') as audio:
                prompt_text = self.config['whisper_prompt']
                result = await self.client.audio.transcriptions.create(
                    model='whisper-1', file=audio, prompt=prompt_text
                )
                return result.text
        except Exception as e:
            logging.exception(e)
            raise Exception(f'⚠️ <i>{localized_text("error", self.config["bot_language"])}.</i> ⚠️\n{str(e)}') from e

    async def get_thread_topic(self, chat_id: str) -> str:
        """
        Gets the topic of the conversation.
        :param chat_id: The chat ID
        :return: The topic of the conversation
        """
        TOPIC_LEN_LIMIT = 30

        if chat_id not in self.conversations or self.is_empty_history(chat_id):
            return ''

        messages = [
            {
                'role': 'system',
                'content': (
                    'You generate conversation titles. '
                    'Return ONLY a title. '
                    'Max 3 words. No punctuation. No explanations.'
                ),
            },
            {'role': 'user', 'content': str(self.conversations[chat_id][1:])},
        ]
        response = await self.client.chat.completions.create(
            model=self.config['model'], messages=messages, temperature=0.4
        )
        topic = response.choices[0].message.content.strip()[:TOPIC_LEN_LIMIT]
        return topic

    async def reset_chat_history(self, chat_id: str, content: Optional[str] = None):
        """
        Resets the conversation history.
        """
        if not content:
            content = self.config['assistant_prompt']

        self.conversations[chat_id] = [{'role': 'system', 'content': content}]
        self.conversations_costs[chat_id] = 0
        self.set_tokens(chat_id, 0)

        await self.init_conv_in_db(chat_id)
        await self.add_conv_in_db(chat_id, 'system', content)

    def is_empty_history(self, chat_id: str) -> bool:
        """
        Checks if the conversation history is empty (only contains the system prompt).
        :param chat_id: The chat ID
        :return: A boolean indicating whether the conversation history is empty
        """
        if chat_id not in self.conversations:
            return True

        return chat_id in self.conversations and len(self.conversations[chat_id]) <= 1

    def __max_age_reached(self, chat_id: str) -> bool:
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
        self, chat_id, tool_call: Union['ChatCompletionMessageToolCall', 'ChoiceDeltaToolCall']
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
                            'arguments': tool_call.function.arguments,
                        },
                    }
                ],
            }
        )
        await self.add_conv_in_db(chat_id, 'assistant', tool_call.function.arguments, tool_call.function.name)

    async def __add_tool_call_result_to_history(self, chat_id, tool_call_id, tool_name: str, result: str):
        """
        Adds a tool call result to the conversation history
        """
        self.conversations[chat_id].append({'role': 'tool', 'tool_call_id': tool_call_id, 'content': result})
        await self.add_conv_in_db(chat_id, 'tool', result, tool_name)

    async def __add_to_history(self, chat_id: str, role: str, content: Union[str, List[dict]]):
        """
        Adds a message to the conversation history.
        :param chat_id: The chat ID
        :param role: The role of the message sender
        :param content: The message content
        """
        self.conversations[chat_id].append({'role': role, 'content': content})

        data = await async_maybe_transform({'message': {'role': role, 'content': content}}, self.MessageDb)
        msg = data['message']

        if isinstance(msg['content'], list):
            msg['content'] = json.dumps(msg['content'])

        # self.conversations[chat_id].append({'role': msg['role'], 'content': msg['content']})
        await self.add_conv_in_db(chat_id, msg['role'], msg['content'])

    def add_cost(self, chat_id: str, cost: float):
        """
        Adds the cost to the conversation.
        """
        self.conversations_costs[chat_id] = cost + self.get_cost(chat_id)

    def get_cost(self, chat_id: str) -> float:
        """
        Gets the cost of the conversation.
        """
        return self.conversations_costs.get(chat_id, 0)

    def set_tokens(self, chat_id, tokens: int):
        """
        Set the tokens of the conversation.
        """
        self.conversations_tokens[chat_id] = tokens

    def get_tokens(self, chat_id: str) -> int:
        """
        Gets the tokens of the conversation.
        """
        return self.conversations_tokens.get(chat_id, 0)

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
        if self.config['model'] in GPT_5_MODELS:
            return base * 97  # ~400k
        raise NotImplementedError(f'Max tokens for model {self.config["model"]} is not implemented yet.')
