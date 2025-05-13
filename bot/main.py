import logging
import os

from dotenv import load_dotenv
from openai_helper import GPT_SEARCH_MODELS, OpenAIHelper, are_functions_available, default_max_output_tokens
from plugin_manager import PluginManager
from telegram_bot import ChatGPTTelegramBot


def read_prompt_from_file(file_path_env_var, default_path, fallback_env_var, default_value) -> str:
    """
    Read prompt from a file, with fallback to environment variable.
    Args:
        file_path_env_var: The environment variable containing the path to the file
        default_path: Default path to use if file_path_env_var is not set
        fallback_env_var: The environment variable to use as fallback
        default_value: Default value to use if neither file nor environment variable exists
    Returns:
        The prompt text
    """
    file_path = os.environ.get(file_path_env_var, default_path)
    if file_path and os.path.isfile(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    logging.info(f'Read prompt from file: {file_path}')
                    return content
        except Exception as e:
            logging.warning(f'Failed to read prompt from file {file_path}: {e}')

    # Fallback to environment variable
    env_value = os.environ.get(fallback_env_var, default_value)
    return env_value


def main() -> None:
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
    )
    # logging.getLogger('openai._base_client').setLevel(logging.DEBUG)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    # Check if the required environment variables are set
    required_values = ['TELEGRAM_BOT_TOKEN', 'OPENAI_API_KEY']
    missing_values = [value for value in required_values if not os.environ.get(value)]
    if len(missing_values) > 0:
        logging.error(f'The following environment values are missing in your .env: {", ".join(missing_values)}')
        exit(1)

    # Setup configurations
    model = os.environ.get('OPENAI_MODEL', 'gpt-4.1-mini')
    functions_available = are_functions_available(model=model)
    max_output_tokens_default = default_max_output_tokens(model=model)

    # Read prompts from files or environment variables
    assistant_prompt = read_prompt_from_file(
        'ASSISTANT_PROMPT_FILE', 'prompts/assistant_prompt.txt', 'ASSISTANT_PROMPT', 'You are a helpful assistant.'
    )
    whisper_prompt = read_prompt_from_file('WHISPER_PROMPT_FILE', 'prompts/whisper_prompt.txt', 'WHISPER_PROMPT', '')

    openai_config = {
        'api_key': os.environ['OPENAI_API_KEY'],
        'show_usage': os.environ.get('SHOW_USAGE', 'true').lower() == 'true',
        'stream': os.environ.get('STREAM', 'true').lower() == 'true',
        'proxy': os.environ.get('PROXY', None) or os.environ.get('OPENAI_PROXY', None),
        'max_history_size': int(os.environ.get('MAX_HISTORY_SIZE', 500)),
        'max_conversation_age_minutes': int(os.environ.get('MAX_CONVERSATION_AGE_MINUTES', 10080)),
        'assistant_prompt': assistant_prompt,
        'max_output_tokens': int(os.environ.get('MAX_OUTPUT_TOKENS', max_output_tokens_default)),
        'n_choices': int(os.environ.get('N_CHOICES', 1)),
        'temperature': float(os.environ.get('TEMPERATURE', 1.0)),
        'image_model': os.environ.get('IMAGE_MODEL', 'dall-e-3'),
        'image_quality': os.environ.get('IMAGE_QUALITY', 'standard'),
        'image_style': os.environ.get('IMAGE_STYLE', 'vivid'),
        'image_size': os.environ.get('IMAGE_SIZE', '1024x1024'),
        'model': model,
        'enable_functions': os.environ.get('ENABLE_FUNCTIONS', str(functions_available)).lower() == 'true',
        'functions_max_consecutive_calls': int(os.environ.get('FUNCTIONS_MAX_CONSECUTIVE_CALLS', 25)),
        'presence_penalty': float(os.environ.get('PRESENCE_PENALTY', 0.0)),
        'frequency_penalty': float(os.environ.get('FREQUENCY_PENALTY', 0.0)),
        'bot_language': os.environ.get('BOT_LANGUAGE', 'en'),
        'show_plugins_used': os.environ.get('SHOW_PLUGINS_USED', 'false').lower() == 'true',
        'whisper_prompt': whisper_prompt,
        'vision_model': os.environ.get('VISION_MODEL', 'gpt-4.1-mini'),
        'vision_detail': os.environ.get('VISION_DETAIL', 'auto'),
        'tts_model': os.environ.get('TTS_MODEL', 'tts-1'),
        'tts_voice': os.environ.get('TTS_VOICE', 'nova'),
        'allowed_chat_ids_to_track': set(os.environ.get('ALLOWED_CHAT_IDS_TO_TRACK', '').split(',')),
        'web_search_context_size': os.environ.get('WEB_SEARCH_CONTEXT_SIZE', 'medium'),
        'web_search_support_annotations': os.environ.get('WEB_SEARCH_SUPPORT_ANNOTATIONS', 'true').lower() == 'true',
    }

    if openai_config['enable_functions'] and not functions_available:
        logging.error(
            f'ENABLE_FUNCTIONS is set to true, but the model {model} does not support it. '
            'Please set ENABLE_FUNCTIONS to false or use a model that supports it.'
        )
        exit(1)

    telegram_config = {
        'token': os.environ['TELEGRAM_BOT_TOKEN'],
        'admin_user_ids': os.environ.get('ADMIN_USER_IDS', '-'),
        'img_gen_access_user_ids': os.environ.get('IMG_GEN_ACCESS_USER_IDS', '-'),
        'allowed_user_ids': os.environ.get('ALLOWED_TELEGRAM_USER_IDS', '*'),
        'enable_quoting': os.environ.get('ENABLE_QUOTING', 'true').lower() == 'true',
        'enable_image_generation': os.environ.get('ENABLE_IMAGE_GENERATION', 'true').lower() == 'true',
        'enable_transcription': os.environ.get('ENABLE_TRANSCRIPTION', 'true').lower() == 'true',
        'enable_vision': os.environ.get('ENABLE_VISION', 'true').lower() == 'true',
        'enable_tts_generation': os.environ.get('ENABLE_TTS_GENERATION', 'true').lower() == 'true',
        'budget_period': os.environ.get('BUDGET_PERIOD', 'monthly').lower(),
        'user_budgets': os.environ.get('USER_BUDGETS', os.environ.get('MONTHLY_USER_BUDGETS', '*')),
        'guest_budget': float(os.environ.get('GUEST_BUDGET', os.environ.get('MONTHLY_GUEST_BUDGET', '100.0'))),
        'stream': os.environ.get('STREAM', 'true').lower() == 'true',
        'proxy': os.environ.get('PROXY', None) or os.environ.get('TELEGRAM_PROXY', None),
        'voice_reply_transcript': os.environ.get('VOICE_REPLY_WITH_TRANSCRIPT_ONLY', 'false').lower() == 'true',
        'voice_reply_prompts': os.environ.get('VOICE_REPLY_PROMPTS', '').split(';'),
        'ignore_group_transcriptions': os.environ.get('IGNORE_GROUP_TRANSCRIPTIONS', 'true').lower() == 'true',
        'ignore_group_vision': os.environ.get('IGNORE_GROUP_VISION', 'true').lower() == 'true',
        'group_trigger_keyword': os.environ.get('GROUP_TRIGGER_KEYWORD', ''),
        'token_price': float(os.environ.get('TOKEN_PRICE', 0.002)),
        'image_prices': [float(i) for i in os.environ.get('IMAGE_PRICES', '0.016,0.018,0.02').split(',')],
        'vision_token_price': float(os.environ.get('VISION_TOKEN_PRICE', '0.01')),
        'image_receive_mode': os.environ.get('IMAGE_FORMAT', 'photo'),
        'tts_model': os.environ.get('TTS_MODEL', 'tts-1-hd'),
        'tts_prices': [float(i) for i in os.environ.get('TTS_PRICES', '0.015,0.030').split(',')],
        'transcription_price': float(os.environ.get('TRANSCRIPTION_PRICE', 0.006)),
        'bot_language': os.environ.get('BOT_LANGUAGE', 'en'),
        'database_url': os.environ.get('DATABASE_URL_TO_DROP_ALL_TABLES'),
        'enable_rate_limit': os.environ.get('ENABLE_RATE_LIMIT', 'true').lower() == 'true',
        'group_rate_limit': int(os.environ.get('GROUP_RATE_LIMIT', '20')),
        'private_rate_limit': float(os.environ.get('PRIVATE_RATE_LIMIT', '1.0')),
        'max_update_frequency': float(os.environ.get('MAX_UPDATE_FREQUENCY', '0.5')),
    }

    if model in GPT_SEARCH_MODELS and openai_config['web_search_support_annotations']:
        # annotations are not supported in streaming mode
        openai_config['stream'] = False
        telegram_config['stream'] = False

    plugin_config = {'plugins': os.environ.get('PLUGINS', '').split(',')}

    # Setup and run ChatGPT and Telegram bot
    plugin_manager = PluginManager(config=plugin_config)
    openai_helper = OpenAIHelper(config=openai_config, plugin_manager=plugin_manager)
    telegram_bot = ChatGPTTelegramBot(config=telegram_config, openai=openai_helper)
    telegram_bot.run()


if __name__ == '__main__':
    main()
