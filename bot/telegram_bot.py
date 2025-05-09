from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4
from venv import logger

import asyncpg
from decorators import with_conversation_lock
from openai_helper import OpenAIHelper, localized_text
from PIL import Image
from pydub import AudioSegment
from telegram import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputMediaDocument,
    InputMediaPhoto,
    InputTextMessageContent,
    Message,
    Update,
    constants,
)
from telegram.error import BadRequest, RetryAfter, TimedOut
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    ChosenInlineResultHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    MessageReactionHandler,
    filters,
)
from usage_tracker import UsageTracker
from utils import (
    add_chat_request_to_usage_tracker,
    edit_message_with_retry,
    error_handler,
    get_forum_thread_id,
    get_remaining_budget,
    get_reply_to_message_id,
    get_stream_cutoff_values,
    handle_direct_result,
    has_image_gen_permission,
    is_allowed,
    is_direct_result,
    is_group_chat,
    is_private_chat,
    is_within_budget,
    message_text,
    split_into_chunks,
    wrap_with_indicator,
)


class RateLimiter:
    """
    Class to handle rate limiting for Telegram API
    """

    def __init__(self, config):
        # Store the configuration
        self.config = config
        self.enabled = config.get('enable_rate_limit', True)
        self.group_limit = config.get('group_rate_limit', 20)  # Messages per minute for groups
        self.private_limit = config.get('private_rate_limit', 1.0)  # Seconds between messages
        self.max_update_frequency = config.get('max_update_frequency', 0.5)  # Maximum updates per second

        # Track last message time per chat
        self.last_update_time: Dict[str, float] = {}
        # Track message count per minute for group chats
        self.group_message_count: Dict[str, int] = {}
        self.group_minute_start: Dict[str, float] = {}
        # Track last update time for streaming
        self.last_stream_update: Dict[str, float] = {}

    async def check_and_wait(self, chat_id: str, is_group: bool = False) -> bool:
        """
        Check if we can send a message and wait if necessary
        Returns True if message can be sent, False if we hit a hard limit
        """
        # If rate limiting is disabled, always allow
        if not self.enabled:
            return True

        current_time = time.time()

        # Initialize tracking for this chat if it doesn't exist
        if chat_id not in self.last_update_time:
            self.last_update_time[chat_id] = 0

        # For group chats, handle the group rate limit (default 20 messages per minute)
        if is_group:
            if chat_id not in self.group_message_count:
                self.group_message_count[chat_id] = 0
                self.group_minute_start[chat_id] = current_time

            # Reset counter if a minute has passed
            if current_time - self.group_minute_start[chat_id] > 60:
                self.group_message_count[chat_id] = 0
                self.group_minute_start[chat_id] = current_time

            # Check if we've hit the group message limit
            if self.group_message_count[chat_id] >= self.group_limit:
                # We've hit the hard limit for this minute
                return False

            # Increment the group message counter
            self.group_message_count[chat_id] += 1

        # Calculate time to wait to meet rate limit
        time_since_last_message = current_time - self.last_update_time[chat_id]
        if time_since_last_message < self.private_limit:
            await asyncio.sleep(self.private_limit - time_since_last_message)

        # Update the last message time
        self.last_update_time[chat_id] = time.time()
        return True

    def should_update(self, chat_id: str, is_group: bool, current_length: int, prev_length: int, cutoff: int) -> bool:
        """
        Decide if we should update the message based on rate limiting and content change size
        This is used for streaming to avoid unnecessary updates
        """
        # If rate limiting is disabled, always update
        if not self.enabled:
            return True

        current_time = time.time()

        # Initialize tracking
        if chat_id not in self.last_update_time:
            self.last_update_time[chat_id] = 0
            return True

        if chat_id not in self.last_stream_update:
            self.last_stream_update[chat_id] = 0

        # Check max update frequency for streaming
        time_since_last_stream_update = current_time - self.last_stream_update[chat_id]
        if time_since_last_stream_update < self.max_update_frequency:
            # If we've updated very recently, only update if significant changes (2x cutoff)
            significant_change = abs(current_length - prev_length) > (cutoff * 2)
            if not significant_change:
                return False

        # For group chats, be more conservative with updates
        if is_group:
            # Check if we're approaching the group message limit
            if chat_id in self.group_message_count:
                # If we're at 80% of the limit, be more selective
                if self.group_message_count[chat_id] >= 0.8 * self.group_limit:
                    # Only update if significant changes (2x cutoff)
                    return abs(current_length - prev_length) > (cutoff * 2)

            # Check time since last update
            time_since_last_message = current_time - self.last_update_time[chat_id]

            # For groups, prefer fewer updates
            if time_since_last_message < self.private_limit * 2:
                # If it's been less than 2x the rate limit,
                # only update if significant changes
                return abs(current_length - prev_length) > (cutoff * 1.5)

        # For private chats, be more frequent but still respect limits
        time_since_last_message = current_time - self.last_update_time[chat_id]
        if time_since_last_message < self.private_limit:
            # If it's been less than the rate limit, only update if significant changes
            return abs(current_length - prev_length) > cutoff

        # If we decide to update, update the last stream update time
        self.last_stream_update[chat_id] = current_time

        # Otherwise update is fine
        return True


class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        """
        self.config = config
        self.openai = openai
        self.rate_limiter = RateLimiter(config)
        bot_language = self.config['bot_language']
        self.commands = [
            # BotCommand(
            #     command='help',
            #     description=localized_text('help_description', bot_language),
            # ),
            BotCommand(
                command='reset',
                description=localized_text('reset_description', bot_language),
            ),
            # BotCommand(
            #     command='stats',
            #     description=localized_text('stats_description', bot_language),
            # ),
            # BotCommand(
            #     command='resend',
            #     description=localized_text('resend_description', bot_language),
            # ),
        ]
        # If imaging is enabled, add the "image" command to the list
        if self.config.get('enable_image_generation', False):
            self.commands.append(
                BotCommand(command='image', description='Generate image from prompt (e.g. /image cat)')
            )

        if self.config.get('enable_tts_generation', False):
            self.commands.append(
                BotCommand(
                    command='tts',
                    description=localized_text('tts_description', bot_language),
                )
            )

        self.group_commands = (
            [
                # BotCommand(
                #     command='chat',
                #     description=localized_text('chat_description', bot_language),
                # )
            ]
            + self.commands
        )
        self.disallowed_message = localized_text('disallowed', bot_language)
        self.budget_limit_message = localized_text('budget_limit', bot_language)
        self.usage = {}
        self.last_message = {}
        self.inline_queries_cache = {}
        self.image_prompts_cache = {}  # Cache for storing image prompts
        self.image_quality_cache = {}
        self.image_to_edit_cache = {}  # Cache for storing image to edit data
        self.replies_tracker = {}
        self.pending_quality_confirmations = {}  # Store pending confirmations

    def get_thread_id(self, update: Update) -> str:
        c = update.effective_chat.id
        m = update.effective_message
        if not m:
            raise ValueError('No message found in update')

        if is_private_chat(update):
            return f'{c}'

        if not m.reply_to_message:
            return f'{c}_{m.id}'

        self.replies_tracker[m.id] = (
            self.replies_tracker[m.reply_to_message.id]
            if m.reply_to_message.id in self.replies_tracker
            else m.reply_to_message.id
        )

        thread_id = self.replies_tracker[m.id]
        return f'{c}_{thread_id}'

    def get_real_thread_id(self, update: Update) -> Optional[int]:
        m = update.effective_message
        if not m:
            raise ValueError('No message found in update')

        if not m.reply_to_message:
            return m.id

        self.replies_tracker[m.id] = (
            self.replies_tracker[m.reply_to_message.id]
            if m.reply_to_message.id in self.replies_tracker
            else m.reply_to_message.id
        )

        return self.replies_tracker[m.id]

    def save_reply(self, msg: Message, update: Update):
        if is_private_chat(update):
            return

        self.replies_tracker[msg.message_id] = self.get_real_thread_id(update)

    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the help menu.
        """
        commands = self.group_commands if is_group_chat(update) else self.commands
        commands_description = [f'/{command.command} - {command.description}' for command in commands]
        bot_language = self.config['bot_language']
        help_text = (
            localized_text('help_text', bot_language)[0]
            + '\n\n'
            + '\n'.join(commands_description)
            + '\n\n'
            + localized_text('help_text', bot_language)[1]
            + '\n\n'
            + localized_text('help_text', bot_language)[2]
        )
        await update.message.reply_text(help_text, disable_web_page_preview=True)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Returns token usage statistics for current day and month.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(
                f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                'is not allowed to request their usage statistics'
            )
            await self.send_disallowed_message(update, context)
            return

        logging.info(
            f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
            'requested their usage statistics'
        )

        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        tokens_today, tokens_month = self.usage[user_id].get_current_token_usage()
        images_today, images_month = self.usage[user_id].get_current_image_count()
        (
            transcribe_minutes_today,
            transcribe_seconds_today,
            transcribe_minutes_month,
            transcribe_seconds_month,
        ) = self.usage[user_id].get_current_transcription_duration()
        vision_today, vision_month = self.usage[user_id].get_current_vision_tokens()
        characters_today, characters_month = self.usage[user_id].get_current_tts_usage()
        current_cost = self.usage[user_id].get_current_cost()

        chat_id = update.effective_chat.id
        chat_messages, chat_token_length = await self.openai.get_conversation_stats(chat_id)
        remaining_budget = get_remaining_budget(self.config, self.usage, update)
        bot_language = self.config['bot_language']

        text_current_conversation = (
            f"*{localized_text('stats_conversation', bot_language)[0]}*:\n"
            f"{chat_messages} {localized_text('stats_conversation', bot_language)[1]}\n"
            f"{chat_token_length} {localized_text('stats_conversation', bot_language)[2]}\n"
            "----------------------------\n"
        )

        # Check if image generation is enabled and, if so, generate the image statistics for today
        text_today_images = ''
        if self.config.get('enable_image_generation', False):
            text_today_images = f"{images_today} {localized_text('stats_images', bot_language)}\n"

        text_today_vision = ''
        if self.config.get('enable_vision', False):
            text_today_vision = f"{vision_today} {localized_text('stats_vision', bot_language)}\n"

        text_today_tts = ''
        if self.config.get('enable_tts_generation', False):
            text_today_tts = f"{characters_today} {localized_text('stats_tts', bot_language)}\n"

        text_today = (
            f"*{localized_text('usage_today', bot_language)}:*\n"
            f"{tokens_today} {localized_text('stats_tokens', bot_language)}\n"
            f"{text_today_images}"  # Include the image statistics for today if applicable
            f"{text_today_vision}"
            f"{text_today_tts}"
            f"{transcribe_minutes_today} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_today} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_today']:.2f}\n"
            "----------------------------\n"
        )

        text_month_images = ''
        if self.config.get('enable_image_generation', False):
            text_month_images = f"{images_month} {localized_text('stats_images', bot_language)}\n"

        text_month_vision = ''
        if self.config.get('enable_vision', False):
            text_month_vision = f"{vision_month} {localized_text('stats_vision', bot_language)}\n"

        text_month_tts = ''
        if self.config.get('enable_tts_generation', False):
            text_month_tts = f"{characters_month} {localized_text('stats_tts', bot_language)}\n"

        # Check if image generation is enabled and, if so, generate the image statistics for the month
        text_month = (
            f"*{localized_text('usage_month', bot_language)}:*\n"
            f"{tokens_month} {localized_text('stats_tokens', bot_language)}\n"
            f"{text_month_images}"  # Include the image statistics for the month if applicable
            f"{text_month_vision}"
            f"{text_month_tts}"
            f"{transcribe_minutes_month} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_month} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_month']:.2f}"
        )

        # text_budget filled with conditional content
        text_budget = '\n\n'
        budget_period = self.config['budget_period']
        if remaining_budget < float('inf'):
            text_budget += (
                f"{localized_text('stats_budget', bot_language)}"
                f"{localized_text(budget_period, bot_language)}: "
                f"${remaining_budget:.2f}.\n"
            )
        # No longer works as of July 21st 2023, as OpenAI has removed the billing API
        # add OpenAI account information for admin request
        # if is_admin(self.config, user_id):
        #     text_budget += (
        #         f"{localized_text('stats_openai', bot_language)}"
        #         f"{self.openai.get_billing_current_month():.2f}"
        #     )

        usage_text = text_current_conversation + text_today + text_month + text_budget
        await update.message.reply_text(usage_text, parse_mode=constants.ParseMode.MARKDOWN)

    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resend the last request
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(
                f'User {update.message.from_user.name}  (id: {update.message.from_user.id})'
                ' is not allowed to resend the message'
            )
            await self.send_disallowed_message(update, context)
            return

        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            logging.warning(
                f'User {update.message.from_user.name} (id: {update.message.from_user.id})'
                ' does not have anything to resend'
            )
            await update.effective_message.reply_text(
                message_thread_id=get_forum_thread_id(update),
                text=localized_text('resend_failed', self.config['bot_language']),
            )
            return

        # Update message text, clear self.last_message and send the request to prompt
        logging.info(
            f'Resending the last prompt from user: {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})'
        )
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)

        await self.prompt(update=update, context=context)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resets the conversation.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(
                f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                'is not allowed to reset the conversation'
            )
            await self.send_disallowed_message(update, context)
            return

        ai_context_id = self.get_thread_id(update)
        logging.info(f'Resetting the conversation for {ai_context_id}.')

        reset_content = message_text(update.message)
        await self.openai.reset_chat_history(chat_id=ai_context_id, content=reset_content)
        sent_msg = await update.effective_message.reply_text(
            message_thread_id=get_forum_thread_id(update),
            text=localized_text('reset_done', self.config['bot_language']),
        )
        self.save_reply(sent_msg, update)

    def _get_quality_reply_markup(self, prompt_id):
        if prompt_id not in self.image_quality_cache or 'highest' not in self.image_quality_cache[prompt_id]:
            return None
        highest = self.image_quality_cache[prompt_id]['highest']
        keyboard = []
        # Always show LOW
        row = [InlineKeyboardButton('LOW', callback_data=f'show_quality:{prompt_id}:low')]
        if highest in ('medium', 'high'):
            row.append(InlineKeyboardButton('MEDIUM', callback_data=f'show_quality:{prompt_id}:medium'))
        if highest == 'high':
            row.append(InlineKeyboardButton('HIGH', callback_data=f'show_quality:{prompt_id}:high'))
        keyboard.append(row)
        # Add improve button if not at highest
        if highest == 'low':
            keyboard.append(
                [
                    InlineKeyboardButton(
                        '🖼️ Improve to Medium Quality ($0.1)', callback_data=f'improve_quality:{prompt_id}:medium'
                    )
                ]
            )
        elif highest == 'medium':
            keyboard.append(
                [
                    InlineKeyboardButton(
                        '❗ High Quality Upgrade ($0.2)', callback_data=f'confirm_quality:{prompt_id}:high'
                    )
                ]
            )
        return InlineKeyboardMarkup(keyboard)

    def _get_confirmation_markup(self, prompt_id):
        keyboard = [
            [
                InlineKeyboardButton('❌ Cancel', callback_data=f'cancel_quality:{prompt_id}'),
                InlineKeyboardButton('✅ Confirm ($0.2)', callback_data=f'improve_quality:{prompt_id}:high'),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    async def handle_show_quality(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        if not has_image_gen_permission(self.config, query.from_user.id):
            return

        parts = query.data.split(':')
        prompt_id = parts[1]
        target_quality = parts[2]

        if prompt_id not in self.image_quality_cache or target_quality not in self.image_quality_cache[prompt_id]:
            await query.answer('Sorry, this quality version is no longer available.')
            return

        file_id = self.image_quality_cache[prompt_id][target_quality]['file_id']
        caption = self.image_quality_cache[prompt_id][target_quality]['caption']
        reply_markup = self._get_quality_reply_markup(prompt_id)

        if self.config['image_receive_mode'] == 'photo':
            await context.bot.edit_message_media(
                chat_id=query.message.chat_id,
                message_id=query.message.message_id,
                media=InputMediaPhoto(media=file_id, caption=caption),
                reply_markup=reply_markup,
            )
        elif self.config['image_receive_mode'] == 'document':
            await context.bot.edit_message_media(
                chat_id=query.message.chat_id,
                message_id=query.message.message_id,
                media=InputMediaDocument(media=file_id, caption=caption),
                reply_markup=reply_markup,
            )

    async def handle_quality_confirmation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        if not has_image_gen_permission(self.config, query.from_user.id):
            return

        parts = query.data.split(':')
        prompt_id = parts[1]
        target_quality = parts[2]

        confirmation_text = (
            '⚠️ Premium Feature: High Quality Generation\n'
            'Cost: $0.2 (= 4 medium quality images)\n\n'
            'High quality images provide:\n'
            '• 1024x1024 resolution\n'
            '• Enhanced details and clarity\n'
            '• Better handling of complex scenes\n\n'
            'Are you sure you want to proceed?'
        )

        # Store the confirmation state
        self.pending_quality_confirmations[prompt_id] = {
            'user_id': query.from_user.id,
            'timestamp': datetime.now(),
            'target_quality': target_quality,
        }

        # Update the message with confirmation dialog
        await context.bot.edit_message_caption(
            chat_id=query.message.chat_id,
            message_id=query.message.message_id,
            caption=confirmation_text,
            reply_markup=self._get_confirmation_markup(prompt_id),
        )

    async def handle_quality_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        parts = query.data.split(':')
        prompt_id = parts[1]

        # Remove from pending confirmations
        if prompt_id in self.pending_quality_confirmations:
            del self.pending_quality_confirmations[prompt_id]

        # Restore original markup
        original_caption = self.image_quality_cache[prompt_id]['medium']['caption']
        await context.bot.edit_message_caption(
            chat_id=query.message.chat_id,
            message_id=query.message.message_id,
            caption=original_caption,
            reply_markup=self._get_quality_reply_markup(prompt_id),
        )

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using DALL·E or GPT Image APIs
        """
        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(
            update, context
        ):
            return

        bot_language = self.config['bot_language']

        if not has_image_gen_permission(self.config, update.message.from_user.id):
            logger.warning(
                f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                'is not allowed to generate images'
            )
            return

        image_query = message_text(update.message)

        if not image_query:
            await update.effective_message.reply_text(
                message_thread_id=get_forum_thread_id(update),
                text=localized_text('image_no_prompt', self.config['bot_language']),
            )
            return

        reply = update.message.reply_to_message
        effective_attachment = reply.effective_attachment if reply else None
        image_to_edit_attachment = image_to_edit = None
        if isinstance(effective_attachment, Sequence):
            image_to_edit_attachment = effective_attachment[-1]

        try:
            if image_to_edit_attachment:
                media_file = await context.bot.get_file(image_to_edit_attachment.file_id)
                image_to_edit = io.BytesIO()
                await media_file.download_to_memory(out=image_to_edit)
                image_to_edit.seek(0)
        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_forum_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=(
                    f"{localized_text('media_download_fail', bot_language)[0]}: "
                    f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                ),
                parse_mode=constants.ParseMode.MARKDOWN,
            )
            return

        user_id = update.message.from_user.id

        action_msg = 'EDITING' if image_to_edit else 'GENERATING'
        logging.info(
            f'New image {action_msg} request received from user {update.message.from_user.name} ' f'(id: {user_id})'
        )

        async def _generate():
            nonlocal user_id
            try:
                image_bytes, image_size, price = await self.openai.generate_image(
                    prompt=image_query, image_to_edit=image_to_edit, user_id=str(user_id)
                )

                prompt_id = str(uuid4())
                self.image_prompts_cache[prompt_id] = image_query
                self.image_quality_cache[prompt_id] = {'highest': 'low'}

                # Store image_to_edit in cache if it exists
                if image_to_edit:
                    # Create a copy of the image data for later use
                    image_to_edit.seek(0)
                    image_copy = io.BytesIO(image_to_edit.read())
                    self.image_to_edit_cache[prompt_id] = image_copy

                # Add username to price caption
                username = update.message.from_user.username or update.message.from_user.first_name
                price_with_user = f'{price}\n\nby @{username}'

                reply_markup = self._get_quality_reply_markup(prompt_id)
                if self.config['image_receive_mode'] == 'photo':
                    sent_msg = await update.effective_message.reply_photo(
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        photo=image_bytes,
                        caption=price_with_user,
                        reply_markup=reply_markup,
                    )
                    file_id = sent_msg.photo[-1].file_id
                elif self.config['image_receive_mode'] == 'document':
                    sent_msg = await update.effective_message.reply_document(
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        document=image_bytes,
                        caption=price_with_user,
                        reply_markup=reply_markup,
                    )
                    file_id = sent_msg.document.file_id
                else:
                    raise Exception(
                        f"env variable IMAGE_RECEIVE_MODE has invalid value {self.config['image_receive_mode']}"
                    )

                self.image_quality_cache[prompt_id]['low'] = {'file_id': file_id, 'caption': price_with_user}

                user_id = update.message.from_user.id
                if user_id not in self.usage:
                    self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

                self.usage[user_id].add_image_request(image_size, self.config['image_prices'])
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage['guests'].add_image_request(image_size, self.config['image_prices'])

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_forum_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('image_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )

        await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_PHOTO)

    async def handle_improve_quality(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        if not has_image_gen_permission(self.config, query.from_user.id):
            return

        parts = query.data.split(':')
        prompt_id = parts[1]
        target_quality = parts[2]

        # Check if this is a confirmed action for high quality
        if target_quality == 'high' and prompt_id not in self.pending_quality_confirmations:
            # If not confirmed, show confirmation dialog
            await self.handle_quality_confirmation(update, context)
            return

        # Clean up confirmation state if it exists
        if prompt_id in self.pending_quality_confirmations:
            del self.pending_quality_confirmations[prompt_id]

        if prompt_id not in self.image_prompts_cache:
            await query.answer('Sorry, the prompt is no longer available.')
            return

        prompt = self.image_prompts_cache[prompt_id]

        # Get image_to_edit from cache if it exists
        image_to_edit = None
        if prompt_id in self.image_to_edit_cache:
            image_to_edit = self.image_to_edit_cache[prompt_id]
            image_to_edit.seek(0)

        loading_keyboard = [[InlineKeyboardButton('⏳ Generating...', callback_data='loading')]]
        loading_markup = InlineKeyboardMarkup(loading_keyboard)
        await context.bot.edit_message_reply_markup(
            chat_id=query.message.chat_id, message_id=query.message.message_id, reply_markup=loading_markup
        )

        async def _generate():
            try:
                user_id = query.from_user.id

                quality_param = 'high' if target_quality == 'high' else 'medium'
                image_bytes, image_size, price = await self.openai.generate_image(
                    prompt=prompt, quality=quality_param, image_to_edit=image_to_edit, user_id=str(user_id)
                )

                # Add username to price caption
                username = query.from_user.username or query.from_user.first_name
                price_with_user = f'{price}\n\nby @{username}'

                self.image_quality_cache[prompt_id]['highest'] = quality_param

                reply_markup = self._get_quality_reply_markup(prompt_id)
                if self.config['image_receive_mode'] == 'photo':
                    sent_msg = await context.bot.edit_message_media(
                        chat_id=query.message.chat_id,
                        message_id=query.message.message_id,
                        media=InputMediaPhoto(image_bytes, caption=price_with_user),
                        reply_markup=reply_markup,
                    )
                    file_id = sent_msg.photo[-1].file_id
                else:
                    sent_msg = await context.bot.edit_message_media(
                        chat_id=query.message.chat_id,
                        message_id=query.message.message_id,
                        media=InputMediaDocument(image_bytes, caption=price_with_user),
                        reply_markup=reply_markup,
                    )
                    file_id = sent_msg.document.file_id

                self.image_quality_cache[prompt_id][quality_param] = {'file_id': file_id, 'caption': price_with_user}

                user_id = query.from_user.id
                if user_id not in self.usage:
                    self.usage[user_id] = UsageTracker(user_id, query.from_user.name)

                self.usage[user_id].add_image_request(image_size, self.config['image_prices'])
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage['guests'].add_image_request(image_size, self.config['image_prices'])

            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=f'Failed to improve image quality: {str(e)}',
                    reply_to_message_id=query.message.message_id,
                )

        await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_PHOTO)

    async def tts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an speech for the given input using TTS APIs
        """
        if not self.config['enable_tts_generation'] or not await self.check_allowed_and_within_budget(update, context):
            return

        tts_query = message_text(update.message)
        if update.message.reply_to_message and update.message.reply_to_message.text:
            reply_text = message_text(update.message.reply_to_message)
            tts_query = f'{reply_text} {tts_query}'.strip()

        if not tts_query:
            await update.effective_message.reply_text(
                message_thread_id=get_forum_thread_id(update),
                text=localized_text('tts_no_prompt', self.config['bot_language']),
            )
            return

        logging.info(
            f'New speech generation request received from user {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})'
        )

        async def _generate():
            try:
                speech_file, text_length = await self.openai.generate_speech(text=tts_query)

                sent_msg = await update.effective_message.reply_voice(
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    voice=speech_file,
                )
                self.save_reply(sent_msg, update)
                speech_file.close()
                # add image request to users usage tracker
                user_id = update.message.from_user.id
                self.usage[user_id].add_tts_request(text_length, self.config['tts_model'], self.config['tts_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage['guests'].add_tts_request(
                        text_length, self.config['tts_model'], self.config['tts_prices']
                    )

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_forum_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('tts_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )

        await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_VOICE)

    async def transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Transcribe audio messages.
        """
        if not self.config['enable_transcription'] or not await self.check_allowed_and_within_budget(update, context):
            return

        if is_group_chat(update) and self.config['ignore_group_transcriptions']:
            logging.info('Transcription coming from group chat, ignoring...')
            return

        ai_context_id = self.get_thread_id(update)
        filename = update.message.effective_attachment.file_unique_id

        async def _execute():
            filename_mp3 = f'{filename}.mp3'
            bot_language = self.config['bot_language']
            try:
                media_file = await context.bot.get_file(update.message.effective_attachment.file_id)
                await media_file.download_to_drive(filename)
            except Exception as e:
                logging.exception(e)
                sent_msg = await update.effective_message.reply_text(
                    message_thread_id=get_forum_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=(
                        f"{localized_text('media_download_fail', bot_language)[0]}: "
                        f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                    ),
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                self.save_reply(sent_msg, update)
                return

            try:
                audio_track = AudioSegment.from_file(filename)
                # FIXME do not save to file
                audio_track.export(filename_mp3, format='mp3')
                logging.info(
                    f'New transcribe request received from user {update.message.from_user.name} '
                    f'(id: {update.message.from_user.id})'
                )

            except Exception as e:
                logging.exception(e)
                # await update.effective_message.reply_text(
                #     message_thread_id=get_forum_thread_id(update),
                #     reply_to_message_id=get_reply_to_message_id(self.config, update),
                #     text=localized_text('media_type_fail', bot_language),
                # )
                if os.path.exists(filename):
                    os.remove(filename)
                return

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

            try:
                transcript = await self.openai.transcribe(filename_mp3)

                transcription_price = self.config['transcription_price']
                self.usage[user_id].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                allowed_user_ids = self.config['allowed_user_ids'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage['guests'].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                # check if transcript starts with any of the prefixes
                response_to_transcription = any(
                    transcript.lower().startswith(prefix.lower()) if prefix else False
                    for prefix in self.config['voice_reply_prompts']
                )

                if self.config['voice_reply_transcript'] and not response_to_transcription:
                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\""
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        sent_msg = await update.effective_message.reply_text(
                            message_thread_id=get_forum_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN,
                        )
                        self.save_reply(sent_msg, update)
                else:
                    # Get the response of the transcript
                    response, total_tokens = await self.openai.get_chat_response(
                        chat_id=ai_context_id, query=transcript, user_id=str(user_id)
                    )

                    self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])
                    if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                        self.usage['guests'].add_chat_tokens(total_tokens, self.config['token_price'])

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = (
                        f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\"\n\n"
                        f"_{localized_text('answer', bot_language)}:_\n{response}"
                    )
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        sent_msg = await update.effective_message.reply_text(
                            message_thread_id=get_forum_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN,
                            disable_web_page_preview=True,
                        )
                        self.save_reply(sent_msg, update)

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_forum_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('transcribe_fail', bot_language)}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
            finally:
                if os.path.exists(filename_mp3):
                    os.remove(filename_mp3)
                if os.path.exists(filename):
                    os.remove(filename)

        await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)

    @with_conversation_lock
    async def vision(self, update: Update, context: ContextTypes.DEFAULT_TYPE, reply: Message = None):
        await self._vision_no_lock(update, context, reply)

    async def _vision_no_lock(self, update: Update, context: ContextTypes.DEFAULT_TYPE, reply: Message = None):
        """
        Interpret image using vision model.
        """
        if not self.config['enable_vision'] or not await self.check_allowed_and_within_budget(update, context):
            return

        ai_context_id = self.get_thread_id(update)
        chat_id = update.effective_chat.id

        if reply is None:
            prompt = update.message.caption
        else:
            prompt = message_text(update.message)

        if reply is None and is_group_chat(update):
            if self.config['ignore_group_vision']:
                logging.info('Vision coming from group chat, ignoring...')
                return
            else:
                no_reply = (
                    update.effective_message.reply_to_message is None
                    or update.effective_message.reply_to_message.from_user.id != context.bot.id
                )

                trigger_keyword = self.config['group_trigger_keyword']
                no_keyword = (prompt is None and trigger_keyword != '') or (
                    prompt is not None and not prompt.lower().startswith(trigger_keyword.lower())
                )

                if no_reply and no_keyword:
                    logging.info('Vision coming from group chat with wrong keyword, ignoring...')
                    return
        elif reply and is_group_chat(update):
            trigger_keyword = self.config['group_trigger_keyword']
            no_keyword = (prompt is None and trigger_keyword != '') or (
                prompt is not None and not prompt.lower().startswith(trigger_keyword.lower())
            )

            if no_keyword:
                logging.info('Vision coming from group chat with wrong keyword, ignoring...')
                return

        effective_attachment = reply.effective_attachment if reply else update.message.effective_attachment
        if isinstance(effective_attachment, Sequence):
            image = effective_attachment[-1]
        else:
            image = effective_attachment

        async def _execute():
            bot_language = self.config['bot_language']
            total_tokens = 0

            try:
                media_file = await context.bot.get_file(image.file_id)
                temp_file = io.BytesIO(await media_file.download_as_bytearray())
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_forum_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=(
                        f"{localized_text('media_download_fail', bot_language)[0]}: "
                        f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                    ),
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                return

            # convert jpg from telegram to png as understood by openai

            temp_file_png = io.BytesIO()

            try:
                original_image = Image.open(temp_file)

                original_image.save(temp_file_png, format='PNG')
                logging.info(
                    f'New vision request received from user {update.message.from_user.name} '
                    f'(id: {update.message.from_user.id})'
                )

            except Exception as e:
                logging.exception(e)
                # await update.effective_message.reply_text(
                #     message_thread_id=get_forum_thread_id(update),
                #     reply_to_message_id=get_reply_to_message_id(self.config, update),
                #     text=localized_text('media_type_fail', bot_language),
                # )

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

            if self.config['stream']:
                stream_response = self.openai.interpret_image_stream(
                    chat_id=ai_context_id, fileobj=temp_file_png, prompt=prompt, user_id=str(user_id)
                )
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                processed_chunks = []  # Track which chunks have been processed
                is_group = is_group_chat(update)
                str_chat_id = str(chat_id)

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content, self.save_reply)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        # Keep track of the last chunk as current content
                        content = stream_chunks[-1]

                        # Process any new complete chunks
                        for chunk_idx in range(len(processed_chunks), len(stream_chunks) - 1):
                            try:
                                # If we have a message already, edit it with the current complete chunk
                                if sent_message is not None:
                                    # Check rate limits before sending
                                    can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                                    if not can_send:
                                        logging.warning(f'Rate limit reached for chat {chat_id}, skipping update')
                                        continue

                                    await edit_message_with_retry(
                                        context,
                                        chat_id,
                                        str(sent_message.message_id),
                                        stream_chunks[chunk_idx],
                                    )

                                # Create a new message for the next chunk (current content)
                                # Check rate limits before sending
                                can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                                if not can_send:
                                    logging.warning(f'Rate limit reached for chat {chat_id}, skipping new message')
                                    # Mark this chunk as processed anyway to avoid creating multiple messages later
                                    processed_chunks.append(chunk_idx)
                                    continue

                                sent_message = await update.effective_message.reply_text(
                                    message_thread_id=get_forum_thread_id(update),
                                    text=content if len(content) > 0 else '...',
                                )
                                self.save_reply(sent_message, update)
                                processed_chunks.append(chunk_idx)
                            except Exception as e:
                                logging.error(f'Error handling chunk: {e}')
                                pass

                        # If we've processed all complete chunks, continue streaming with the last chunk
                        if len(processed_chunks) == len(stream_chunks) - 1:
                            # We've handled all complete chunks, continue with normal streaming for the last chunk
                            pass
                        else:
                            # We still have unprocessed complete chunks, skip this iteration
                            continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            # Check rate limits before sending first message
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                            if not can_send:
                                logging.warning(f'Rate limit reached for chat {chat_id}, waiting for next update')
                                continue

                            if sent_message is not None:
                                await context.bot.delete_message(
                                    chat_id=sent_message.chat_id,
                                    message_id=sent_message.message_id,
                                )
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_forum_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=content,
                            )
                            self.save_reply(sent_message, update)
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content

                        try:
                            # Instead of waiting, check if we should update the message
                            should_update = tokens != 'not_finished' or self.rate_limiter.should_update(
                                str_chat_id, is_group, len(content), len(prev), cutoff
                            )

                            # If we shouldn't update, skip this iteration
                            if not should_update:
                                continue

                            # Otherwise, check rate limits and update if possible
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                            if not can_send:
                                logging.warning(f'Rate limit reached for chat {chat_id}, skipping update')
                                continue

                            use_markdown = tokens != 'not_finished'
                            await edit_message_with_retry(
                                context,
                                chat_id,
                                str(sent_message.message_id),
                                text=content,
                                markdown=use_markdown,
                            )

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue

                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        # Add a small delay between updates
                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:
                try:
                    interpretation, total_tokens = await self.openai.interpret_image(
                        ai_context_id, temp_file_png, prompt=prompt, user_id=str(user_id)
                    )

                    try:
                        # Check rate limits before sending
                        is_group = is_group_chat(update)
                        str_chat_id = str(chat_id)
                        can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)

                        if can_send:
                            sent_msg = await update.effective_message.reply_text(
                                message_thread_id=get_forum_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=interpretation,
                                parse_mode=constants.ParseMode.MARKDOWN,
                            )
                            self.save_reply(sent_msg, update)
                        else:
                            # If rate limit reached, try without markdown
                            logging.warning(f'Rate limit reached for chat {chat_id}, trying again in 1 second')
                            await asyncio.sleep(1)
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)

                            if can_send:
                                sent_msg = await update.effective_message.reply_text(
                                    message_thread_id=get_forum_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                                    text=interpretation,
                                )
                                self.save_reply(sent_msg, update)
                            else:
                                logging.error('Failed to send vision response due to rate limits')
                    except BadRequest:
                        try:
                            # Check rate limits before retrying
                            is_group = is_group_chat(update)
                            str_chat_id = str(chat_id)
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)

                            if can_send:
                                sent_msg = await update.effective_message.reply_text(
                                    message_thread_id=get_forum_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                                    text=interpretation,
                                )
                                self.save_reply(sent_msg, update)
                            else:
                                logging.error('Failed to send vision response due to rate limits')
                        except Exception as e:
                            logging.exception(e)
                            await update.effective_message.reply_text(
                                message_thread_id=get_forum_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                                parse_mode=constants.ParseMode.MARKDOWN,
                            )
                except Exception as e:
                    logging.exception(e)
                    await update.effective_message.reply_text(
                        message_thread_id=get_forum_thread_id(update),
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                        parse_mode=constants.ParseMode.MARKDOWN,
                    )
            vision_token_price = self.config['vision_token_price']
            self.usage[user_id].add_vision_tokens(total_tokens, vision_token_price)

            allowed_user_ids = self.config['allowed_user_ids'].split(',')
            if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                self.usage['guests'].add_vision_tokens(total_tokens, vision_token_price)

        await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)

    async def reaction(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        React to incoming reactions and respond accordingly.
        """
        logging.info(
            f'New reaction received from user {update.effective_sender.name} (id: {update.effective_sender.id})'
        )
        new_reactions = set()
        if update.message_reaction.new_reaction:
            new_reactions = {r.emoji for r in update.message_reaction.new_reaction}

        if '👍' not in new_reactions and '👎' not in new_reactions:
            return

        emoji_to_message = {
            '👍': 'Yes',
            '👎': 'No',
        }

        new_update = Update(
            update_id=update.update_id,
            message=Message(  # required to behave like usual prompt
                message_id=update.message_reaction.message_id,
                date=update.message_reaction.date,
                chat=update.message_reaction.chat,
                from_user=update.message_reaction.user,
                text=''.join(emoji_to_message[emoji] for emoji in new_reactions),
                reply_to_message=Message(  # required for context id resolving
                    message_id=update.message_reaction.message_id,
                    date=update.message_reaction.date,
                    chat=update.message_reaction.chat,
                ),
            ),
        )

        # required to enable shortcuts:
        new_update.set_bot(update.get_bot())
        new_update.message.set_bot(update.get_bot())

        # now call with fake compatible update
        await self.prompt(new_update, context)

    @with_conversation_lock
    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        React to incoming messages and respond accordingly.
        """
        if update.edited_message or not update.message or update.message.via_bot:
            return

        if not await self.check_allowed_and_within_budget(update, context):
            return

        ai_context_id = self.get_thread_id(update)
        logging.info(f'New message received from user {update.message.from_user.name} (CTX: {ai_context_id})')
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        prompt = message_text(update.message)
        self.last_message[chat_id] = prompt

        if update.message.reply_to_message and update.message.reply_to_message.effective_attachment:
            return await self._vision_no_lock(update, context, update.message.reply_to_message)

        if is_group_chat(update):
            trigger_keyword = self.config['group_trigger_keyword']

            if prompt.lower().startswith(trigger_keyword.lower()) or update.message.text.lower().startswith('/chat'):
                if prompt.lower().startswith(trigger_keyword.lower()):
                    prompt = prompt[len(trigger_keyword) :].strip()

                if (
                    update.message.reply_to_message
                    and update.message.reply_to_message.text
                    and update.message.reply_to_message.from_user.id != context.bot.id
                ):
                    reply_text = message_text(update.message.reply_to_message)
                    prompt = f'"{reply_text}" {prompt}'
            else:
                if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
                    logging.info('Message is a reply to the bot, allowing...')
                else:
                    logging.warning('Message does not start with trigger keyword, ignoring...')
                    return

        try:
            total_tokens = 0

            if self.config['stream']:
                await update.effective_message.reply_chat_action(
                    action=constants.ChatAction.TYPING,
                    message_thread_id=get_forum_thread_id(update),
                )

                stream_response = self.openai.get_chat_response_stream(
                    chat_id=ai_context_id, query=prompt, user_id=str(user_id)
                )
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                processed_chunks = []  # Track which chunks have been processed
                is_group = is_group_chat(update)
                str_chat_id = str(chat_id)

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content, self.save_reply)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        # Keep track of the last chunk as current content
                        content = stream_chunks[-1]

                        # Process any new complete chunks
                        for chunk_idx in range(len(processed_chunks), len(stream_chunks) - 1):
                            try:
                                # If we have a message already, edit it with the current complete chunk
                                if sent_message is not None:
                                    # Check rate limits before sending
                                    can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                                    if not can_send:
                                        logging.warning(f'Rate limit reached for chat {chat_id}, skipping update')
                                        continue

                                    await edit_message_with_retry(
                                        context,
                                        chat_id,
                                        str(sent_message.message_id),
                                        stream_chunks[chunk_idx],
                                    )

                                # Create a new message for the next chunk (current content)
                                # Check rate limits before sending
                                can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                                if not can_send:
                                    logging.warning(f'Rate limit reached for chat {chat_id}, skipping new message')
                                    # Mark this chunk as processed anyway to avoid creating multiple messages later
                                    processed_chunks.append(chunk_idx)
                                    continue

                                sent_message = await update.effective_message.reply_text(
                                    message_thread_id=get_forum_thread_id(update),
                                    text=content if len(content) > 0 else '...',
                                )
                                self.save_reply(sent_message, update)
                                processed_chunks.append(chunk_idx)
                            except Exception as e:
                                logging.error(f'Error handling chunk: {e}')
                                pass

                        # If we've processed all complete chunks, continue streaming with the last chunk
                        if len(processed_chunks) == len(stream_chunks) - 1:
                            # We've handled all complete chunks, continue with normal streaming for the last chunk
                            pass
                        else:
                            # We still have unprocessed complete chunks, skip this iteration
                            continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            # Check rate limits before sending first message
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                            if not can_send:
                                logging.warning(f'Rate limit reached for chat {chat_id}, waiting for next update')
                                continue

                            if sent_message is not None:
                                await context.bot.delete_message(
                                    chat_id=sent_message.chat_id,
                                    message_id=sent_message.message_id,
                                )
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_forum_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=content,
                            )
                            self.save_reply(sent_message, update)
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content

                        try:
                            # Instead of waiting, check if we should update the message
                            should_update = tokens != 'not_finished' or self.rate_limiter.should_update(
                                str_chat_id, is_group, len(content), len(prev), cutoff
                            )

                            # If we shouldn't update, skip this iteration
                            if not should_update:
                                continue

                            # Otherwise, check rate limits and update if possible
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                            if not can_send:
                                logging.warning(f'Rate limit reached for chat {chat_id}, skipping update')
                                continue

                            use_markdown = tokens != 'not_finished'
                            await edit_message_with_retry(
                                context,
                                chat_id,
                                str(sent_message.message_id),
                                text=content,
                                markdown=use_markdown,
                            )

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue

                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        # Add a small delay between updates
                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:

                async def _reply():
                    nonlocal total_tokens
                    response, total_tokens = await self.openai.get_chat_response(
                        chat_id=ai_context_id, query=prompt, user_id=str(user_id)
                    )

                    if is_direct_result(response):
                        return await handle_direct_result(self.config, update, response, self.save_reply)

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    chunks = split_into_chunks(response)

                    # Check if we're in a group
                    is_group = is_group_chat(update)
                    str_chat_id = str(chat_id)

                    for index, chunk in enumerate(chunks):
                        try:
                            # Check rate limits before sending
                            can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                            if not can_send:
                                # If rate limit reached, add a delay and notify
                                logging.warning(f'Rate limit reached for chat {chat_id}, waiting...')
                                if index > 0:
                                    # Only add this notification for subsequent chunks
                                    await update.effective_message.reply_text(
                                        message_thread_id=get_forum_thread_id(update),
                                        text='⚠️ Rate limit reached. Remaining response will be sent shortly.',
                                    )
                                await asyncio.sleep(60)  # Wait for a minute
                                # Try again after waiting
                                can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)

                            if can_send:
                                sent_msg = await update.effective_message.reply_text(
                                    message_thread_id=get_forum_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config, update)
                                    if index == 0
                                    else None,
                                    text=chunk,
                                    parse_mode=constants.ParseMode.MARKDOWN,
                                    disable_web_page_preview=True,
                                )
                                self.save_reply(sent_msg, update)
                            else:
                                logging.error('Failed to send chunk due to rate limits even after waiting')
                        except Exception:
                            try:
                                # Check rate limits before retrying
                                can_send = await self.rate_limiter.check_and_wait(str_chat_id, is_group)
                                if not can_send:
                                    logging.warning(f'Rate limit reached for chat {chat_id}, skipping chunk')
                                    continue

                                sent_msg = await update.effective_message.reply_text(
                                    message_thread_id=get_forum_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config, update)
                                    if index == 0
                                    else None,
                                    text=chunk,
                                    disable_web_page_preview=True,
                                )
                                self.save_reply(sent_msg, update)
                            except Exception as exception:
                                raise exception

                await wrap_with_indicator(update, context, _reply, constants.ChatAction.TYPING)

            add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_forum_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"{localized_text('chat_fail', self.config['bot_language'])} {str(e)}",
                parse_mode=constants.ParseMode.MARKDOWN,
            )

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the inline query. This is run when you type: @botusername <query>
        """
        query = update.inline_query.query
        user_id = update.inline_query.from_user.id
        name = update.inline_query.from_user.name

        if len(query) < 3:
            return

        if not await self.check_allowed_and_within_budget(update, context, is_inline=True):
            logging.warning(f'User {name} (id: {user_id}) not allowed or over budget')
            return

        result_id = str(uuid4())
        self.inline_queries_cache[result_id] = query
        await self.send_inline_query_result(update, result_id, message_content=query)

    async def send_inline_query_result(self, update: Update, result_id, message_content, callback_data=''):
        """
        Send inline query result with a placeholder message that will be updated with the actual response
        """
        try:
            bot_language = self.config['bot_language']
            loading_tr = localized_text('loading', bot_language)
            answer_tr = localized_text('answer', bot_language)

            placeholder_text = f'{message_content}\n\n_{answer_tr}:_\n{loading_tr}'

            # Add a placeholder button
            reply_markup = InlineKeyboardMarkup(
                [[InlineKeyboardButton('⏳ Generating...', callback_data='generating')]]
            )

            inline_query_result = InlineQueryResultArticle(
                id=result_id,
                title=localized_text('ask_chatgpt', bot_language),
                input_message_content=InputTextMessageContent(
                    placeholder_text, parse_mode=constants.ParseMode.MARKDOWN
                ),
                description=message_content,
                thumbnail_url='https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea'
                '-b02a7a32149a.png',
                reply_markup=reply_markup,
            )

            await update.inline_query.answer([inline_query_result], cache_time=0)
        except Exception as e:
            logging.error(f'Failed to send inline result for result_id {result_id}: {str(e)}')
            logging.exception(e)

    async def handle_chosen_inline_result(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the chosen inline result and generate the response
        """
        if not update.chosen_inline_result:
            logging.warning('Received empty chosen_inline_result')
            return

        result_id = update.chosen_inline_result.result_id
        inline_message_id = update.chosen_inline_result.inline_message_id
        user_id = update.chosen_inline_result.from_user.id
        name = update.chosen_inline_result.from_user.name

        # Retrieve the query from cache
        query = self.inline_queries_cache.get(result_id)
        if not query:
            logging.error(f'Query not found in cache for result_id: {result_id}')
            error_message = f'{localized_text("error", self.config["bot_language"])}. {localized_text("try_again", self.config["bot_language"])}'
            await edit_message_with_retry(
                context, chat_id=None, message_id=inline_message_id, text=error_message, is_inline=True
            )
            return

        logging.info(f'User {name} (id: {user_id}) selected result_id: {result_id} ({query})')
        self.inline_queries_cache.pop(result_id)

        bot_language = self.config['bot_language']
        answer_tr = localized_text('answer', bot_language)
        loading_tr = localized_text('loading', bot_language)
        total_tokens = 0
        str_user_id = str(user_id)  # Use user_id as chat_id for inline messages

        try:
            if self.config['stream']:
                stream_response = self.openai.get_chat_response_stream(
                    chat_id=str(user_id), query=query, user_id=str(user_id)
                )
                i = 0
                prev = ''
                backoff = 0
                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        logging.info('Received direct result, not supported in inline mode')
                        unavailable_message = localized_text('function_unavailable_in_inline_mode', bot_language)
                        await edit_message_with_retry(
                            context,
                            chat_id=None,
                            message_id=inline_message_id,
                            text=f'{query}\n\n_{answer_tr}:_\n{unavailable_message}',
                            is_inline=True,
                        )
                        return

                    if len(content.strip()) == 0:
                        continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            # Check rate limits before sending first update
                            can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                            if not can_send:
                                logging.warning(f'Rate limit reached for user {user_id}, waiting for next update')
                                continue

                            await edit_message_with_retry(
                                context,
                                chat_id=None,
                                message_id=inline_message_id,
                                text=f'{query}\n\n{answer_tr}:\n{content}',
                                is_inline=True,
                            )
                        except Exception:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content
                        try:
                            # Instead of waiting, check if we should update the message
                            should_update = tokens != 'not_finished' or self.rate_limiter.should_update(
                                str_user_id, False, len(content), len(prev), cutoff
                            )

                            # If we shouldn't update, skip this iteration
                            if not should_update:
                                continue

                            # Check rate limits before updating message
                            can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                            if not can_send:
                                logging.warning(f'Rate limit reached for user {user_id}, skipping update')
                                continue

                            use_markdown = tokens != 'not_finished'
                            divider = '_' if use_markdown else ''
                            text = f'{query}\n\n{divider}{answer_tr}:{divider}\n{content}'

                            # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                            text = text[:4096]

                            await edit_message_with_retry(
                                context,
                                chat_id=None,
                                message_id=inline_message_id,
                                text=text,
                                markdown=use_markdown,
                                is_inline=True,
                            )

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue
                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue
                        except Exception:
                            backoff += 5
                            continue

                        # Add delay between updates to respect rate limits
                        await asyncio.sleep(0.1)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:
                # Show loading message
                # Check rate limits before sending
                can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                if can_send:
                    await context.bot.edit_message_text(
                        inline_message_id=inline_message_id,
                        text=f'{query}\n\n_{answer_tr}:_\n{loading_tr}',
                        parse_mode=constants.ParseMode.MARKDOWN,
                    )

                response, total_tokens = await self.openai.get_chat_response(
                    chat_id=str(user_id), query=query, user_id=str(user_id)
                )

                if is_direct_result(response):
                    logging.info('Received direct result, not supported in inline mode')
                    unavailable_message = localized_text('function_unavailable_in_inline_mode', bot_language)

                    # Check rate limits before sending final message
                    can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                    if can_send:
                        await edit_message_with_retry(
                            context,
                            chat_id=None,
                            message_id=inline_message_id,
                            text=f'{query}\n\n_{answer_tr}:_\n{unavailable_message}',
                            is_inline=True,
                        )
                    return

                text_content = f'{query}\n\n_{answer_tr}:_\n{response}'
                # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                text_content = text_content[:4096]

                # Check rate limits before sending final message
                can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                if can_send:
                    await edit_message_with_retry(
                        context,
                        chat_id=None,
                        message_id=inline_message_id,
                        text=text_content,
                        is_inline=True,
                    )
                else:
                    logging.warning(f'Rate limit reached for user {user_id}, waiting to send final response')
                    await asyncio.sleep(1)  # Wait a bit
                    # Try one more time
                    can_send = await self.rate_limiter.check_and_wait(str_user_id, False)
                    if can_send:
                        await edit_message_with_retry(
                            context,
                            chat_id=None,
                            message_id=inline_message_id,
                            text=text_content,
                            is_inline=True,
                        )

            add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)

        except Exception as e:
            logging.error(f'Failed to respond to an inline query: {str(e)}')
            logging.exception(e)
            localized_answer = localized_text('chat_fail', self.config['bot_language'])
            await edit_message_with_retry(
                context,
                chat_id=None,
                message_id=inline_message_id,
                text=f'{query}\n\n_{answer_tr}:_\n{localized_answer} {str(e)}',
                is_inline=True,
            )

    async def check_allowed_and_within_budget(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_inline=False
    ) -> bool:
        """
        Checks if the user is allowed to use the bot and if they are within their budget
        :param update: Telegram update object
        :param context: Telegram context object
        :param is_inline: Boolean flag for inline queries
        :return: Boolean indicating if the user is allowed to use the bot
        """
        name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
        user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id

        if not await is_allowed(self.config, update, context, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) is not allowed to use the bot')
            return False
        if not is_within_budget(self.config, self.usage, update, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) reached their usage limit')
            await self.send_budget_reached_message(update, context, is_inline)
            return False

        return True

    async def send_disallowed_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the disallowed message to the user.
        """
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_forum_thread_id(update),
                text=self.disallowed_message,
                disable_web_page_preview=True,
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.disallowed_message)

    async def send_budget_reached_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the budget reached message to the user.
        """
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_forum_thread_id(update), text=self.budget_limit_message
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.budget_limit_message)

    async def post_init(self, application: Application) -> None:
        """
        Post initialization hook for the bot.
        """
        await application.bot.set_my_commands(self.group_commands, scope=BotCommandScopeAllGroupChats())
        await application.bot.set_my_commands(self.commands)

        if self.config['database_url']:
            self.openai.db_pool = await asyncpg.create_pool(dsn=self.config['database_url'])
            async with self.openai.db_pool.acquire() as connection:
                await connection.execute('drop schema public cascade')
                await connection.execute('create schema public')

    async def post_shutdown(self, _: Application) -> None:
        if self.openai.db_pool:
            await self.openai.db_pool.close()

    def run(self):
        """
        Runs the bot indefinitely until the user presses Ctrl+C
        """
        application = (
            ApplicationBuilder()
            .token(self.config['token'])
            .proxy_url(self.config['proxy'])
            .get_updates_proxy_url(self.config['proxy'])
            .post_init(self.post_init)
            .post_shutdown(self.post_shutdown)
            .concurrent_updates(True)
            .build()
        )

        application.add_handler(CommandHandler('reset', self.reset))
        application.add_handler(CommandHandler('image', self.image))
        application.add_handler(CommandHandler('tts', self.tts))
        # application.add_handler(CommandHandler('start', self.help))
        # application.add_handler(CommandHandler('stats', self.stats))
        # application.add_handler(CommandHandler('resend', self.resend))
        # application.add_handler(
        #     CommandHandler(
        #         'chat',
        #         self.prompt,
        #         filters=filters.ChatType.GROUP | filters.ChatType.SUPERGROUP,
        #     )
        # )
        application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, self.vision))
        application.add_handler(
            MessageHandler(
                filters.AUDIO
                | filters.VOICE
                | filters.Document.AUDIO
                | filters.VIDEO
                | filters.VIDEO_NOTE
                | filters.Document.VIDEO,
                self.transcribe,
            )
        )
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))
        application.add_handler(
            MessageReactionHandler(
                self.reaction, message_reaction_types=MessageReactionHandler.MESSAGE_REACTION_UPDATED
            )
        )
        application.add_handler(CallbackQueryHandler(self.handle_improve_quality, pattern='^improve_quality:'))
        application.add_handler(CallbackQueryHandler(self.handle_show_quality, pattern='^show_quality:'))
        application.add_handler(CallbackQueryHandler(self.handle_quality_confirmation, pattern='^confirm_quality:'))
        application.add_handler(CallbackQueryHandler(self.handle_quality_cancel, pattern='^cancel_quality:'))
        application.add_handler(
            InlineQueryHandler(
                self.inline_query,
                chat_types=[
                    constants.ChatType.GROUP,
                    constants.ChatType.SUPERGROUP,
                    constants.ChatType.PRIVATE,
                ],
            )
        )
        # Add handler for chosen inline results
        application.add_handler(ChosenInlineResultHandler(self.handle_chosen_inline_result))

        application.add_error_handler(error_handler)

        application.run_polling(allowed_updates=Update.ALL_TYPES)
