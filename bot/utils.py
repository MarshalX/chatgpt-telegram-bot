from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
from typing import Callable, Optional

import telegram
from telegram import ChatMember, Message, MessageEntity, Update, constants
from telegram.ext import CallbackContext, ContextTypes
from telegram.helpers import escape_markdown
from usage_tracker import UsageTracker


def message_text(message: Message) -> str:
    """
    Returns the text of a message, excluding any bot commands.
    """
    message_txt = message.text_markdown_v2
    if not message_txt:
        return ''

    for _, text in sorted(
        message.parse_entities([MessageEntity.BOT_COMMAND]).items(),
        key=(lambda item: item[0].offset),
    ):
        message_txt = message_txt.replace(escape_markdown(text, version=2), '').strip()

    return message_txt.strip()


async def is_user_in_group(update: Update, context: CallbackContext, user_id: int) -> bool:
    """
    Checks if user_id is a member of the group
    """
    try:
        chat_member = await context.bot.get_chat_member(update.message.chat_id, user_id)
        return chat_member.status in [
            ChatMember.OWNER,
            ChatMember.ADMINISTRATOR,
            ChatMember.MEMBER,
        ]
    except telegram.error.BadRequest as e:
        if str(e) == 'User not found':
            return False
        else:
            raise e
    except Exception as e:
        raise e


def get_forum_thread_id(update: Update) -> int | None:
    """
    Gets the message thread id for the update, if any
    """
    if update.effective_message and update.effective_message.is_topic_message:
        return update.effective_message.message_thread_id
    return None


def get_stream_cutoff_values(update: Update, content: str) -> int:
    """
    Gets the stream cutoff values for the message length
    """
    if is_group_chat(update):
        # group chats have stricter flood limits
        return 180 if len(content) > 1000 else 120 if len(content) > 200 else 90 if len(content) > 50 else 50
    return 90 if len(content) > 1000 else 45 if len(content) > 200 else 25 if len(content) > 50 else 15


def is_group_chat(update: Update) -> bool:
    """
    Checks if the message was sent from a group chat
    """
    if not update.effective_chat:
        return False
    return update.effective_chat.type in [
        constants.ChatType.GROUP,
        constants.ChatType.SUPERGROUP,
    ]


def is_private_chat(update: Update) -> bool:
    """
    Checks if the message was sent from a private chat
    """
    if not update.effective_chat:
        return False
    return update.effective_chat.type == constants.ChatType.PRIVATE


def split_into_chunks(text: str, chunk_size: int = 4096) -> list[str]:
    """
    Splits a string into chunks of a given size.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


async def wrap_with_indicator(
    update: Update,
    context: CallbackContext,
    coroutine,
    chat_action: constants.ChatAction = '',
    is_inline=False,
):
    """
    Wraps a coroutine while repeatedly sending a chat action to the user.
    """
    task = context.application.create_task(coroutine(), update=update)
    while not task.done():
        if not is_inline:
            context.application.create_task(
                update.effective_chat.send_action(chat_action, message_thread_id=get_forum_thread_id(update))
            )
        try:
            await asyncio.wait_for(asyncio.shield(task), 4.5)
        except asyncio.TimeoutError:
            pass


async def edit_message_with_retry(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int | None,
    message_id: str,
    text: str,
    markdown: bool = True,
    is_inline: bool = False,
):
    """
    Edit a message with retry logic in case of failure (e.g. broken markdown)
    :param context: The context to use
    :param chat_id: The chat id to edit the message in
    :param message_id: The message id to edit
    :param text: The text to edit the message with
    :param markdown: Whether to use markdown parse mode
    :param is_inline: Whether the message to edit is an inline message
    :return: None
    """
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(message_id) if not is_inline else None,
            inline_message_id=message_id if is_inline else None,
            text=text,
            parse_mode=constants.ParseMode.HTML if markdown else None,
            disable_web_page_preview=True,
        )
    except telegram.error.BadRequest as e:
        if str(e).startswith('Message is not modified'):
            return
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(message_id) if not is_inline else None,
                inline_message_id=message_id if is_inline else None,
                text=text,
            )
        except Exception as e:
            logging.warning(f'Failed to edit message: {str(e)}')
            raise e

    except Exception as e:
        logging.warning(str(e))
        raise e


async def error_handler(_: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles errors in the telegram-python-bot library.
    """
    logging.error(f'Exception while handling an update: {context.error}', exc_info=context.error)


async def is_allowed(config, update: Update, context: CallbackContext, is_inline=False) -> bool:
    """
    Checks if the user is allowed to use the bot.
    """
    if config['allowed_user_ids'] == '*':
        return True

    user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id
    if is_admin(config, user_id):
        return True
    name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
    allowed_user_ids = config['allowed_user_ids'].split(',')
    # Check if user is allowed
    if str(user_id) in allowed_user_ids:
        return True
    # Check if it's a group a chat with at least one authorized member
    if not is_inline and is_group_chat(update):
        admin_user_ids = config['admin_user_ids'].split(',')
        for user in itertools.chain(allowed_user_ids, admin_user_ids):
            if not user.strip():
                continue
            if await is_user_in_group(update, context, user):
                logging.info(f'{user} is a member. Allowing group chat message...')
                return True
        logging.info(f'Group chat messages from user {name} ' f'(id: {user_id}) are not allowed')
    return False


def is_admin(config, user_id: int, log_no_admin=False) -> bool:
    """
    Checks if the user is the admin of the bot.
    The first user in the user list is the admin.
    """
    if config['admin_user_ids'] == '-':
        if log_no_admin:
            logging.info('No admin user defined.')
        return False

    admin_user_ids = config['admin_user_ids'].split(',')

    # Check if user is in the admin user list
    if str(user_id) in admin_user_ids:
        return True

    return False


def has_image_gen_permission(config, user_id: int) -> bool:
    if config['img_gen_access_user_ids'] == '-':
        return False

    img_gen_access_user_ids = config['img_gen_access_user_ids'].split(',')
    if str(user_id) in img_gen_access_user_ids:
        return True

    return False


def get_user_budget(config, user_id) -> float | None:
    """
    Get the user's budget based on their user ID and the bot configuration.
    :param config: The bot configuration object
    :param user_id: User id
    :return: The user's budget as a float, or None if the user is not found in the allowed user list
    """

    # no budget restrictions for admins and '*'-budget lists
    if is_admin(config, user_id) or config['user_budgets'] == '*':
        return float('inf')

    user_budgets = config['user_budgets'].split(',')
    if config['allowed_user_ids'] == '*':
        # same budget for all users, use value in first position of budget list
        if len(user_budgets) > 1:
            logging.warning(
                'multiple values for budgets set with unrestricted user list '
                'only the first value is used as budget for everyone.'
            )
        return float(user_budgets[0])

    allowed_user_ids = config['allowed_user_ids'].split(',')
    if str(user_id) in allowed_user_ids:
        user_index = allowed_user_ids.index(str(user_id))
        if len(user_budgets) <= user_index:
            logging.warning(f'No budget set for user id: {user_id}. Budget list shorter than user list.')
            return 0.0
        return float(user_budgets[user_index])
    return None


def get_remaining_budget(config, usage, update: Update, is_inline=False) -> float:
    """
    Calculate the remaining budget for a user based on their current usage.
    :param config: The bot configuration object
    :param usage: The usage tracker object
    :param update: Telegram update object
    :param is_inline: Boolean flag for inline queries
    :return: The remaining budget for the user as a float
    """
    # Mapping of budget period to cost period
    budget_cost_map = {
        'monthly': 'cost_month',
        'daily': 'cost_today',
        'all-time': 'cost_all_time',
    }

    user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id
    name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
    if user_id not in usage:
        usage[user_id] = UsageTracker(user_id, name)

    # Get budget for users
    user_budget = get_user_budget(config, user_id)
    budget_period = config['budget_period']
    if user_budget is not None:
        cost = usage[user_id].get_current_cost()[budget_cost_map[budget_period]]
        return user_budget - cost

    # Get budget for guests
    if 'guests' not in usage:
        usage['guests'] = UsageTracker('guests', 'all guest users in group chats')
    cost = usage['guests'].get_current_cost()[budget_cost_map[budget_period]]
    return config['guest_budget'] - cost


def is_within_budget(config, usage, update: Update, is_inline=False) -> bool:
    """
    Checks if the user reached their usage limit.
    Initializes UsageTracker for user and guest when needed.
    :param config: The bot configuration object
    :param usage: The usage tracker object
    :param update: Telegram update object
    :param is_inline: Boolean flag for inline queries
    :return: Boolean indicating if the user has a positive budget
    """
    user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id
    name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
    if user_id not in usage:
        usage[user_id] = UsageTracker(user_id, name)
    remaining_budget = get_remaining_budget(config, usage, update, is_inline=is_inline)
    return remaining_budget > 0


def add_chat_request_to_usage_tracker(usage, config, user_id, used_tokens):
    """
    Add chat request to usage tracker
    :param usage: The usage tracker object
    :param config: The bot configuration object
    :param user_id: The user id
    :param used_tokens: The number of tokens used
    """
    try:
        if int(used_tokens) == 0:
            logging.warning('No tokens used. Not adding chat request to usage tracker.')
            return
        # add chat request to users usage tracker
        usage[user_id].add_chat_tokens(used_tokens, config['token_price'])
        # add guest chat request to guest usage tracker
        allowed_user_ids = config['allowed_user_ids'].split(',')
        if str(user_id) not in allowed_user_ids and 'guests' in usage:
            usage['guests'].add_chat_tokens(used_tokens, config['token_price'])
    except Exception as e:
        logging.warning(f'Failed to add tokens to usage_logs: {str(e)}')
        pass


def get_reply_to_message_id(config, update: Update):
    """
    Returns the message id of the message to reply to
    :param config: Bot configuration object
    :param update: Telegram update object
    :return: Message id of the message to reply to, or None if quoting is disabled
    """
    if config['enable_quoting'] or is_group_chat(update):
        return update.message.message_id
    return None


def is_direct_result(response: any) -> bool:
    """
    Checks if the dict contains a direct result that can be sent directly to the user
    :param response: The response value
    :return: Boolean indicating if the result is a direct result
    """
    if not isinstance(response, dict):
        try:
            json_response = json.loads(response)
            return json_response.get('direct_result', False)
        except:
            return False
    else:
        return response.get('direct_result', False)


async def handle_direct_result(config, update: Update, response: any, save_reply: Optional[Callable] = None):
    """
    Handles a direct result from a plugin
    """
    if not isinstance(response, dict):
        response = json.loads(response)

    result = response['direct_result']
    kind = result['kind']

    common_args = {
        'message_thread_id': get_forum_thread_id(update),
        'reply_to_message_id': get_reply_to_message_id(config, update),
    }

    sent_msg = None
    if kind == 'photo':
        sent_msg = await update.effective_message.reply_photo(
            **common_args, photo=result['photo'], caption=result.get('caption')
        )
    elif kind == 'album':
        media = [telegram.InputMediaPhoto(media=photo) for photo in result['photos']]
        if 'caption' in result:
            media[0] = telegram.InputMediaPhoto(media=result['photos'][0], caption=result['caption'])

        sent_msgs = await update.effective_message.reply_media_group(**common_args, media=media)
        sent_msg = sent_msgs[-1] if sent_msgs else None
    elif kind in {'gif', 'file', 'document'}:
        sent_msg = await update.effective_message.reply_document(
            **common_args, document=result['document'], caption=result.get('caption')
        )
    elif kind == 'reaction':
        await update.effective_message.set_reaction(reaction=result['reaction'])
    elif kind == 'voice':
        sent_msg = await update.effective_message.reply_voice(
            **common_args, voice=result['voice'], caption=result.get('caption')
        )
    elif kind == 'video_note':
        sent_msg = await update.effective_message.reply_video_note(
            **common_args,
            video_note=result['video_note'],
        )
    elif kind == 'video':
        sent_msg = await update.effective_message.reply_video(
            **common_args, video=result['video'], caption=result.get('caption')
        )
    elif kind == 'audio':
        sent_msg = await update.effective_message.reply_audio(
            **common_args, audio=result['audio'], caption=result.get('caption')
        )
    elif kind == 'dice':
        sent_msg = await update.effective_message.reply_dice(**common_args, emoji=result['emoji'])
    elif kind == 'poll':
        sent_msg = await update.effective_message.reply_poll(
            **common_args,
            question=result['question'],
            options=result['options'],
            is_anonymous=result.get('is_anonymous', False),
            allows_multiple_answers=result.get('allows_multiple_answers', False),
        )
    elif kind == 'location':
        sent_msg = await update.effective_message.reply_location(
            **common_args,
            latitude=result['latitude'],
            longitude=result['longitude'],
        )
    elif kind == 'venue':
        sent_msg = await update.effective_message.reply_venue(
            **common_args,
            latitude=result['latitude'],
            longitude=result['longitude'],
            title=result['title'],
            address=result['address'],
        )
    elif kind == 'contact':
        sent_msg = await update.effective_message.reply_contact(
            **common_args,
            phone_number=result['phone_number'],
            first_name=result['first_name'],
            last_name=result['last_name'],
        )

    if save_reply and sent_msg:
        save_reply(sent_msg, update)


# Function to encode the image
def encode_image(fileobj):
    image = base64.b64encode(fileobj.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{image}'


def decode_image(imgbase64):
    image = imgbase64[len('data:image/jpeg;base64,') :]
    return base64.b64decode(image)
