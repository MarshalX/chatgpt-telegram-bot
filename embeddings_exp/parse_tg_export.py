import json
import logging
import time
from typing import Any, Dict, List

# Configure logging with more detailed format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def parse_tg_export(filename: str) -> List[Dict[str, Any]]:
    """
    Parse Telegram export JSON file into a list of messages.

    Args:
        filename: Path to the Telegram export JSON file

    Returns:
        List of parsed messages
    """
    start_time = time.time()
    logger.info(f'Starting to parse Telegram export from {filename}')

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            logger.debug('Reading JSON file...')
            data = json.load(f)
            logger.info(f"Successfully loaded JSON file with {len(data.get('messages', []))} total entries")
    except FileNotFoundError:
        logger.error(f'File not found: {filename}')
        raise
    except json.JSONDecodeError:
        logger.error(f'Invalid JSON format in file: {filename}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error while reading file: {str(e)}')
        raise

    messages = []
    processed_count = 0
    skipped_count = 0

    for msg in data.get('messages', []):
        # Only process normal messages
        if msg.get('type') != 'message':
            skipped_count += 1
            continue

        # Extract user_id and username
        user_id = msg.get('from_id')
        username = msg.get('from')

        # Extract timestamp (as int if possible)
        timestamp = msg.get('date_unixtime')
        if timestamp is not None:
            try:
                timestamp = int(timestamp)
            except Exception as e:
                logger.warning(f'Failed to convert timestamp to int: {e}')

        # Extract and combine text
        text = msg.get('text', '')
        if isinstance(text, str):
            message_text = text
        elif isinstance(text, list):
            # Each element can be a string or dict with "text"
            parts = []
            for part in text:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get('text', ''))
            message_text = ''.join(parts)
        else:
            message_text = ''

        messages.append({'user_id': user_id, 'username': username, 'timestamp': timestamp, 'text': message_text})
        processed_count += 1

        # Log progress for every 1000 messages
        if processed_count % 1000 == 0:
            logger.info(f'Processed {processed_count} messages...')

    end_time = time.time()
    processing_time = end_time - start_time

    logger.info(f'Parsing completed in {processing_time:.2f} seconds')
    logger.info(f'Total messages processed: {processed_count}')
    logger.info(f'Messages skipped (non-message type): {skipped_count}')

    return messages


if __name__ == '__main__':
    try:
        result = parse_tg_export('chat_dataset/result.json')
        logger.info(f'Successfully extracted {len(result)} messages')
    except Exception as e:
        logger.error(f'Failed to parse Telegram export: {str(e)}')
