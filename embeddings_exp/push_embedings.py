import logging

from parse_tg_export import parse_tg_export
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def group_messages_with_overlap(messages, window_minutes=5, step_minutes=2, min_messages=5, min_new_messages=2):
    """Group messages with a sliding window approach as a generator.

    Args:
        messages: List of message dictionaries with 'timestamp' keys
        window_minutes: Base window size in minutes
        step_minutes: Time to advance the window in minutes
        min_messages: Minimum number of messages per window
        min_new_messages: Minimum number of new messages required to create a new window
    """
    if not messages:
        return

    messages_sorted = sorted(messages, key=lambda m: m['timestamp'])

    # Convert window and step to seconds for integer timestamp operations
    window_seconds = window_minutes * 60
    step_seconds = step_minutes * 60

    # Initialize window indices
    start_idx = 0
    end_idx = 0
    last_window_end_idx = 0

    # Advance the window while there are messages to process
    with tqdm(total=len(messages_sorted), desc='Grouping messages', unit='window') as pbar:
        while start_idx < len(messages_sorted):
            # Get timestamp of the start message in this window
            window_start_time = messages_sorted[start_idx]['timestamp']
            window_end_time = window_start_time + window_seconds

            # Find the end of the time window
            while end_idx < len(messages_sorted) and messages_sorted[end_idx]['timestamp'] < window_end_time:
                end_idx += 1

            # Ensure minimum number of messages in the window
            if end_idx - start_idx < min_messages and end_idx < len(messages_sorted):
                end_idx = min(start_idx + min_messages, len(messages_sorted))

            # Create a window only if we have enough new messages since the last window
            if end_idx - last_window_end_idx >= min_new_messages:
                group = messages_sorted[start_idx:end_idx]
                if group:
                    yield group
                    last_window_end_idx = end_idx

                    # Update progress bar
                    pbar.update(end_idx - last_window_end_idx + min_new_messages)

            # Advance start_idx by step_time or at least one message
            next_start_time = window_start_time + step_seconds
            while (
                start_idx < len(messages_sorted)
                and start_idx < end_idx
                and messages_sorted[start_idx]['timestamp'] < next_start_time
            ):
                start_idx += 1

            # If we didn't advance, move to the next message to avoid being stuck
            if start_idx == last_window_end_idx:
                start_idx += 1


def create_conversation_block(message_group):
    """Create a conversation block from a group of messages."""
    # Sort messages by timestamp to maintain conversation flow
    sorted_messages = sorted(message_group, key=lambda m: m['timestamp'])
    conversation = []

    for message in sorted_messages:
        username = message['username'] or f"User{message['user_id']}"
        conversation.append(f"{username}: {message['text']}")

    return '\n'.join(conversation)


# Load chat data from Telegram export
logger.info('Loading chat data from Telegram export...')

chat_dump = parse_tg_export('chat_dataset/result.json')

# load only few messages for testing
chat_dump = chat_dump[:100000]

logger.info(f'Loaded {len(chat_dump)} messages from export')

# Load the sentence transformer model
logger.info('Loading sentence transformer model...')
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

# Group messages and create conversation blocks
logger.info('Grouping messages into conversation blocks...')
valid_messages = [entry for entry in chat_dump if entry['text'].strip()]
message_groups_generator = group_messages_with_overlap(valid_messages, window_minutes=5, step_minutes=2)

# Create conversation blocks from message groups
logger.info('Creating conversation blocks...')
conversation_blocks = []
for group in tqdm(message_groups_generator, desc='Creating blocks'):
    block = create_conversation_block(group)
    if block:
        conversation_blocks.append(
            {
                'text': block,
                'timestamp': group[0]['timestamp'],
                'message_count': len(group),
                'users': list(set(m['username'] or f"User{m['user_id']}" for m in group)),
            }
        )

logger.info(f'Created {len(conversation_blocks)} conversation blocks')

# Generate embeddings for conversation blocks
logger.info('Generating embeddings for conversation blocks...')
block_texts = [block['text'] for block in conversation_blocks]
# embeddings = model.encode(block_texts, show_progress_bar=True)
embeddings = model.encode(block_texts, show_progress_bar=True, normalize_embeddings=False)
logger.info(f'Generated embeddings with shape: {embeddings.shape}')

# Connect to local Qdrant
logger.info('Connecting to Qdrant...')
client = QdrantClient(host='localhost', port=6333)

collection_name = 'telegram_conversation_embeddings'

# Create collection if it doesn't exist
if collection_name not in [c.name for c in client.get_collections().collections]:
    logger.info(f'Creating new collection: {collection_name}')
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
    )

# Prepare and upload points in batches
logger.info('Starting batched upload to Qdrant...')
batch_size = 1000
current_batch = []

for idx, (block, vector) in enumerate(
    tqdm(zip(conversation_blocks, embeddings), total=len(conversation_blocks), desc='Uploading to Qdrant')
):
    point = PointStruct(
        id=idx,
        vector=vector.tolist(),
        payload={
            'text': block['text'],
            'timestamp': block['timestamp'],
            'message_count': block['message_count'],
            'users': block['users'],
        },
    )
    current_batch.append(point)

    # When batch is full or this is the last item, upload the batch
    if len(current_batch) >= batch_size or idx == len(conversation_blocks) - 1:
        client.upsert(collection_name=collection_name, points=current_batch)
        current_batch = []  # Clear the batch after upload

logger.info(f'Successfully inserted {len(conversation_blocks)} conversation blocks with embeddings into Qdrant.')
