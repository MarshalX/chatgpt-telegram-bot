import logging

from parse_tg_export import parse_tg_export
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load chat data from Telegram export
logger.info('Loading chat data from Telegram export...')

chat_dump = parse_tg_export('chat_dataset/result.json')
# load only few messages for testing
chat_dump = chat_dump[:10000]

logger.info(f'Loaded {len(chat_dump)} messages from export')


# Load the sentence transformer model
logger.info('Loading sentence transformer model...')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Extract messages for embedding
messages = [entry['text'] for entry in chat_dump if entry['text'].strip()]  # Only include non-empty messages
logger.info(f'Extracted {len(messages)} non-empty messages for embedding')

# Generate embeddings
logger.info('Generating embeddings...')
embeddings = model.encode(messages)
logger.info(f'Generated embeddings with shape: {embeddings.shape}')

# Connect to local Qdrant
logger.info('Connecting to Qdrant...')
client = QdrantClient(host='localhost', port=6333)

collection_name = 'telegram_chat_embeddings'

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
valid_message_count = 0
current_batch = []

for idx, (entry, vector) in enumerate(zip(chat_dump, embeddings)):
    if not entry['text'].strip():
        continue

    point = PointStruct(
        id=idx,
        vector=vector.tolist(),
        payload={
            'user_id': entry['user_id'],
            'username': entry['username'],
            'timestamp': entry['timestamp'],
            'text': entry['text'],
        },
    )
    current_batch.append(point)
    valid_message_count += 1

    # When batch is full or this is the last item, upload the batch
    if len(current_batch) >= batch_size or idx == len(chat_dump) - 1:
        logger.info(
            f'Uploading batch of {len(current_batch)} points to Qdrant... (Total progress: {valid_message_count}/{len(chat_dump)})'
        )
        client.upsert(collection_name=collection_name, points=current_batch)
        current_batch = []  # Clear the batch after upload

logger.info(f'Successfully inserted {valid_message_count} chat messages with embeddings into Qdrant.')
