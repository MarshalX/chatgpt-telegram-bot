from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def print_result(result):
    print('\nTop results:')

    # Handle different response formats
    if hasattr(result, 'points'):
        # QueryResponse object
        points = result.points
    elif isinstance(result, dict) and 'points' in result:
        # Dictionary with 'points' key
        points = result['points']
    else:
        # Assume it's a list or similar iterable
        points = result

    if not points:
        print('No results found.')
        return

    for i, point in enumerate(points, 1):
        # Handle different point formats
        if hasattr(point, 'payload'):
            payload = point.payload
            score = getattr(point, 'score', 0.0)
        elif isinstance(point, dict):
            payload = point.get('payload', {})
            score = point.get('score', 0.0)
        else:
            print(f'{i}. [Format unknown] Unable to display this result')
            print(f'   Point type: {type(point)}')
            print(f'   Point data: {point}')
            print('-' * 60)
            continue

        print(f"{i}. [{payload.get('username', 'Unknown')}] {payload.get('text', '')}")
        print(f"   user_id: {payload.get('user_id')}, timestamp: {payload.get('timestamp')}")
        print(f'   Score: {score:.4f}')
        print('-' * 60)


def main():
    # Connect to Qdrant
    client = QdrantClient(host='localhost', port=6333)
    collection_name = 'telegram_chat_embeddings'

    # Load the same embedding model as used for indexing
    print('Loading embedding model...')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("Qdrant CLI search. Type your query and press Enter. Type 'exit' to quit.")
    while True:
        try:
            query = input('\nSearch query: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting.')
            break
        if query.lower() in ('exit', 'quit', 'q'):
            print('Goodbye!')
            break
        if not query:
            continue

        # Embed the query
        # query_vector = model.encode([query])[0].tolist()
        query_vector = model.encode(query).tolist()

        # Search in Qdrant
        try:
            hits = client.query_points(collection_name=collection_name, query=query_vector, limit=10, with_payload=True)
            # Debug output to see what we got
            # print(f"Type of hits: {type(hits)}")
            # print(f"Structure of hits: {hits}")
            # if hasattr(hits, 'points'):
            #     print(f"Type of hits.points: {type(hits.points)}")
            #     print(f"Structure of first point: {hits.points[0] if hits.points else 'None'}")
            # elif isinstance(hits, list) and hits:
            #     print(f"Type of first item: {type(hits[0])}")
            #     print(f"Structure of first item: {hits[0]}")
        except Exception as e:
            print(f'Error during search: {e}')
            continue

        if not hits:
            print('No results found.')
        else:
            print_result(hits)


if __name__ == '__main__':
    main()
