import time

from sentence_transformers import SentenceTransformer

sentences = ['This is an example sentence', 'Each sentence is converted']

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

start = time.time()

embeddings = model.encode(sentences)
print(embeddings)

end = time.time()
print(f'Time taken: {end - start} seconds')
