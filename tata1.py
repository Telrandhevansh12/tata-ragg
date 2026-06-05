dataset = []
with open('cat-facts.txt', 'r' , encoding='utf-8') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')

import ollama

EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'llama3.2:1b'
# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a,b):
  dot_product = sum([x * y for x, y in zip(a,b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([y ** 2 for y in b]) ** 0.5
  return dot_product / (norm_a * norm_b) 

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((similarity, chunk))
    similarities.sort(key=lambda x : x[0], reverse=True)
    return similarities[:top_n]
 
input_query = input('ask me question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for similarity, chunk in retrieved_knowledge:
    print(f'- (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = f''' You are a helpful CHATBOT. 
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for similarity, chunk in retrieved_knowledge])}
'''
stream = ollama.chat(
   model=LANGUAGE_MODEL,
    messages=[
         {'role': 'system', 'content' : instruction_prompt},
         {'role': 'user', 'content' : input_query},
    ],
    stream=True,
)

print('Chatbot response:')
for chunk in stream:
   print(chunk['message']['content'], end='', flush=True)

   