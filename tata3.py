import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import ollama
import pickle
import os

# ==========================================
# LOAD AND CHUNK DATASET
# ==========================================

with open('cat-facts.txt', 'r', encoding='utf-8') as file:

    text = file.read()

# Better chunking
chunk_size = 300

dataset = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

print(f'Loaded {len(dataset)} chunks')

# ==========================================
# OLLAMA MODELS
# ==========================================

EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'llama3.2:1b'

# ==========================================
# VECTOR DATABASE
# ==========================================

VECTOR_DB = []
VECTOR_DB_FILE = 'vector_db.pkl'

# ==========================================
# CREATE EMBEDDINGS
# ==========================================

def add_chunk_to_database(chunk):

    embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunk
    )['embeddings'][0]

    VECTOR_DB.append((chunk, embedding))

# ==========================================
# LOAD OR CREATE VECTOR DATABASE
# ==========================================

if os.path.exists(VECTOR_DB_FILE):

    with open(VECTOR_DB_FILE, 'rb') as f:
        VECTOR_DB = pickle.load(f)

    print('Loaded saved vector database')

else:

    for i, chunk in enumerate(dataset):

        add_chunk_to_database(chunk)

        print(f'Added chunk {i+1}/{len(dataset)}')

    with open(VECTOR_DB_FILE, 'wb') as f:
        pickle.dump(VECTOR_DB, f)

    print('Vector database saved')

# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(a, b):

    dot_product = sum(
        [x * y for x, y in zip(a, b)]
    )

    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([y ** 2 for y in b]) ** 0.5

    return dot_product / (norm_a * norm_b)

# ==========================================
# RETRIEVAL FUNCTION
# ==========================================

def retrieve(query, top_n=5):

    query_embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )['embeddings'][0]

    similarities = []

    for chunk, embedding in VECTOR_DB:

        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        similarities.append((similarity, chunk))

    # Sort by similarity score
    similarities.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return similarities[:top_n]

# ==========================================
# CHAT MEMORY
# ==========================================

conversation_history = []

# ==========================================
# CHAT FUNCTION
# ==========================================

def ask_bot():

    input_query = entry.get().strip()

    if not input_query:
        return

    # Show user message
    chat_area.insert(
        tk.END,
        f'You: {input_query}\n\n'
    )

    # ==========================================
    # RETRIEVE KNOWLEDGE
    # ==========================================

    retrieved_knowledge = retrieve(input_query)

    chat_area.insert(
        tk.END,
        'Retrieved Knowledge:\n\n'
    )

    for similarity, chunk in retrieved_knowledge:

        chat_area.insert(
            tk.END,
            f'Similarity: {similarity:.2f}\n'
        )

        chat_area.insert(
            tk.END,
            f'{chunk}\n'
        )

        chat_area.insert(
            tk.END,
            '-' * 50 + '\n'
        )

    # ==========================================
    # CREATE RAG PROMPT
    # ==========================================

    instruction_prompt = f'''
You are a helpful CHATBOT.

Use ONLY the following context
to answer the question.

Do NOT make up information.

Context:

{chr(10).join([chunk for similarity, chunk in retrieved_knowledge])}
'''

    # ==========================================
    # SAVE USER MESSAGE
    # ==========================================

    conversation_history.append({
        'role': 'user',
        'content': input_query
    })

    # ==========================================
    # CREATE MESSAGES
    # ==========================================

    messages = [
        {
            'role': 'system',
            'content': instruction_prompt
        }
    ] + conversation_history

    # ==========================================
    # GENERATE RESPONSE
    # ==========================================

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=messages,
        stream=True,
    )

    chat_area.insert(
        tk.END,
        '\nBot: '
    )

    full_response = ''

    for chunk in stream:

        content = chunk['message']['content']

        full_response += content

        chat_area.insert(
            tk.END,
            content
        )

        chat_area.see(tk.END)

        root.update_idletasks()

    chat_area.insert(
        tk.END,
        '\n\n'
    )

    # ==========================================
    # SAVE ASSISTANT RESPONSE
    # ==========================================

    conversation_history.append({
        'role': 'assistant',
        'content': full_response
    })

    # Clear input box
    entry.delete(0, tk.END)

# ==========================================
# GUI SETUP
# ==========================================

root = tk.Tk()

root.title('Offline RAG Chatbot')

root.geometry('900x650')

root.configure(bg='#1e1e1e')

# ==========================================
# HEADER
# ==========================================

header = tk.Label(
    root,
    text='Offline RAG Chatbot',
    font=('Arial', 20, 'bold'),
    bg='#1e1e1e',
    fg='white'
)

header.pack(pady=10)

# ==========================================
# CHAT AREA
# ==========================================

chat_area = ScrolledText(
    root,
    wrap=tk.WORD,
    font=('Arial', 12),
    bg='#2b2b2b',
    fg='white',
    insertbackground='white'
)

chat_area.pack(
    padx=10,
    pady=10,
    fill=tk.BOTH,
    expand=True
)

# ==========================================
# INPUT FRAME
# ==========================================

input_frame = tk.Frame(
    root,
    bg='#1e1e1e'
)

input_frame.pack(
    fill=tk.X,
    padx=10,
    pady=10
)

# ==========================================
# INPUT BOX
# ==========================================

entry = tk.Entry(
    input_frame,
    font=('Arial', 14),
    bg='#2b2b2b',
    fg='white',
    insertbackground='white'
)

entry.pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    padx=(0, 10)
)

# ==========================================
# SEND BUTTON
# ==========================================

send_button = tk.Button(
    input_frame,
    text='Send',
    command=ask_bot,
    font=('Arial', 12, 'bold'),
    bg='#4CAF50',
    fg='white',
    padx=20,
    pady=5
)

send_button.pack(side=tk.RIGHT)

# ==========================================
# ENTER KEY SUPPORT
# ==========================================

root.bind(
    '<Return>',
    lambda event: ask_bot()
)

# ==========================================
# WELCOME MESSAGE
# ==========================================

chat_area.insert(
    tk.END,
    'Offline RAG Chatbot Ready!\n\n'
)

# ==========================================
# START GUI
# ==========================================

root.mainloop()