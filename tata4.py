import ollama
import pdfplumber
import fitz  # PyMuPDF for extracting images
import io

# =====================================================================
# 1. THE COMPLETE CONTENT EXTRACTION FUNCTION (TEXT + TABLES + IMAGES)
# =====================================================================
def extract_everything_from_pdf(pdf_path):
    all_content = []
    
    # --- PART A: Extract Text & Tables ---
    print("Extracting text and tables...")
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_content.append(page_text)
            
            tables = page.extract_tables()
            for table in tables:
                markdown_table = ""
                for row in table:
                    cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                    markdown_table += "| " + " | ".join(cleaned_row) + " |\n"
                if markdown_table:
                    all_content.append("\n" + markdown_table + "\n")

    # --- PART B: Extract Images & Generate Descriptions ---
    print("Checking for images...")
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        if image_list:
            print(f"Found {len(image_list)} image(s) on Page {page_num + 1}")
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            print(f" -> Asking llama3.2-vision to transcribe handwriting...")
            try:
                response = ollama.chat(
                    model='qwen3.5:9b',
                    messages=[{
                        'role': 'user',
                        'content': '''This image contains handwritten notes. 
                                      Your sole task is to act as an accurate OCR engine.
                                      Read the handwriting and transcribe it perfectly into clean digital text.
                                      Do not describe the color of the ink, the paper, or the layout. 
                                      Just output the transcribed words exactly as they are written.''',
                        'images': [image_bytes]
                    }]
                )
                image_description = response['message']['content']
                
                # Tag it as a handwritten note so the chatbot knows its source
                formatted_description = f"[HANDWRITTEN NOTE FROM PAGE {page_num+1}]:\n{image_description}"
                all_content.append(formatted_description)
            except Exception as e:
             print(f"Could not describe image: {e}. Make sure llama3.2-vision is downloaded.")
            
    return "\n\n".join(all_content)

# =====================================================================
# 2. RUN EXTRACTION & SMART CHUNKING
# =====================================================================
text = extract_everything_from_pdf('sample1.pdf')

# Smart paragraph and table chunking strategy
paragraphs = text.split("\n\n") 
dataset = []
current_chunk = ""

for para in paragraphs:
    if len(current_chunk) + len(para) < 1500: 
        current_chunk += para + "\n\n"
    else:
        if current_chunk:
            dataset.append(current_chunk.strip())
        current_chunk = para

if current_chunk:
    dataset.append(current_chunk.strip())

print(f'\nLoaded {len(dataset)} chunks into memory.')

# =====================================================================
# 3. VECTOR DATABASE SETUP & EMBEDDING
# =====================================================================
EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'llama3.2:1b'
VECTOR_DB = []

def add_chunk_to_database(chunk):
    # 1. If the chunk is empty or just blank spaces, skip it entirely!
    if not chunk or str(chunk).strip() == "":
        print("⚠️ Skipped an empty text chunk.")
        return 

    response = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
    
    # 2. Check if 'embeddings' actually came back with data before using [0]
    if 'embeddings' in response and len(response['embeddings']) > 0:
        embedding = response['embeddings'][0]
        VECTOR_DB.append((chunk, embedding))
    else:
        print(f"⚠️ Failed to generate embedding for this chunk.")
        return
    
print("Generating embeddings for database...")
for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f' Added chunk {i+1}/{len(dataset)} to the database')

# =====================================================================
# 4. SIMILARITY SEARCH & RETRIEVAL
# =====================================================================
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
 
# =====================================================================
# 5. INTERACTIVE CHATBOT EXTENSION
# =====================================================================
input_query = input('\nAsk me a question (about text, tables, or images): ')
retrieved_knowledge = retrieve(input_query)

print('\n--- Retrieved knowledge ---')
for similarity, chunk in retrieved_knowledge:
    print(f'- (similarity: {similarity:.2f})\n{chunk}\n')
print('---------------------------\n')

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
print() # Clean newline at the end 
