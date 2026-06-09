import hashlib
import heapq
import io
import math
import os
import pickle
from pathlib import Path

import fitz  # PyMuPDF
import ollama
import pdfplumber
import streamlit as st


st.set_page_config(page_title="Local Multi-PDF RAG Brain", page_icon="PDF", layout="wide")
st.title("Local Multi-PDF Intelligent RAG System")
st.write("Upload multiple PDFs to extract text, tables, and optional visual context.")


EMBEDDING_MODEL = "nomic-embed-text"
LANGUAGE_MODEL = "llama3.2:1b"
VISION_MODEL = "llava"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CLIENT = ollama.Client(host=OLLAMA_HOST)

CHUNK_SIZE = 1_200
EMBEDDING_BATCH_SIZE = 16
TOP_MATCHES = 3
CACHE_FILE = Path("vector_db.pkl")


def init_state():
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = load_vector_db()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = load_processed_files(st.session_state.vector_db)


def load_vector_db():
    if not CACHE_FILE.exists():
        return []

    try:
        with CACHE_FILE.open("rb") as file:
            db = pickle.load(file)
    except Exception:
        return []

    normalized_db = []
    for entry in db:
        if len(entry) == 5:
            normalized_db.append(entry)
        elif len(entry) == 4:
            file_name, chunk, embedding, embedding_norm = entry
            normalized_db.append(
                (f"legacy:{file_name}", file_name, chunk, embedding, embedding_norm)
            )
        elif len(entry) == 3:
            file_name, chunk, embedding = entry
            normalized_db.append(
                (f"legacy:{file_name}", file_name, chunk, embedding, vector_norm(embedding))
            )
        elif len(entry) == 2:
            chunk, embedding = entry
            normalized_db.append(
                ("legacy:unknown", "Unknown file", chunk, embedding, vector_norm(embedding))
            )

    return normalized_db


def save_vector_db():
    try:
        with CACHE_FILE.open("wb") as file:
            pickle.dump(st.session_state.vector_db, file)
    except Exception as error:
        st.warning(f"Could not save vector database cache: {error}")


def load_processed_files(vector_db):
    return sorted({file_hash for file_hash, _, _, _, _ in vector_db})


def display_processed_files(vector_db):
    files = {}
    for file_hash, file_name, _, _, _ in vector_db:
        files[file_hash] = file_name
    return files


def file_sha256(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def extract_everything_from_bytes(file_name, file_bytes, enable_tables, enable_vision):
    all_content = []

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                all_content.append(f"[Source: {file_name}, Page: {page_num}]\n{page_text}")

            if enable_vision:
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
                image_bytes = pix.tobytes("png")

                try:
                    response = OLLAMA_CLIENT.chat(
                        model=VISION_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "Describe this PDF page for a retrieval system. "
                                    "Focus on visible charts, drawings, text, logos, and structure."
                                ),
                                "images": [image_bytes],
                            }
                        ],
                    )
                    image_description = response["message"]["content"]
                    all_content.append(
                        f"[Visual Context From File: {file_name}, Page: {page_num}]\n{image_description}"
                    )
                except Exception:
                    if page_text:
                        all_content.append(
                            f"[Image Text Fallback From File: {file_name}, Page: {page_num}]\n{page_text}"
                        )

    if enable_tables:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables():
                    markdown_rows = []
                    for row in table:
                        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                        markdown_rows.append("| " + " | ".join(cleaned_row) + " |")

                    if markdown_rows:
                        table_text = "\n".join(markdown_rows)
                        all_content.append(f"[Source: {file_name}, Table Page: {page_num}]\n{table_text}")

    return "\n\n".join(all_content)


def chunk_text(text, max_chars=CHUNK_SIZE):
    chunks = []
    seen = set()
    current_chunk = []
    current_size = 0

    def add_chunk(chunk):
        chunk = chunk.strip()
        fingerprint = " ".join(chunk.lower().split())
        if fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            chunks.append(chunk)

    def flush_current_chunk():
        nonlocal current_chunk, current_size
        if current_chunk:
            add_chunk("\n\n".join(current_chunk))
            current_chunk = []
            current_size = 0

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if current_chunk and current_size + len(paragraph) > max_chars:
            flush_current_chunk()

        while len(paragraph) > max_chars:
            add_chunk(paragraph[:max_chars])
            paragraph = paragraph[max_chars:].strip()

        if paragraph:
            current_chunk.append(paragraph)
            current_size += len(paragraph) + 2

    flush_current_chunk()
    return chunks


def add_chunks_to_database(
    file_hash,
    file_name,
    chunks,
    progress_bar=None,
    progress_start=0,
    progress_total=1,
):
    added_count = 0
    processed_count = 0

    for batch in batched(chunks, EMBEDDING_BATCH_SIZE):
        try:
            response = OLLAMA_CLIENT.embed(model=EMBEDDING_MODEL, input=batch)
            embeddings = response.get("embeddings", [])
        except Exception as error:
            st.error(f"Failed to generate embeddings: {error}")
            continue

        if len(embeddings) != len(batch):
            st.warning("Ollama returned fewer embeddings than expected for one batch.")

        for chunk, embedding in zip(batch, embeddings):
            st.session_state.vector_db.append(
                (file_hash, file_name, chunk, embedding, vector_norm(embedding))
            )
            added_count += 1

        processed_count += len(batch)
        if progress_bar:
            progress_bar.progress(
                min((progress_start + processed_count) / max(progress_total, 1), 1.0)
            )

    return added_count


def retrieve(query, top_n=TOP_MATCHES):
    if not st.session_state.vector_db:
        return []

    try:
        response = OLLAMA_CLIENT.embed(model=EMBEDDING_MODEL, input=str(query).strip())
        embeddings = response.get("embeddings", [])
        if not embeddings:
            st.error("Ollama failed to generate an embedding for your query.")
            return []
        query_embedding = embeddings[0]
    except Exception as error:
        st.error(f"Error connecting to embedding model during retrieval: {error}")
        return []

    query_norm = vector_norm(query_embedding)
    if query_norm == 0:
        return []

    def score(entry):
        _, file_name, chunk, embedding, embedding_norm = entry
        if embedding_norm == 0:
            return 0, file_name, chunk

        dot_product = sum(x * y for x, y in zip(query_embedding, embedding))
        similarity = dot_product / (query_norm * embedding_norm)
        return similarity, file_name, chunk

    return heapq.nlargest(
        top_n,
        (score(entry) for entry in st.session_state.vector_db),
        key=lambda item: item[0],
    )


init_state()


with st.sidebar:
    st.header("Document Control Center")
    uploaded_files = st.file_uploader(
        "Upload your PDF files here:",
        type=["pdf"],
        accept_multiple_files=True,
    )

    processing_mode = st.selectbox(
        "Processing Mode",
        ["Fast: text only", "Balanced: text + tables", "Detailed: text + tables + vision"],
        help="Fast mode is best for uploading many PDFs at once.",
    )

    selected_answer_model = st.selectbox(
        "Answer Model",
        ["llama3.2:1b", "qwen3.5:9b"],
        help="The smaller model is faster. The larger model usually gives better answers.",
    )
    LANGUAGE_MODEL = selected_answer_model

    extract_tables = processing_mode in {
        "Balanced: text + tables",
        "Detailed: text + tables + vision",
    }
    use_vision = processing_mode == "Detailed: text + tables + vision"

    if processing_mode == "Fast: text only":
        st.caption("Fast upload tip: this mode skips slow table and image analysis.")
    elif processing_mode == "Balanced: text + tables":
        st.warning("Table extraction can take longer for large PDFs.")
    else:
        st.warning("Detailed mode is slow because each page image is analyzed by the vision model.")

    st.metric("Loaded files", len(st.session_state.processed_files))
    st.metric("Total chunks", len(st.session_state.vector_db))

    if uploaded_files:
        prepared_uploads = []
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_hash = file_sha256(file_bytes)
            if file_hash not in st.session_state.processed_files:
                prepared_uploads.append((uploaded_file.name, file_hash, file_bytes))

        if not prepared_uploads:
            st.info("All uploaded files are already indexed.")

        if prepared_uploads and st.button("Process & Index Documents"):
            total_added = 0

            with st.spinner("Processing documents into vector database..."):
                prepared_files = []
                total_chunks = 0

                for file_name, file_hash, file_bytes in prepared_uploads:
                    st.write(f"Extracting: {file_name}")
                    extracted_text = extract_everything_from_bytes(
                        file_name,
                        file_bytes,
                        extract_tables,
                        use_vision,
                    )
                    chunks = chunk_text(extracted_text)
                    prepared_files.append((file_hash, file_name, chunks))
                    total_chunks += len(chunks)

                progress_bar = st.progress(0)
                completed_chunks = 0

                for file_index, (file_hash, file_name, chunks) in enumerate(prepared_files, start=1):
                    st.write(
                        f"Embedding file {file_index} of {len(prepared_files)}: "
                        f"{file_name} ({len(chunks)} chunks)"
                    )
                    added_count = add_chunks_to_database(
                        file_hash,
                        file_name,
                        chunks,
                        progress_bar=progress_bar,
                        progress_start=completed_chunks,
                        progress_total=total_chunks,
                    )
                    completed_chunks += len(chunks)
                    total_added += added_count

                    if added_count:
                        st.session_state.processed_files.append(file_hash)

                save_vector_db()
                progress_bar.progress(1.0)

            st.success(
                f"Indexed {total_added} new chunks. "
                f"Total chunks in memory: {len(st.session_state.vector_db)}"
            )

    if st.session_state.processed_files:
        st.write("---")
        st.write("**Currently Loaded Files:**")
        processed_file_names = display_processed_files(st.session_state.vector_db)
        for file_hash in st.session_state.processed_files:
            filename = processed_file_names.get(file_hash, file_hash[:12])
            st.write(f"- {filename}")

    if st.button("Clear App Memory"):
        st.session_state.vector_db = []
        st.session_state.chat_history = []
        st.session_state.processed_files = []
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        st.rerun()


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if input_query := st.chat_input("Ask a question about your uploaded knowledge files..."):
    with st.chat_message("user"):
        st.write(input_query)

    st.session_state.chat_history.append({"role": "user", "content": input_query})

    if not st.session_state.vector_db:
        with st.chat_message("assistant"):
            st.warning("Please upload and index PDF files using the sidebar first.")
    else:
        retrieved_knowledge = retrieve(input_query)

        if retrieved_knowledge:
            with st.expander("View pulled document source context"):
                for similarity, file_name, chunk in retrieved_knowledge:
                    st.write(f"**Similarity: {similarity:.2f} | File: {file_name}**")
                    st.code(chunk, language="markdown")

            context_text = "\n".join(
                f" - Source file: {file_name}\n{chunk}"
                for similarity, file_name, chunk in retrieved_knowledge
            )

            instruction_prompt = f"""
You are a helpful chatbot.
Use only the following document context to answer the question.
Do not make up new information.
Always mention the source file or page when available.

Context:
{context_text}
"""

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                try:
                    stream = OLLAMA_CLIENT.chat(
                        model=LANGUAGE_MODEL,
                        messages=[
                            {"role": "system", "content": instruction_prompt},
                            {"role": "user", "content": input_query},
                        ],
                        stream=True,
                    )

                    for chunk in stream:
                        full_response += chunk["message"]["content"]
                        response_placeholder.markdown(full_response + "|")

                    response_placeholder.markdown(full_response)
                except Exception as error:
                    full_response = f"Could not generate an answer: {error}"
                    response_placeholder.error(full_response)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": full_response}
            )
        else:
            with st.chat_message("assistant"):
                st.warning("I could not find relevant context for that question.")
