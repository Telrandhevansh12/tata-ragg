import hashlib
import heapq
import io
import math
import os
import pickle
import re
from pathlib import Path

import fitz  # PyMuPDF
import ollama
import pdfplumber
import streamlit as st


st.set_page_config(page_title="Local Document and Image RAG Brain", page_icon="PDF", layout="wide")
st.title("Local Document and Image RAG System")
st.write("Drag and drop PDFs or images to extract text, tables, and rich visual descriptions.")


EMBEDDING_MODEL = "nomic-embed-text"
LANGUAGE_MODEL = "llama3.2:1b"
VISION_MODEL = "llava"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CLIENT = ollama.Client(host=OLLAMA_HOST)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
EMBEDDING_BATCH_SIZE = 16
TOP_MATCHES = 6
CACHE_FILE = Path("vector_db.pkl")
SUPPORTED_IMAGE_TYPES = {"png", "jpg", "jpeg", "webp"}
SUPPORTED_DOCUMENT_TYPES = {"pdf"}
SUPPORTED_FILE_TYPES = sorted(SUPPORTED_DOCUMENT_TYPES | SUPPORTED_IMAGE_TYPES)

VISION_PROMPT = """
Analyze this PDF page for a strict document question-answering system.

Be factual. Do not guess names, numbers, labels, or values that are not clearly visible.

Include:
- Page type: mostly text, scanned page, form, chart, diagram, photo, screenshot, table, or mixed
- Main visual elements and where they appear on the page
- Exact visible titles, labels, captions, legends, axes, and values when readable
- Chart or table meaning, including trends and comparisons only when visible
- Diagram relationships, arrows, process steps, or object connections when visible
- Logos, seals, signatures, stamps, icons, or screenshots
- Any uncertainty, blur, cut-off content, or unreadable text

Write the answer in this structure:
Page summary:
Visible text and labels:
Images/charts/diagrams/tables:
Important details for Q&A:
Uncertain or unreadable:
"""


def init_state():
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = load_vector_db()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = load_processed_files(st.session_state.vector_db)
    if "file_store" not in st.session_state:
        st.session_state.file_store = {}
    if "top_matches" not in st.session_state:
        st.session_state.top_matches = TOP_MATCHES
    if "min_similarity" not in st.session_state:
        st.session_state.min_similarity = 0.15


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
        if len(entry) == 6:
            normalized_db.append(entry)
        elif len(entry) == 5:
            file_hash, file_name, chunk, embedding, embedding_norm = entry
            normalized_db.append((file_hash, file_name, None, chunk, embedding, embedding_norm))
        elif len(entry) == 4:
            file_name, chunk, embedding, embedding_norm = entry
            normalized_db.append(
                (f"legacy:{file_name}", file_name, None, chunk, embedding, embedding_norm)
            )
        elif len(entry) == 3:
            file_name, chunk, embedding = entry
            normalized_db.append(
                (f"legacy:{file_name}", file_name, None, chunk, embedding, vector_norm(embedding))
            )
        elif len(entry) == 2:
            chunk, embedding = entry
            normalized_db.append(
                ("legacy:unknown", "Unknown file", None, chunk, embedding, vector_norm(embedding))
            )

    return normalized_db


def save_vector_db():
    try:
        with CACHE_FILE.open("wb") as file:
            pickle.dump(st.session_state.vector_db, file)
    except Exception as error:
        st.warning(f"Could not save vector database cache: {error}")


def load_processed_files(vector_db):
    return sorted({file_hash for file_hash, _, _, _, _, _ in vector_db})


def display_processed_files(vector_db):
    files = {}
    for file_hash, file_name, _, _, _, _ in vector_db:
        files[file_hash] = file_name
    return files


def delete_indexed_file(file_hash):
    st.session_state.vector_db = [
        entry for entry in st.session_state.vector_db if entry[0] != file_hash
    ]
    st.session_state.processed_files = [
        existing_hash
        for existing_hash in st.session_state.processed_files
        if existing_hash != file_hash
    ]
    st.session_state.file_store.pop(file_hash, None)
    save_vector_db()


def file_sha256(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def tokenize(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def get_installed_ollama_models():
    try:
        response = OLLAMA_CLIENT.list()
    except Exception as error:
        return None, str(error)

    models = response.get("models", [])
    names = set()
    for model in models:
        model_name = model.get("name") or model.get("model")
        if model_name:
            names.add(model_name)
            names.add(model_name.split(":")[0])

    return names, None


def display_model_status(selected_answer_model, needs_vision):
    installed_models, error = get_installed_ollama_models()
    if error:
        st.error(f"Could not connect to Ollama at {OLLAMA_HOST}: {error}")
        return

    required_models = [EMBEDDING_MODEL, selected_answer_model]
    if needs_vision:
        required_models.append(VISION_MODEL)

    missing_models = [
        model for model in required_models
        if model not in installed_models and model.split(":")[0] not in installed_models
    ]

    if missing_models:
        st.warning("Missing Ollama model(s): " + ", ".join(missing_models))
        st.code("\n".join(f"ollama pull {model}" for model in missing_models))
    else:
        st.success("Required Ollama models are available.")


def render_source_preview(file_hash, file_name, page_number):
    stored_file = st.session_state.file_store.get(file_hash)
    if not stored_file:
        return

    file_extension = Path(file_name).suffix.lower().lstrip(".")
    file_bytes = stored_file["bytes"]

    if file_extension in SUPPORTED_IMAGE_TYPES:
        st.image(file_bytes, caption=file_name, use_container_width=True)
        return

    if file_extension == "pdf" and page_number:
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                page = doc[page_number - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                st.image(
                    pix.tobytes("png"),
                    caption=f"{file_name}, page {page_number}",
                    use_container_width=True,
                )
        except Exception as error:
            st.caption(f"Could not preview source page: {error}")


def describe_page_image(file_name, page_num, page, page_text):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image_bytes = pix.tobytes("png")
    extracted_text_hint = page_text[:2_000] if page_text else "No selectable text was extracted."

    response = OLLAMA_CLIENT.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{VISION_PROMPT}\n\n"
                    "Selectable text extracted from this same page, if any:\n"
                    f"{extracted_text_hint}\n\n"
                    "Use the page image as the primary source. Use the extracted text only "
                    "to confirm readable labels and reduce mistakes."
                ),
                "images": [image_bytes],
            }
        ],
    )
    image_description = response["message"]["content"].strip()

    return (
        f"[Visual Description: {file_name}, Page: {page_num}]\n"
        f"{image_description}"
    )


def describe_uploaded_image(file_name, image_bytes):
    response = OLLAMA_CLIENT.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{VISION_PROMPT}\n\n"
                    "This uploaded file is a standalone image, not a PDF page. "
                    "Use the image as the source and describe all readable visual details."
                ),
                "images": [image_bytes],
            }
        ],
    )
    image_description = response["message"]["content"].strip()

    return f"[Image Description: {file_name}]\n{image_description}"


def extract_everything_from_bytes(file_name, file_bytes, enable_tables, enable_vision):
    all_content = []

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                all_content.append(f"[Source: {file_name}, Page: {page_num}]\n{page_text}")

            if enable_vision:
                try:
                    all_content.append(describe_page_image(file_name, page_num, page, page_text))
                except Exception as error:
                    st.warning(
                        f"Could not describe image content in {file_name}, page {page_num}: {error}"
                    )
                    if page_text:
                        all_content.append(
                            f"[Image Description Fallback: {file_name}, Page: {page_num}]\n"
                            f"The vision model could not analyze this page. Text found on the page:\n"
                            f"{page_text}"
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


def extract_image_from_bytes(file_name, file_bytes):
    try:
        return describe_uploaded_image(file_name, file_bytes)
    except Exception as error:
        st.warning(f"Could not describe image file {file_name}: {error}")
        return ""


def chunk_text(text, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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
            step = max(max_chars - overlap, 1)
            paragraph = paragraph[step:].strip()

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
            page_number = extract_page_number(chunk)
            st.session_state.vector_db.append(
                (file_hash, file_name, page_number, chunk, embedding, vector_norm(embedding))
            )
            added_count += 1

        processed_count += len(batch)
        if progress_bar:
            progress_bar.progress(
                min((progress_start + processed_count) / max(progress_total, 1), 1.0)
            )

    return added_count


def extract_page_number(chunk):
    markers = ["Page:", "Table Page:"]
    for marker in markers:
        if marker in chunk:
            after_marker = chunk.split(marker, 1)[1]
            page_text = after_marker.split("]", 1)[0].strip()
            if page_text.isdigit():
                return int(page_text)
    return None


def retrieve(query, top_n=TOP_MATCHES, min_similarity=0.0):
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
    query_tokens = tokenize(query)

    def score(entry):
        file_hash, file_name, page_number, chunk, embedding, embedding_norm = entry
        if embedding_norm == 0:
            return 0, file_hash, file_name, page_number, chunk

        dot_product = sum(x * y for x, y in zip(query_embedding, embedding))
        similarity = dot_product / (query_norm * embedding_norm)
        chunk_tokens = tokenize(chunk)
        keyword_score = 0
        if query_tokens and chunk_tokens:
            keyword_score = len(query_tokens & chunk_tokens) / len(query_tokens)
        final_score = similarity + (0.08 * keyword_score)
        return final_score, file_hash, file_name, page_number, chunk

    matches = heapq.nlargest(
        top_n,
        (score(entry) for entry in st.session_state.vector_db),
        key=lambda item: item[0],
    )
    return [match for match in matches if match[0] >= min_similarity]


init_state()


with st.sidebar:
    st.header("File Control Center")
    uploaded_files = st.file_uploader(
        "Drag and drop PDF or image files here:",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help="Supported files: PDF, PNG, JPG, JPEG, and WebP.",
    )

    processing_mode = st.selectbox(
        "Processing Mode",
        ["Fast: text only", "Balanced: text + tables", "Detailed: text + tables + images"],
        help="Detailed mode describes images, charts, diagrams, screenshots, and logos.",
    )

    selected_answer_model = st.selectbox(
        "Answer Model",
        ["llama3.2:1b", "qwen3.5:9b"],
        help="The smaller model is faster. The larger model usually gives better answers.",
    )
    LANGUAGE_MODEL = selected_answer_model

    st.session_state.top_matches = st.slider(
        "Source chunks used for answers",
        min_value=3,
        max_value=12,
        value=st.session_state.top_matches,
        help="Higher values give the answer model more evidence but can be slower.",
    )
    st.session_state.min_similarity = st.slider(
        "Minimum source score",
        min_value=0.0,
        max_value=0.6,
        value=st.session_state.min_similarity,
        step=0.05,
        help="Raise this to ignore weak matches. Lower it if answers miss relevant files.",
    )

    extract_tables = processing_mode in {
        "Balanced: text + tables",
        "Detailed: text + tables + images",
    }
    use_vision = processing_mode == "Detailed: text + tables + images"

    if processing_mode == "Fast: text only":
        st.caption("Fast upload tip: PDFs use text only. Image files still need image analysis.")
    elif processing_mode == "Balanced: text + tables":
        st.warning("Table extraction can take longer for large PDFs. Image files use image analysis.")
    else:
        st.warning(
            "Detailed mode is slow because each PDF page is analyzed by the vision model."
        )
        st.caption("Make sure Ollama has the vision model installed: ollama pull llava")

    uploaded_image_present = any(
        Path(uploaded_file.name).suffix.lower().lstrip(".") in SUPPORTED_IMAGE_TYPES
        for uploaded_file in uploaded_files or []
    )
    with st.expander("Model status"):
        display_model_status(selected_answer_model, use_vision or uploaded_image_present)

    st.metric("Loaded files", len(st.session_state.processed_files))
    st.metric("Total chunks", len(st.session_state.vector_db))

    if uploaded_files:
        prepared_uploads = []
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_hash = file_sha256(file_bytes)
            st.session_state.file_store[file_hash] = {
                "name": uploaded_file.name,
                "bytes": file_bytes,
            }
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
                    file_extension = Path(file_name).suffix.lower().lstrip(".")
                    st.write(f"Extracting: {file_name}")

                    if file_extension in SUPPORTED_IMAGE_TYPES:
                        extracted_text = extract_image_from_bytes(file_name, file_bytes)
                    else:
                        extracted_text = extract_everything_from_bytes(
                            file_name,
                            file_bytes,
                            extract_tables,
                            use_vision,
                        )

                    chunks = chunk_text(extracted_text)
                    if chunks:
                        prepared_files.append((file_hash, file_name, chunks))
                        total_chunks += len(chunks)
                    else:
                        st.warning(f"No searchable content was extracted from {file_name}.")

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
            file_extension = Path(filename).suffix.lower().lstrip(".") or "file"
            file_columns = st.columns([0.18, 0.58, 0.24])
            file_columns[0].caption(file_extension.upper())
            file_columns[1].write(filename)
            if file_columns[2].button("Delete", key=f"delete-{file_hash}"):
                delete_indexed_file(file_hash)
                st.rerun()

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


if input_query := st.chat_input("Ask a question about your uploaded PDFs or images..."):
    with st.chat_message("user"):
        st.write(input_query)

    st.session_state.chat_history.append({"role": "user", "content": input_query})

    if not st.session_state.vector_db:
        with st.chat_message("assistant"):
            st.warning("Please upload and index PDF or image files using the sidebar first.")
    else:
        retrieved_knowledge = retrieve(
            input_query,
            top_n=st.session_state.top_matches,
            min_similarity=st.session_state.min_similarity,
        )

        if retrieved_knowledge:
            with st.expander("View pulled document source context"):
                for similarity, file_hash, file_name, page_number, chunk in retrieved_knowledge:
                    page_label = f", Page: {page_number}" if page_number else ""
                    st.write(
                        f"**Score: {similarity:.2f} | File: {file_name}{page_label}**"
                    )
                    render_source_preview(file_hash, file_name, page_number)
                    st.code(chunk, language="markdown")

            context_text = "\n".join(
                f" - Source file: {file_name}"
                f"{f', page: {page_number}' if page_number else ''}\n{chunk}"
                for similarity, file_hash, file_name, page_number, chunk in retrieved_knowledge
            )

            instruction_prompt = f"""
You are a helpful chatbot.
Use only the following document context to answer the question.
Do not make up new information, numbers, names, labels, dates, or visual details.
If the answer is not clearly supported by the context, say that the uploaded documents do not provide enough evidence.
When images, charts, diagrams, screenshots, logos, or visual layouts are relevant,
describe them using the visual descriptions in the context.
Prefer exact wording from the context for labels and values.
Always mention the source file and page number when available.

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
