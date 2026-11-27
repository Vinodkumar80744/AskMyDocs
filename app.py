
import shutil
from pathlib import Path
from html import escape

import streamlit as st

from src.embeddings.embedder import Embedder
from src.generation.llm import GroqLLM
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.pdf_loader import PDFLoader
from src.rag.pipeline import RAGPipeline
from src.retrieval.vector_store import VectorStore


DOCUMENTS_PATH = Path("data/documents")
VECTORSTORE_PATH = Path("vectorstore")
STAGING_PATH = Path("vectorstore_staging")
BACKUP_PATH = Path("vectorstore_backup")

TOP_K = 5
SCORE_THRESHOLD = 0.25

st.set_page_config(
    page_title="RAG Studio",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F8FC;
}

header[data-testid="stHeader"] {
    background: #F7F8FC;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263244;
    min-width: 270px;
}

[data-testid="stSidebar"] > div:first-child {
    background: #111827;
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #1F2937;
    border: 1px solid #374151;
    padding: 12px;
    border-radius: 12px;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #FFFFFF;
}

[data-testid="stSidebar"] .stButton button {
    background: #263244;
    color: #FFFFFF;
    border: 1px solid #374151;
    border-radius: 10px;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: #3730A3;
    border-color: #818CF8;
}

.brand {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #FFFFFF;
}

.brand-caption {
    font-size: 12px;
    color: #9CA3AF;
    margin-top: 5px;
}

.sidebar-label {
    font-size: 10px;
    font-weight: 700;
    color: #94A3B8;
    letter-spacing: 1.5px;
    margin-top: 28px;
    margin-bottom: 10px;
}

.main-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1.3px;
    color: #111827;
}

.main-subtitle {
    color: #64748B;
    font-size: 14px;
    margin-top: 5px;
    margin-bottom: 22px;
}

.status-card {
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    border-radius: 12px;
    padding: 13px 16px;
    color: #047857;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 24px;
}

.welcome-card {
    background: linear-gradient(135deg, #FFFFFF, #EEF2FF);
    border: 1px solid #E0E7FF;
    border-radius: 20px;
    padding: 35px 25px;
    text-align: center;
    margin: 20px 0;
}

.welcome-icon {
    font-size: 38px;
}

.welcome-title {
    color: #111827;
    font-size: 25px;
    font-weight: 800;
    margin-top: 12px;
}

.welcome-description {
    color: #64748B;
    font-size: 14px;
    line-height: 1.7;
    max-width: 650px;
    margin: 10px auto 0 auto;
}

div[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E8ECF4;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(15,23,42,0.025);
}

div[data-testid="stChatMessage"] p {
    color: #1E293B;
    font-size: 14px;
    line-height: 1.8;
}

div[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE2EE !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 18px rgba(15,23,42,0.06) !important;
}

div[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #111827 !important;
    caret-color: #4F46E5 !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #94A3B8 !important;
}

div[data-testid="stBottom"],
div[data-testid="stBottom"] > div {
    background: #F7F8FC !important;
}

div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
}

.stButton button {
    border-radius: 11px;
    font-weight: 600;
    min-height: 42px;
}

.page-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 11px;
    padding: 30px 0 5px;
}
</style>
""", unsafe_allow_html=True)


def recover_vectorstore():

    if BACKUP_PATH.exists() and not VECTORSTORE_PATH.exists():
        BACKUP_PATH.rename(VECTORSTORE_PATH)

    if BACKUP_PATH.exists() and VECTORSTORE_PATH.exists():
        shutil.rmtree(BACKUP_PATH)


@st.cache_resource
def initialize_rag():

    recover_vectorstore()

    embedder = Embedder()
    llm = GroqLLM()

    vectorstore_exists = (
        (VECTORSTORE_PATH / "index.faiss").exists()
        and (VECTORSTORE_PATH / "documents.json").exists()
    )

    if vectorstore_exists:
        vector_store = VectorStore.load(str(VECTORSTORE_PATH))
    else:
        ingestion = IngestionPipeline(embedder=embedder)

        vector_store = ingestion.ingest_directory(
            directory=str(DOCUMENTS_PATH),
            vectorstore_path=str(VECTORSTORE_PATH)
        )

    rag = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm
    )

    return rag, vector_store.index.ntotal


def rebuild_vectorstore():

    if STAGING_PATH.exists():
        shutil.rmtree(STAGING_PATH)

    if BACKUP_PATH.exists():
        shutil.rmtree(BACKUP_PATH)

    embedder = Embedder()
    ingestion = IngestionPipeline(embedder=embedder)

    try:
        new_store = ingestion.ingest_directory(
            directory=str(DOCUMENTS_PATH),
            vectorstore_path=str(STAGING_PATH)
        )

        if VECTORSTORE_PATH.exists():
            VECTORSTORE_PATH.rename(BACKUP_PATH)

        try:
            STAGING_PATH.rename(VECTORSTORE_PATH)
        except Exception:
            if BACKUP_PATH.exists() and not VECTORSTORE_PATH.exists():
                BACKUP_PATH.rename(VECTORSTORE_PATH)
            raise

        if BACKUP_PATH.exists():
            shutil.rmtree(BACKUP_PATH)

        return new_store.index.ntotal

    except Exception:
        if STAGING_PATH.exists():
            shutil.rmtree(STAGING_PATH)

        if BACKUP_PATH.exists() and not VECTORSTORE_PATH.exists():
            BACKUP_PATH.rename(VECTORSTORE_PATH)

        raise


def refresh_rag():

    initialize_rag.clear()

    st.session_state.messages = []
    st.session_state.pending_question = None

    st.rerun()


def display_sources(sources):

    if not sources:
        st.info("No relevant source documents found.")
        return

    with st.expander(f"📑 View {len(sources)} source references"):

        for index, source in enumerate(sources, start=1):

            source_name = escape(str(source["source"]))

            st.markdown(
                f"""
                <div class="page-card">
                    <div style="color:#4F46E5;font-size:11px;font-weight:700;">
                        SOURCE {index}
                    </div>
                    <div style="color:#111827;font-weight:600;margin-top:9px;">
                        📄 {source_name}
                    </div>
                    <div style="color:#64748B;font-size:13px;margin-top:8px;">
                        Page: {source['page']} &nbsp; · &nbsp;
                        Chunk: {source['chunk']}
                    </div>
                    <div style="color:#64748B;font-size:12px;margin-top:5px;">
                        Similarity score: {source['score']:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_assistant(rag):

    st.markdown(
        '<div class="main-title">AI Knowledge Assistant</div>'
        '<div class="main-subtitle">'
        'Ask questions and discover insights from your documents.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="status-card">'
        '● &nbsp; SYSTEM OPERATIONAL &nbsp; · &nbsp; '
        'Your knowledge base is ready'
        '</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.messages:

        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-icon">✦</div>
                <div class="welcome-title">
                    Your knowledge, intelligently connected.
                </div>
                <div class="welcome-description">
                    Ask anything about your documents.
                    Receive AI-generated answers with references
                    to the original PDF sources.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("##### Suggested questions")

        suggestions = [
            "What is Python?",
            "What is machine learning?",
            "Explain database systems.",
            "What is Retrieval-Augmented Generation?"
        ]

        columns = st.columns(2)

        for index, suggestion in enumerate(suggestions):

            with columns[index % 2]:

                if st.button(
                    suggestion,
                    key=f"suggestion_{index}",
                    use_container_width=True
                ):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":
                display_sources(message.get("sources", []))

    question = st.chat_input(
        "Ask anything about your documents..."
    )

    if question is None and st.session_state.pending_question:

        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching your knowledge base..."):

                try:
                    result = rag.ask(
                        question,
                        top_k=TOP_K,
                        score_threshold=SCORE_THRESHOLD
                    )

                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)
                    display_sources(sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as error:

                    error_message = f"Unable to generate an answer: {error}"

                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    })


def render_documents():

    DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)

    st.markdown(
        '<div class="main-title">Document Library</div>'
        '<div class="main-subtitle">'
        'Upload, manage and remove PDF documents from your knowledge base.'
        '</div>',
        unsafe_allow_html=True
    )

    documents = sorted(DOCUMENTS_PATH.glob("*.pdf"))

    st.metric("Available PDF documents", len(documents))

    st.markdown("### 📤 Upload PDF documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF files."
    )

    if st.button(
        "Upload and Index Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files
    ):

        saved_files = []

        try:
            for uploaded_file in uploaded_files:

                filename = Path(uploaded_file.name).name
                destination = DOCUMENTS_PATH / filename
                file_bytes = uploaded_file.getvalue()

                if not file_bytes.startswith(b"%PDF-"):
                    st.warning(f"{filename}: This does not appear to be a valid PDF.")
                    continue

                if destination.exists():
                    st.warning(f"{filename}: A document with this name already exists.")
                    continue

                temporary_file = DOCUMENTS_PATH / f".upload_{filename}"

                try:
                    temporary_file.write_bytes(file_bytes)

                    loaded_pages = PDFLoader().load(str(temporary_file))

                    if not loaded_pages:
                        st.warning(f"{filename}: No readable pages were found.")
                        continue

                    destination.write_bytes(file_bytes)
                    saved_files.append(destination)

                    st.success(f"Uploaded: {filename}")

                finally:
                    if temporary_file.exists():
                        temporary_file.unlink()

            if saved_files:

                with st.spinner("Rebuilding the knowledge base. Please wait..."):

                    total_vectors = rebuild_vectorstore()

                st.success(
                    f"Successfully indexed {len(saved_files)} PDF(s). "
                    f"Total vectors: {total_vectors}"
                )

                refresh_rag()

            else:
                st.info("No new documents were uploaded.")

        except Exception as error:

            for saved_file in saved_files:
                if saved_file.exists():
                    saved_file.unlink()

            st.error(f"Upload or indexing failed: {error}")
            st.warning("Your previous knowledge base has been preserved.")

    st.divider()

    st.markdown("### 📑 Existing documents")

    documents = sorted(DOCUMENTS_PATH.glob("*.pdf"))

    if not documents:
        st.info("No PDF documents found.")
        return

    for document in documents:

        st.markdown(
            f"""
            <div class="page-card">
                <div style="font-size:16px;font-weight:600;color:#111827;">
                    📄 {escape(document.name)}
                </div>
                <div style="font-size:12px;color:#64748B;margin-top:7px;">
                    PDF document · Stored in your document library
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("###  🚮 Remove a document")

    selected_document = st.selectbox(
        "Select a PDF to remove",
        options=[document.name for document in documents],
        key="delete_document_selection"
    )

    confirm_delete = st.checkbox(
        f"I confirm that I want to permanently remove {selected_document}.",
        key="confirm_document_deletion"
    )

    if st.button(
        "Delete Document and Re-index",
        type="primary",
        use_container_width=True,
        disabled=not confirm_delete
    ):

        if len(documents) <= 1:
            st.error(
                "At least one PDF must remain in the knowledge base. "
                "Upload another PDF before deleting this one."
            )
            return

        document_path = DOCUMENTS_PATH / selected_document
        backup_document = DOCUMENTS_PATH / f".deleting_{selected_document}"

        try:
            document_path.rename(backup_document)

            try:
                with st.spinner("Rebuilding the knowledge base..."):

                    total_vectors = rebuild_vectorstore()

                backup_document.unlink()

                st.success(
                    f"{selected_document} was deleted successfully. "
                    f"Remaining vectors: {total_vectors}"
                )

                refresh_rag()

            except Exception:
                if backup_document.exists():
                    backup_document.rename(document_path)
                raise

        except Exception as error:
            st.error(f"Document deletion failed: {error}")
            st.warning("The document and previous knowledge base were restored.")


def render_knowledge_base(total_vectors):

    st.markdown(
        '<div class="main-title">Knowledge Base</div>'
        '<div class="main-subtitle">'
        'Overview of your document retrieval system.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Stored vectors", total_vectors)

    with col2:
        st.metric(
            "PDF documents",
            len(list(DOCUMENTS_PATH.glob("*.pdf")))
        )

    st.markdown("### Technology stack")

    technologies = [
        ("Embeddings", "Sentence Transformers · all-MiniLM-L6-v2"),
        ("Vector database", "FAISS · Inner Product Search"),
        ("Language model", "Groq API"),
        ("PDF processing", "PyMuPDF"),
        ("Application", "Python · Streamlit")
    ]

    for name, description in technologies:

        st.markdown(
            f"""
            <div class="page-card">
                <div style="font-weight:700;color:#4F46E5;">
                    {name}
                </div>
                <div style="color:#334155;font-size:14px;margin-top:6px;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def main():

    DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "page" not in st.session_state:
        st.session_state.page = "AI Assistant"

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    try:
        with st.spinner("Initializing RAG Studio..."):
            rag, total_vectors = initialize_rag()

    except Exception as error:
        st.error(f"Initialization failed: {error}")
        st.stop()

    with st.sidebar:

        st.markdown(
            """
            <div class="brand">✦ RAG STUDIO</div>
            <div class="brand-caption">
                Intelligent Knowledge Assistant
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-label">WORKSPACE</div>',
            unsafe_allow_html=True
        )

        page = st.radio(
            "Navigation",
            [
                "AI Assistant",
                "Document Library",
                "Knowledge Base"
            ],
            key="page",
            label_visibility="collapsed"
        )

        st.markdown(
            '<div class="sidebar-label">KNOWLEDGE BASE</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "PDF Documents",
                len(list(DOCUMENTS_PATH.glob("*.pdf")))
            )

        with col2:
            st.metric("Vectors", total_vectors)

        st.markdown(
            '<div class="sidebar-label">SYSTEM STATUS</div>',
            unsafe_allow_html=True
        )

        st.success("AI Engine Connected")

        st.caption("Groq · FAISS · Sentence Transformers")

        st.divider()

        if st.button(
            " 🚮 Clear Conversation",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

        st.markdown(
            """
            <div style="
                color:#9CA3AF;
                font-size:11px;
                margin-top:35px;
                line-height:1.8;
            ">
                RAG STUDIO<br>
                Document Intelligence Platform<br>
                Built with Python and Streamlit
            </div>
            """,
            unsafe_allow_html=True
        )

    if page == "AI Assistant":
        render_assistant(rag)

    elif page == "Document Library":
        render_documents()

    elif page == "Knowledge Base":
        render_knowledge_base(total_vectors)

    st.markdown(
        """
        <div class="footer">
            Powered by Groq AI · FAISS · Sentence Transformers
            <br>
            RAG Studio © 2026
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
