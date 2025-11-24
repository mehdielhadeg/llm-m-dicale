# imports
import os
import streamlit as st
from pathlib import Path
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage

# CONFIG 
DOCS_FOLDER = "documents"
VECTOR_FOLDER = "vector_store"
os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# MODELS
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    hf_token = os.getenv("HF_TOKEN") or st.sidebar.text_input(
        "Token Hugging Face", type="password", key="hf"
    )
    if not hf_token:
        st.stop()

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        max_new_tokens=1024,
    )
    return embeddings, ChatHuggingFace(llm=llm)


embeddings, chat_model = load_models()

# VECTORSTORE 
vectorstore = None
if os.path.exists(os.path.join(VECTOR_FOLDER, "faiss.index")):
    try:
        vectorstore = FAISS.load_local(
            VECTOR_FOLDER, embeddings, "faiss.index", allow_dangerous_deserialization=True
        )
    except:
        st.warning("Index corrompu → recréation à la prochaine ingestion")


#  PDF → TEXT 
def pdf_to_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


#  INGEST 
def ingest_pdf(file_path):
    global vectorstore
    if vectorstore:
        if any(file_path.name in d.metadata.get("source", "") for d in vectorstore.docstore._dict.values()):
            return f"Déjà chargé : {file_path.name}"

    text = pdf_to_text(file_path)
    if not text.strip():
        return "PDF vide"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"source": file_path.name}) for c in chunks]

    if not vectorstore:
        vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        vectorstore.add_documents(docs)

    vectorstore.save_local(VECTOR_FOLDER, "faiss.index")
    return f"Chargé : {file_path.name} ({len(docs)} morceaux)"


# CONVERSATION BUILDER 
def build_messages(prompt, context):
    messages = [
        SystemMessage(
            content=(
                "Tu es un assistant médical expert. "
                "Réponds uniquement en français et uniquement à partir des documents fournis. "
                "Si l'information n'est pas dans les documents, réponds : "
                "'Je ne trouve pas cette information dans les documents chargés.'"
            )
        )
    ]

    # Ajout de l'historique conversationnel (10 derniers messages)
    for m in st.session_state.messages[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})

    # Ajouter le prompt enrichi par le contexte RAG
    enriched_prompt = f"Contexte documentaire :\n{context}\n\nQuestion : {prompt}"
    messages.append({"role": "user", "content": enriched_prompt})

    return messages


#  UI 
st.set_page_config(page_title="DocQA-MS", page_icon="doctor", layout="centered")
st.title("DocQA-MS — Assistant Médical")

with st.sidebar:
    st.header("Documents")
    up = st.file_uploader("Ajouter PDFs", type="pdf", accept_multiple_files=True)
    if up:
        for f in up:
            p = Path(DOCS_FOLDER) / f.name
            p.write_bytes(f.getbuffer())
            with st.spinner(f"{f.name}..."):
                st.success(ingest_pdf(p))

    if st.button("Recharger tous les PDFs"):
        for p in Path(DOCS_FOLDER).glob("*.pdf"):
            ingest_pdf(p)
        st.success("OK")
        st.rerun()

    if st.button("Nouvelle conversation", type="primary"):
        st.session_state.messages = []
        st.rerun()

    if vectorstore:
        srcs = len({d.metadata["source"] for d in vectorstore.docstore._dict.values()})
        chunks = vectorstore.index.ntotal
        st.metric("Documents", srcs)
        st.metric("Morceaux", chunks)
    else:
        st.info("Aucun document")


#  CHAT 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources : {', '.join(msg['sources'])}")


# Question utilisateur
if prompt := st.chat_input("Votre question médicale..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            if not vectorstore:
                reponse = "Aucun document chargé."
                sources = []
            else:
                # RAG = recherche contextuelle
                docs_scores = vectorstore.similarity_search_with_score(prompt, k=8)
                relevant = [doc for doc, score in docs_scores if score < 0.75]

                if not relevant:
                    relevant = [doc for doc, _ in docs_scores[:5]]

                context = "\n\n".join([d.page_content for d in relevant])
                sources = list({d.metadata["source"] for d in relevant})

                # Construction des messages conversationnels
                messages = build_messages(prompt, context)

                # Appel LLM
                answer = chat_model.invoke(messages).content.strip()
                reponse = answer or "Aucune réponse générée."

            st.markdown(reponse)
            if sources:
                st.caption(f"Sources : {', '.join(sources)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": reponse,
        "sources": sources if sources else None
    })
