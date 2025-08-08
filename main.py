# main.py
import streamlit as st
import os
from tempfile import NamedTemporaryFile
from rag import process_urls, generate_answer, process_docs

st.title('Real Estate Research Tool')

# ------------------ URL Processing ------------------
st.sidebar.header("📡 Process URLs")
url1 = st.sidebar.text_input('Enter URL 1')
url2 = st.sidebar.text_input('Enter URL 2')
url3 = st.sidebar.text_input('Enter URL 3')

placeholder = st.empty()
process_url_button = st.sidebar.button('Process URLs')
if process_url_button:
    urls = [url for url in [url1, url2, url3] if url != ""]
    if len(urls) == 0:
        placeholder.text("You must provide at least one Valid URL.")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

# ------------------ Document Upload Processing ------------------
# ---------------- Document Upload Processing ----------------
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Docs"):
    if not uploaded_files:
        placeholder.warning("Please upload at least one document.")
    else:
        temp_files = []
        for uploaded_file in uploaded_files:
            with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.read())
                temp_files.append((tmp.name, uploaded_file.name))
        for status in process_docs(temp_files):
            placeholder.info(status)


# ------------------ Query Section ------------------
# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

query = placeholder.text_input("Ask a Question")
if query:
    try:
        answer, sources = generate_answer(query)
        # st.header("Answer:")
        # st.write(answer)
        # if sources:
        #     st.subheader("Sources:")
        #     for source in sources.split("/n"):
        #         st.write(source)
        st.session_state.history.append((query, answer, sources))
    except Exception as e:
        placeholder.text("You must process URLs or upload documents first.")


# Display conversation
for q, a, s in st.session_state.history:
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 Assistant:** {a}")

    if s:
        st.markdown(f"**Sources:** {s}")

    # if docs:
    #     with st.expander("📄 View Retrieved Chunks"):
    #         for doc in docs:
    #             text = doc.page_content
    #             # Optional: naive highlight of query terms
    #             for term in q.split():
    #                 text = text.replace(term, f"**{term}**")
    #             st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
    #             st.markdown(text)
    st.markdown("---")
