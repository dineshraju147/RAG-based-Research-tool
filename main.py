import streamlit as st
from rag import process_urls , generate_answer, process_docs
from tempfile import NamedTemporaryFile


st.title('Research Assistant Tool')

# ------------------ URL Processing ------------------
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
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    'Upload a PDF or TXT file',
    type=['pdf', 'txt'],
    accept_multiple_files=True
)
process_docs_button = st.sidebar.button('Process Docs')
if process_docs_button:
    if not uploaded_files:
        placeholder.text("You must provide at least one document.")
    else:
        temp_paths = []
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp:
                tmp.write(uploaded_file.read())
                temp_paths.append(tmp.name)
        for status in process_docs(temp_paths):
            placeholder.text(status)


query = placeholder.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)
        if sources:
            st.subheader("Sources:")
            for source in sources.split("/n"):
                st.write(source)
    except Exception as e:
        placeholder.text("You must process urls first")

