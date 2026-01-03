import streamlit as st
import time
import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# take environment variables from .env (especially openai api key)
load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
LLM_MODEL= "gemini-2.5-flash"
EMBEDDING_MODEL="gemini-embedding-001"
DB_DIR="local_faiss_index"

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
query=""
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

# Loading embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)

if process_url_clicked:
    query=""
    # Load the urls
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading the data...")
    docs = loader.load()

    # Split the docs
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ","],
        chunk_size=1000
    )
    main_placeholder.text("Splitting the data...")
    split_docs = text_splitter.split_documents(docs)

    # Creating the embeddings
    main_placeholder.text("Embedding the data...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    time.sleep(2)

    # Save the FAISS index
    main_placeholder.text("Saving.....")
    vectorstore.save_local(DB_DIR)
    main_placeholder.text("Pre processing complete...")

query = main_placeholder.text_input("Question: ")
if query:
    # Load the vectorstore
    loaded_vectorstore = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

    if loaded_vectorstore:
        # Create a retrieval chain
        # Create a retriever
        retriever = loaded_vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        # Run the chain with the query
        results = qa_chain.invoke(query)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(results["result"])

        # Display sources, if available
        sources = results.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)