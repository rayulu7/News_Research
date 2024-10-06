import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (API key)
load_dotenv()

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Get up to 3 URLs from the user
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")  # Button to trigger URL processing
file_path = "faiss_store_openai.pkl"  # Path to store the FAISS index 

main_placeholder = st.empty()  # Placeholder for main content area

# Set OpenAI model with free-tier limits
llm = OpenAI(
    temperature=0.7,
    max_tokens=100,  # Lower token limit to avoid hitting free-tier restrictions
    model_name="text-davinci-003"  # Ensuring we use a free-tier compatible model
)

if process_url_clicked and urls:
    try:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Loading data...")  # Placeholder text while loading data
        data = loader.load()

        # Split the data into smaller documents
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=500  # Smaller chunks to fit API limits
        )
        docs = text_splitter.split_documents(data)
        main_placeholder.text("Splitting text into smaller chunks...")

        # Create embeddings and build FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        pkl = vectorstore_openai.serialize_to_bytes()
        main_placeholder.text("Building embeddings...")

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(pkl, f)
        main_placeholder.text("Processing complete. You can now ask questions.")

    except Exception as e:
        # Check for specific error related to quota
        if "quota" in str(e).lower() or "insufficient_quota" in str(e).lower():
            st.error("You've reached your daily API data limit. Try again tomorrow or upgrade your plan.")  # Updated error message
        else:
            st.error(f"Error occurred while processing URLs: {e}")

# Input field for user queries
query = main_placeholder.text_input("Ask your question:")

# Process the user's query and provide answers
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                pkl = pickle.load(f)

            # Deserialize FAISS index and set up retrieval chain
            vectorstore = FAISS.deserialize_from_bytes(
                embeddings=OpenAIEmbeddings(), 
                serialized=pkl, 
                allow_dangerous_deserialization=True
            )
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            
            # Get the answer from the chain
            result = chain({"question": query}, return_only_outputs=True)
            answer = result.get("answer", "No answer found.")
            sources = result.get("sources", "")

            # Display the answer
            st.header("Answer")
            st.write(answer)

            # Display sources, if any
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)

        except Exception as e:
            # Check for specific error related to quota
            if "quota" in str(e).lower() or "insufficient_quota" in str(e).lower():
                st.error("API data usage limit reached; please upgrade or retry after quota reset.")  # Updated error message
            else:
                st.error(f"Error occurred while processing query: {e}")
    else:
        st.error("No data available. Please process the URLs first.")
