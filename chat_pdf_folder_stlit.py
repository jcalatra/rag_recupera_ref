import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
#from langchain_community.document_loaders import PdfReader
#from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# Configurar la clave API de OpenAI
load_dotenv()

# Initialize OpenAI language model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

# Crear embeddings y almacenar en Chroma
embeddings = OpenAIEmbeddings()
db = Chroma(embedding_function=embeddings)

# Crear un retriever
retriever = db.as_retriever(search_kwargs={"k": 1})

# Preprara el prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """Usa el contexto dado para responder la pregunta. 
     Si no sabes la respuesta, di que no sabes. 
     Contexto: {context}"""),
    ("human", "{input}")])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

#
# MAIN
#

def main():
    # Streamlit app
    st.title("RAG Chat con LangChain y Metadatos")

    # Input folder path
    folder_path = st.text_input("Pon la carpeta donde están los pdfs:", "")
    #pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=False)
    
    # Carga y procesa documentos al apretar el botón
    if st.button("Load and Process Documents"):
        if folder_path:
            # Cargar documentos
            loader = PyPDFDirectoryLoader(folder_path)
            #documents = loader.load()
            pages = []
            for doc in loader.lazy_load():
                pages.append(doc)
                                      
            # Dividir texto en chunks
            # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            # doc_chunks = text_splitter.split_documents(pages)
                
            # Almacenar embeddings en Chroma
            db = Chroma.from_documents(pages, embeddings)
        
            st.success("Documents loaded and processed successfully!")
        else:
            st.error("Please enter a valid folder path.")

 
    # Query input
    query = st.text_input("Enter your query:", "")


    # Answer the query on button click
    if st.button("Get Answer"):
        if query:
            # Usa la cadena así:
            result = chain.invoke({"input": query})
            
            st.write("Result", result)

            # Muestra la respuesta
            answer = result['answer']
            references = []
            for doc in result['context']:
                references.append(doc.metadata['source'])
            
            st.write(f"Respuesta: {answer}")
            st.write(f"Referencias: {references}")
        else:
            st.error("Please enter a valid query.")


if __name__ == '__main__':
    main()