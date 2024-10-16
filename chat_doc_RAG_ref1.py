#
# Origen Perpléxity
#
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
#from langchain_community.llms import OpenAI
#import os
from dotenv import load_dotenv


# Configurar la clave API de OpenAI
load_dotenv()
#os.environ["OPENAI_API_KEY"] = "tu_clave_api_aqui"

# Cargar documentos
loader = TextLoader("../files/final_recomm.txt")
documents = loader.load()

# Dividir texto en chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Crear embeddings y almacenar en Chroma
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

# Crear un retriever
retriever = db.as_retriever(search_kwargs={"k": 1})

# Configurar el modelo de lenguaje
#llm = OpenAI(temperature=0)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI

# # Configura tu retriever aquí
# retriever = ...

# llm = ChatOpenAI()
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Usa el contexto dado para responder la pregunta. Si no sabes la respuesta, di que no sabes."),
#     ("human", "{input}")
# ])

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# chain = create_retrieval_chain(retriever, question_answer_chain)

# # Usa la cadena así:
# response = chain.invoke({"input": "Tu pregunta aquí"})


# Crear la cadena de RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Función para chatear
def chat():
    while True:
        query = input("Tú: ")
        if query.lower() == 'salir':
            break
        
        result = qa_chain({"query": query})
        answer = result['result']
        source = result['source_documents'][0].metadata['source']
        
        print(f"Asistente: {answer}")
        print(f"Fuente: {source}")

# Iniciar el chat
if __name__ == "__main__":
    print("¡Bienvenido al chat RAG! Escribe 'salir' para terminar.")
    chat()