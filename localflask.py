from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from chromadb import AdminClient, AsyncHttpClient, HttpClient

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

import ollama

class LocalOllamaFlask():
    def __init__(self, model, baseUrl = None) -> None:
        self.model = model
        self.chromaURL = baseUrl
        self.htpp_client = HttpClient("http://localhost:8000")
        self.query_embedding = OllamaEmbeddings(model=model)
        
    def CreateCollection(self, private : bool, name : str):
        metadata = {"model": self.model, "Private" : private}
        self.htpp_client.create_collection(name=name, metadata=metadata)
        
    def LoadDocument(self, path : str, CollectionName : str):
        ollama_embeddings = OllamaEmbeddings(model="mistral:latest")
        ChromaClient = Chroma(client=self.htpp_client, collection_name=collectionName, embedding_function=ollama_embeddings, create_collection_if_not_exists=False)
        loader = TextLoader(path)
        text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=250,
                    chunk_overlap=100,
                    length_function=len
                    )

        doc = loader.load()
        collection = self.htpp_client.get_collection(name=CollectionName)
        text_chunks = text_splitter.split_documents(doc)
        ChromaClient.add_documents(text_chunks)
        print("Done")
    
    def RetrieveChunks(self, query : str, collectionName : str):
        collection = self.htpp_client.get_collection(name=collectionName)  # Replace with your collection name
        ollama_embeddings = OllamaEmbeddings(model="mistral:latest")
        vs = Chroma(client=self.htpp_client, collection_name=collectionName, embedding_function=ollama_embeddings, create_collection_if_not_exists=False)
        retriever = vs.as_retriever(search_type="similarity")
        docs = retriever.invoke(query, kwargs=1)
        for doc in docs:
            print(doc)
        return docs

    
    








prompt_template = PromptTemplate(
input_variables=["docs", "query"],
template="""
    You are a knowledgeable assistant. Use the following retrieved information to answer the user's query.

    Retrieved documents:
    {docs}

    User's question: {query}
    
    Provide a concise and accurate response.
    """
)



def chat_with_ollama(query_text: str, CollectionName: str, chromaDB : LocalOllamaFlask):
    print("Here")
    """Chat with Ollama using the retrieved documents and a prompt template."""
    # Step 1: Retrieve relevant documents using the query
    retrieved_docs = chromaDB.RetrieveChunks(query_text, CollectionName)
    
    # Step 2: Join retrieved documents as a single string
     # Join the top documents
    
    # Step 3: Generate the final prompt using the prompt template
    final_prompt = prompt_template.format(
        docs=retrieved_docs,
        query=query_text
    )
    
    chat_model = ChatOllama(model=chromaDB.model, disable_streaming=False)
    #stream = chat_model.astream(final_prompt)
    # Step 4: Pass the final prompt to Ollama chat
    stream = ollama.chat(
    model="mistral:latest",
    messages=[{'role': 'user', 'content': final_prompt }],
    stream=True,
    )
    print("here")
    for chunk in stream:
        print(chunk['message']['content'], end=' ')

    
      
client = LocalOllamaFlask("mistral:latest" ,"http://localhost:8000")
collectionName = "OSILayer2"
chat_with_ollama("What do switches do?", collectionName, client)