
"""-------------------------------------------------------------------------------------
Problem Statement   :   Build a Document Q&A Chatbot using Groq and Llama3
                
Author              :   Vaishali M. Jorwekar  

Date                :   27 Oct 2026  
                
-------------------------------------------------------------------------------------"""
import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community .vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import os
import time,base64

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
###############################################################################
#   Function        :   getAPIKey
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Load the GROQ API key from the environment variable
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Oct 2026  
###############################################################################
def getAPIKey():
    #os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    #groq_API_key = os.getenv("GROQ_API_KEY")
    return os.getenv("GROQ_API_KEY") 
###############################################################################
#   Function        :   getLLMModel
#   Input Params    :   groq_API_key
#   Output Params   :   None
#   Description     :   Load the Llama3 model using the GROQ API key 
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Oct 2026  
###############################################################################
def getLLMModel(groq_API_key):
    llm=ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_API_key)
    return llm
###############################################################################
#   Function        :   getPrompt
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Create a prompt template for the chatbot using the Llama3 model 
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Oct 2026  
###############################################################################
def getPrompt():
    prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only. 
    If the question cannot be answered using the context, respond with "I don't know.""
    <context>
    {context}
    <question>:{input}
    
     """ 
    )          
    return prompt

###############################################################################
#   Function        :   createVectorEmbeddings
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Create vector Embeddings for the documents using the OllamaEmbeddings and store them in a FAISS vector store
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Oct 2026  
###############################################################################
def createVectorEmbeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                                        chunk_overlap=0)
        st.session_state.finalDocuments = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.finalDocuments,
                                                        st.session_state.embeddings)
        
###############################################################################
#   Function        :   displayLoadedDocuments
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Display the loaded documents in the sidebar
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Oct 2026  
###############################################################################
def displayLoadedDocuments():   
    if "vectors" in st.session_state:
        st.title("Loaded Documents:")
        documentNames = list(set([doc.metadata["source"] for doc in st.session_state.finalDocuments]))
        for name in documentNames:
            name=name.split("/")[-1]
            st.write(name)
        st.write("### 📄 Click to View Documents:")
        for name in documentNames:    
            file_path = os.path.join("research_papers", name)
            st.markdown(f"🔗 [{name}]({file_path})") 
    else:
        st.write("No documents loaded yet. Please create document embeddings first.") 
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Oct 2026  
###############################################################################
def main():
    load_dotenv()
    #   Load the GROQ API key from the environment variable
    groq_API_key = getAPIKey()
    
    #   Load llm model
    llm=getLLMModel(groq_API_key)
    
    #   Prompt Building
    prompt=getPrompt()
    
    #   
    question=st.text_input("Ask a question from the documents:")
    submitButton=st.button("Get Answer")    
    with st.sidebar:
        displayLoadedDocuments()
        if st.sidebar.button("Create Document Embeddings"):
            createVectorEmbeddings()
            st.write("Document embeddings created successfully!")
        
    if question and submitButton:
        documentChain=create_stuff_documents_chain(llm=llm, prompt=prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrievalChain = create_retrieval_chain(
                                            retriever=retriever,
                                            combine_docs_chain=documentChain
                            )
        startTime=time.process_time()
        response=retrievalChain.invoke({"input":question})
        endTime=time.process_time()
        st.write(f"Processing Time:{endTime-startTime} seconds")
        st.write(f"Answer:{response['answer']}")
        
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    
    main()