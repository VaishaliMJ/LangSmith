"""-------------------------------------------------------------------------------------
Problem Statement   :   RAG Based QnA With PDF And History
                
Author              :   Vaishali M. Jorwekar  

Date                :   28 Oct 2026  
                
-------------------------------------------------------------------------------------"""
import streamlit as st
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import uuid
###############################################################################
#   Function        :   generateSessionId
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Generate Session Id
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Oct 2026
###############################################################################
def generateSessionId():
    sessionId=uuid.uuid4()
    return str(sessionId)
###############################################################################
#   Function        :   getAPIKey
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Load the GROQ API key from the environment variable
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Oct 2026  
###############################################################################
def getAPIKey():
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    #groq_API_key = os.getenv("GROQ_API_KEY")
    return os.environ["GROQ_API_KEY"]
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
#   Function        :   createPrompt
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Create User Prompt
#   Author          :   Vaishali M Jorwekar
#   Date            :   27 Oct 2026  
###############################################################################
def createPrompt():
    prompt=("Given a chat history and latest user question"
            "Which might reference context in the chat history",
            "formulate a standalone question which can be understood"
            "without chat history. Do not answer the question, just reformulate"
            " it if  needed and otherwise return as it is")
    contexualizePrompt=ChatPromptTemplate.from_messages(
        [
        ("system",prompt),
        MessagesPlaceholder("chatHistory"),
        ("human","{input}"),
        
        ]
    )
    return contexualizePrompt
###############################################################################
#   Function        :   appPageSettings
#   Input Params    :   embeddings
#   Output Params   :   None
#   Description     :   Set the page title and layout for the Streamlit app
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Oct 2026  
###############################################################################
def appPageSettings(embeddings):
    st.title("RAG Based Chatbot with PDF upload and chat history")  
    st.write("Upload PDF's")
    #   Load the GROQ API key from the environment variable
    groq_API_key = getAPIKey()

    if groq_API_key is not None:
        llm=getLLMModel(groq_API_key)
        
        
        session_id=generateSessionId()
        if "store" not in st.session_state:
            st.session_state.store={}
         
        uploadedFiles=st.file_uploader("Upload PDF files",
                                        type=["pdf"],
                                        accept_multiple_files=True)   
        st.write(uploadedFiles)
        #   Prcess the uploaded files and create vector embeddings 
        if uploadedFiles:
            documents=[]
            for uploadedFile in uploadedFiles:
                tempPDFPath=f"./temp.pdf"
                with open(tempPDFPath,"wb") as file:
                    file.write(uploadedFile.getvalue())
                    fileName=uploadedFile.name
                    
                loader=PyPDFLoader(tempPDFPath)
                docs=loader.load()
                documents.extend(docs)
                
            textSplitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)    
            splits=textSplitter.split_documents(documents)
            vectorStore=Chroma.from_documents(documents=splits,embedding=embeddings) 
            retriever=vectorStore.as_retriever()
        
            prompt=createPrompt()    
            
            historyAwareRetriever=create_history_aware_retriever(llm,retriever,prompt)   
         
        #   Question Prompt
            systemPrompt=(
                "You are an assistant for question-answering tasks."
                "Use the following pieces of retireved context to answer the question"
                "If you dont know the answer, say that you 'Don't know the answer'"
                "Use three sentences to keep the answer concise"
                "\n\n"
                "{context}"
                
            )   
            questionPrompt=ChatPromptTemplate.from_messages(
                [
                    ("system",systemPrompt),
                    MessagesPlaceholder("chatHistory"),
                    ("human","{input}")
                ]
            )    
            
            questionAnswerChain=create_stuff_documents_chain(llm,questionPrompt)
            ragChain=create_retrieval_chain(historyAwareRetriever,questionAnswerChain)   
        
            #======================================================================
            def getSessionHistory(session:str)->BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id]=ChatMessageHistory()
                return st.session_state.store[session_id]
            #======================================================================
            
            conversationalRAGChain=RunnableWithMessageHistory(
                                    ragChain,getSessionHistory,
                                    input_messages_key="input",
                                    history_messages_key="chatHistory",
                                    output_messages_key="answer"
            )  
            
            userInput=st.text_input("Your Question:")
            if userInput:
                sessionHistory=getSessionHistory(session_id)
                response=conversationalRAGChain.invoke(
                    {"input":userInput},
                    config={
                        "configurable":{"session_id":session_id}
                    }
                )
                st.write(st.session_state.store)
                st.write("Assistant:",response['answer'])
                st.write("Chat History : ",sessionHistory.messages)
            
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Oct 2026  
###############################################################################
def main():
    load_dotenv()
   
    
    os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    appPageSettings(embeddings)
    

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
###############################################################################




