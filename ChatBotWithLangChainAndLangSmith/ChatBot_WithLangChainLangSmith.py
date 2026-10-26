####################################################################################################
#   File Name       :   ChatBot_WithLangChainLangSmith.py
#   Description     :   This file contains the code for the ChatBot using LangChain and LangSmith.
#   Author          :   Vaishali M Jorwekar
#   Date            :   2024-06-20
#####################################################################################################
#   Imports
from dotenv import load_dotenv
import streamlit as st
import subprocess
import ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

##############################################################################
#   Function        :   getPromptTemplate
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   This function returns the prompt template.
#   Author          :   Vaishali M Jorwekar
###############################################################################
def getPromptTemplate():
    #   Prompt Template for the chatbot 
    prompt=ChatPromptTemplate.from_messages(
        [
        ("system", "You are a helpful assistant."),
        ("user", "Answer this question:{question}")
        ]
    )
    return prompt
##############################################################################
#   Function        :   generateResponse
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   This function generates the response from the chatbot 
#                       using LangChain and LangSmith.
#   Author          :   Vaishali M Jorwekar
###############################################################################
def generateResponse(question,modelName,temperature,maxTokens):
    prompt=getPromptTemplate()
    model=ChatOllama(model=modelName,
                     temperature=temperature,
                     num_predict=maxTokens)
    parser=StrOutputParser()
    chain= prompt | model | parser
    result = chain.invoke({"question": question})
    return result
##############################################################################
#   Function        :   startOllamaModel
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   This function starts the Ollama model using subprocess.
#   Author          :   Vaishali M Jorwekar
###############################################################################
def startOllamaModel(model):
    if ":" not in model.strip():
        model = f"{model}:latest"
    try:
        """Check if the Ollama model is already running"""
        installedModels=ollama.list()
        downloadedModels = [m['model'] for m in installedModels.get('models', [])]
        
        st.sidebar.info(f"Installed Models: {downloadedModels}")
        if model not in downloadedModels:
            st.sidebar.info(f"Model {model} is not installed. Please install the model first or \
                            select different model.")
            return False
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to start Ollama: {e}")
        return False
    

##############################################################################
#   Function        :   getPageSettings
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   This functions designs the page settings for the chatbot 
#   Author          :   Vaishali M Jorwekar
###############################################################################
def getPageSettings():
    if "modelStarted" not in st.session_state:
        st.session_state.modelStarted = False

    st.sidebar.title("ChatBot Settings")
    #   Model Selection
    model=st.sidebar.selectbox("Select Model",["llama3.1","llama2"])
    startOllamaButton=st.sidebar.button("Start Ollama Model")
    maxTokens=st.sidebar.slider("Select Max Tokens",min_value=200,max_value=1500)
    temperature=st.sidebar.slider("Select Temperature",min_value=0.0,max_value=1.0)
    
    
    if startOllamaButton:
        st.title("ChatBot using LangChain and LangSmith")
        # First Start the Ollama model
        success=startOllamaModel(model) 
        if success:
            st.session_state.modelStarted = True
            st.sidebar.success(f"Model {model} started successfully.")
        
    if st.session_state.modelStarted:    
        st.write("Ask a question :")
        userQuestion=st.text_input("Enter your question here:")
        submitQuestion=st.button("Get Response")
        if userQuestion and submitQuestion:
            response=generateResponse(userQuestion,model,temperature,maxTokens)
            st.write("Response from the chatbot:")
            st.write(response)
        else:
            st.write("Please enter a question and click on 'Get Response' button.")                
##############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    load_dotenv()
    getPageSettings()
    


###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    
    main()