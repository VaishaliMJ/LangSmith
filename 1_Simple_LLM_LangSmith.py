from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser




###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    load_dotenv()
    
    prompTemp=PromptTemplate.from_template("Answer this question:{question}")
    
    model=ChatOllama(model="llama3",temperature=0.7)
    
    parser=StrOutputParser()
    
    chain= prompTemp | model | parser
    
    result = chain.invoke({"question": "What is the capital of India?"})
    print(result)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    
    main()