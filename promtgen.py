from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    # api_key="" # Optional if not set as an environment variable
    groq_api_key = '##token'
)
def chat_with_template(human_response):
    # Define the system message
    system_message = "You are a helpful assistant."
    
    # Define the human message using the input parameter
    human_message = human_response
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])
    
    # Create a chat chain with the prompt template and chat API
    chain = prompt | chat
    
    # Invoke the chain with the human response
    response = chain.invoke({"text": human_response})
    
    return response.content
