from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import re

# Step 1: Initialize the Llama-based model
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="llama-3.1-70b-versatile"
)

# Step 2: Load the documents from your specified folder
docs_folder = '/root/Aravind/Llama/llm_env/Docs for model'
loader = DirectoryLoader(docs_folder)
documents = loader.load()

# Step 3: Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Step 4: Use HuggingFace-based embeddings (no API key required)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Store the embeddings in a FAISS vectorstore for retrieval
vector_store = FAISS.from_documents(texts, embeddings)

# Step 6: Initialize the retrieval-based QA system
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Conversation history
conversation_history = []

# Function to clean RAG responses, removing references to the source text
def clean_response(response):
    # Remove document references from the response
    response = re.sub(
        r"(according to the (provided )?(text|document|context)|from the provided text|"
        r"the text does not explicitly mention|not explicitly defined in the context|text also mentions that|"
        r"however, it does mention|the provided text does not mention|in the context of provided text|"
        r"this document describes|based on the information provided|the text mentions that|this document discusses how)",  # Added both phrases here
        "", 
        response,
        flags=re.IGNORECASE
    ).strip().lstrip('., ')
    
    return response if response else None  # Return None if the response is empty after cleaning

# Function to determine if fallback is needed based on inadequate response indicators
def needs_fallback(response):
    # Keywords and phrases suggesting insufficient information
    inadequate_phrases = [
        "not explicitly define", "not described", "does not explain", 
        "no detailed information", "not specifically provided", 
        "no clear definition", "the provided text does not mention", 
        "does not contain", "doesn't provide", "no relevant information", 
        "don't know", "does not explain", "doesn't explain", "this document does not say", 
        "this document doesn't say", "I don't know", "does not list"
    ]
    # Trigger fallback if any inadequate phrase is present in the response
    return any(phrase in response.lower() for phrase in inadequate_phrases) or not response

# Main chat function
def chat_with_model():
    print("Welcome to the Groq-based Chatbot! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Process conversation without repeatedly adding duplicate questions
        if conversation_history and conversation_history[-1] != f"User: {user_input}":
            conversation_history.append(f"User: {user_input}")
        context = "\n".join(conversation_history)
        
        # Retrieve the RAG response
        try:
            raw_response = retrieval_qa.invoke({"query": user_input})['result']  # Query directly without full context
            cleaned_response = clean_response(raw_response)  # Clean document references
            
            # Fallback if the response is inadequate or contains inadequate phrases
            if cleaned_response is None or needs_fallback(cleaned_response):
                response = llm.invoke(user_input).content  # Use LLM fallback if RAG response is insufficient
            else:
                response = cleaned_response  # Use the RAG response if it’s adequate
        except Exception as e:
            print(f"Error retrieving docs: {e}. Fallback to model.")
            response = llm.invoke(user_input).content
        
        # Add the bot response to conversation history and print it
        conversation_history.append(f"Bot: {response}")
        print(f"Bot: {response}")

# Start the chatbot
chat_with_model()
