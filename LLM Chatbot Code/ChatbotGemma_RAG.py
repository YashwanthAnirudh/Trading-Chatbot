from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_community.document_loaders import DirectoryLoader  # Updated import
from langchain.text_splitter import CharacterTextSplitter  # Missing import added
import re  # Import for regex cleaning in clean_response function
import os

# Step 1: Initialize the Llama-based model
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="gemma2-9b-it"
)

# Step 2: Load the documents from your specified folder
docs_folder = '/root/Aravind/Llama/llm_env/Docs for model'
loader = DirectoryLoader(docs_folder)  # Load files from the folder
documents = loader.load()

# Step 3: Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Step 4: Use HuggingFace-based embeddings (no API key required)
# This loads a pre-trained SentenceTransformer model from HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Store the embeddings in a FAISS vectorstore for retrieval
vector_store = FAISS.from_documents(texts, embeddings)

# Step 6: Initialize the retrieval-based QA system
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # You can use 'map_reduce' or 'refine' depending on the complexity you want
    retriever=vector_store.as_retriever()
)

# This list will store the conversation history
conversation_history = []

# Function to clean RAG responses, removing references to the source text
def clean_response(response):
    # Remove document references from the response
    response = re.sub(
        r"(according to the (provided )?(text|document|context)|from the provided text|"
        r"the text does not explicitly mention|not explicitly defined in the context|text also mentions that|"
        r"however, it does mention|the provided text does not mention|in the context of provided text|provided,"
        r"this document describes|based on the information provided|the text mentions that|this document discusses how)",  # Added both phrases here
        "", 
        response,
        flags=re.IGNORECASE
    ).strip().lstrip('., ')
    
    return response if response else None  # Return None if the response is empty after cleaning


def chat_with_model():
    print("Welcome to the Groq-based Chatbot! Type 'exit' to end the conversation.")

    while True:
        # Get user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Add user input to the conversation history
        conversation_history.append(f"User: {user_input}")

        # Concatenate the conversation history to give the model context
        context = "\n".join(conversation_history)

        # Step 7: Try to retrieve the answer from your documents first
        try:
            # Extract just the 'result' key from the returned dictionary
            doc_response = retrieval_qa.invoke({"query": user_input})['result']

            # Check if the response indicates no relevant info in docs
            if any(phrase in doc_response.lower() for phrase in ["does not contain", "doesn't provide", "no relevant information", "don't know", "does not explain", "doesn't explain", "this document does not ", "this document doesn't say", "I don't know", "doesn't say"]):
                # Fallback to LLM if RAG doesn't provide a sufficient answer
                response = llm.invoke(user_input).content
                response_type = "LLM"  # Mark response as coming from LLM
            else:
                # Clean the response
                response = clean_response(doc_response) or doc_response  # Fall back if cleaning removes all content
                response_type = "RAG"  # Mark response as coming from RAG

        except Exception as e:
            print(f"Error retrieving docs: {e}. Fallback to model.")
            try:
                # Attempt fallback with the base LLM model
                response = llm.invoke(user_input).content
                response_type = "LLM"  # Mark response as coming from LLM
            except Exception as fallback_error:
                print(f"Error with fallback model: {fallback_error}")
                response = "I'm currently experiencing technical difficulties. Please try again later."
                response_type = "LLM"  # Use LLM as the final fallback

        # Add the bot response to the conversation history with the source type
        conversation_history.append(f"Bot ({response_type}): {response}")

        # Print the response
        print(f"Bot ({response_type}): {response}")

# Start the chatbot
chat_with_model()
