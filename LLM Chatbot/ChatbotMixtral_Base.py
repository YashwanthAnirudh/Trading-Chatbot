from langchain_groq import ChatGroq
import os

# Step 1: Initialize the Llama-based model
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="mixtral-8x7b-32768"
)

# This list will store the conversation history
conversation_history = []

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
        
        # Invoke the base Llama model with the conversation context
        response = llm.invoke(context).content
        
        # Add the bot response to the conversation history
        conversation_history.append(f"Bot: {response}")
        
        # Print the response
        print(f"Bot: {response}")

# Start the chatbot
chat_with_model()
