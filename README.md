# Trading-Chatbot

# problem statement
As a global player in cryptocurrency trading, Ananda Exchange needs actionable insights from vast and complex trading data. Existing models struggle to keep pace with the dynamic and volatile nature of cryptocurrency markets. This project aims to develop advanced analytics and predictive tools to interpret trading signals, identify market opportunities, and mitigate risks effectively.

# Chatbot

This section details the comprehensive approach used to evaluate, configure, and optimize various open-source Large Language Models (LLMs) for developing a high-performance chatbot tailored to cryptocurrency support. The methodology includes model selection, the development of chatbots with distinct configurations, integration of Retrieval-Augmented Generation (RAG), and the evaluation of each configuration’s performance based on established metrics.

**1. Open-Source LLM Selection and Evaluation**

Initial exploration included a broad range of open-source LLMs, with evaluation based on computational efficiency, response accuracy, and compatibility with cryptocurrency-related inquiries. Following a detailed assessment, Llama, Mistral, Gemma, and XGen were selected as the final LLMs for chatbot development. These models were chosen due to their strengths in handling complex queries, efficient resource usage, and adaptability for crypto-specific language.

•	Llama: Selected for its resource efficiency, making it ideal for lightweight deployments without sacrificing performance.

•	Mistral: Known for balancing processing demands with high memory efficiency, particularly suited to chatbot implementations requiring ongoing context retention.

•	Gemma: Offers strong multilingual support, ensuring accessibility to a diverse user base within the cryptocurrency community.

•	XGen: Recognized for its versatility and capability to process a broad range of queries, making it a reliable choice for diverse crypto-related inquiries.

**2. Chatbot Development Across Configurations**

To comprehensively assess each LLM’s capabilities, three chatbot configurations were developed for each finalized model:
•	Standard LLM-Based Chatbot:

o	This chatbot configuration leverages a selected LLM in isolation, without any retrieval mechanisms. The user’s prompt is processed directly by the LLM, which generates a response based solely on its training. The model can recall recent interactions within a session, allowing for continuity in conversation. This configuration serves as the baseline, offering insights into the LLM’s performance without augmented data.

•	LLM-Based Chatbot with RAG Integration:

o	This configuration includes a Retrieval-Augmented Generation (RAG) framework to enhance response relevance by grounding outputs in real-time data. Upon receiving a user prompt, the chatbot searches for relevant information within RAG-supported documents, processing these as data chunks. If relevant data is found, it is combined with the prompt before being processed by the LLM, enhancing factual accuracy. If no relevant data is retrieved, the chatbot falls back to a standard LLM-based response. This setup reduces the risk of hallucinations and improves the chatbot's ability to deliver reliable, data-backed answers.

•	LLM-Based Chatbot with RAG and Tailored Response Mechanism:

o	For this advanced configuration, the chatbot is optimized to deliver responses closely aligned with standardized answers for frequently asked questions. When a prompt is given, the chatbot searches for semantically similar questions within RAG-provided documents. If an exact or closely related match is found, the chatbot responds with the exact text from the document. If no match exists, the system searches for related data and combines it with the prompt for processing, similar to the RAG-only configuration. This setup ensures that commonly asked questions receive consistent, accurate responses, particularly valuable for high-stakes, factual information in the cryptocurrency domain.

**3. Comparative Evaluation of Chatbot Configurations**

After developing these configurations for each LLM (Llama, Mistral, Gemma, and XGen), a comparative evaluation will be conducted to identify the optimal model setup. Performance will measured against standard responses using the following metrics:

•	BLEU (Bilingual Evaluation Understudy Score): Measures word overlap between the chatbot’s response and standard answers, where higher scores indicate closer lexical similarity.

•	ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Evaluates recall of n-grams within responses, utilizing ROUGE-1, ROUGE-2, and ROUGE-L scores to capture different n-gram recalls.

•	BERT Score: Assesses semantic similarity based on embeddings, making it particularly suited for LLM evaluation where meaning over exact words is significant.

•	Cosine Similarity (Embedding): Calculates similarity in the embedding space, typically utilizing Sentence Transformers or similar embeddings, providing insights into overall alignment with intended answers.

The comparative results from each LLM configuration will be analyzed to determine the best-performing chatbot setup, particularly in terms of accuracy and response quality for cryptocurrency-related queries.

**4. Final Selection and Deployment Preparations**

Based on the comparative evaluation, the configuration demonstrating the highest accuracy and consistency with standard responses will be selected for deployment. This finalized chatbot will be integrated into a user-friendly interface (UI) designed for accessibility across Ananda Exchange’s platform, enabling efficient, 24/7 support. Final deployment preparations focused on ensuring seamless performance across devices, aligning with the goal of delivering reliable user experience for cryptocurrency support.

This methodology outlines each step of the approach, including model selection, configuration development, and performance assessment, ensuring clarity and reproducibility for stakeholders. Each section provides insights into not only the actions taken but also the rationale, making this approach adaptable to similar applications in cryptocurrency and beyond.
![image](https://github.com/user-attachments/assets/55d5c5e6-9cdb-4d37-bb94-f6749b1779c6)

# Documentation used for RAG and fine tuning.
1.	Crypto Literature
2.	Factual Data
3.	Coin Base Analysis Reports
4.	Strategies & Techniques of Trading
5.	Market Structure
6.	Crypto Banking Reports
7.	Chart books
   
We prepared standard answers for the provided standard 96 questions. Now each model has 3 versions like Standard LLM-Based Chatbot, LLM-Based Chatbot with RAG Integration and LLM-Based Chatbot with RAG and Tailored Response Mechanism. However, LLM-Based Chatbot with RAG and Tailored Response Mechanism has no difference with the standard answers, ignoring it. Now we have only two versions, so we get six responses from the models. To compare these responses we used the following metrics and respective weights for those metrics.

We developed standard answers for a set of 96 predefined questions.
Each model has three configurations: (1) a Standard LLM-Based Chatbot, (2) an LLM-Based Chatbot with RAG Integration, and (3) an LLM-Based Chatbot with RAG Integration and a Tailored Response Mechanism. However, upon evaluation, the responses generated by the third configuration (RAG Integration with Tailored Response Mechanism) were identical to the standard answers, rendering this configuration redundant for comparison purposes.

Consequently, we focused on the first two configurations, resulting in six distinct responses across the models for each question. To assess and compare these responses, we employed specific evaluation metrics, each assigned a corresponding weight.


# Evaluation metrics used and weights allocation:

•	BLEU (Bilingual Evaluation Understudy Score): Measures word overlap between the chatbot’s response and standard answers, where higher scores indicate closer lexical similarity.

**Reason for the Weight:** BLEU measures precision by evaluating the exact n-gram overlap between the model's generated response and the reference response. While BLEU is a standard metric for evaluating machine translation and other text generation tasks, it has certain limitations that influence its lower weight in this task.

**Weight Justification:** Since semantic meaning and response fluency (as evaluated by ROUGE and BERTScore) are more crucial for chatbot evaluation, BLEU is given a relatively low weight (0.1). We use BLEU primarily for precision but not as the main driver for overall evaluation.

•	ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Evaluates recall of n-grams within responses, utilizing ROUGE-1, ROUGE-2, and ROUGE-L scores to capture different n-gram recalls.

**Reason for the Weight:** ROUGE metrics are designed to evaluate recall, which is critical for determining whether the generated response covers all relevant content from the reference response. Here's a breakdown of why these metrics received their specific weights.

ROUGE-1 (0.2): Measures unigram overlap, focusing on single word matches. This is the basic form of recall, ensuring that key terms in the reference are captured by the model's output. While important, it doesn't capture phrase-level fluency or structure, hence it gets a lower weight than ROUGE-2 and ROUGE-L.

ROUGE-2 (0.25): Measures bigram overlap, i.e., the overlap of two consecutive words. This is more informative than ROUGE-1 because it captures meaningful word pairs and how well the model maintains phrase-level integrity. This is especially important for evaluating fluency and logical structure, so ROUGE-2 gets a slightly higher weight.

ROUGE-L (0.25): Measures the longest common subsequence, which reflects how well the model preserves overall structure and logical flow in its response. This metric is highly valued for evaluating coherence and ensuring that the response is grammatically correct and maintains the essence of the reference, making it more critical for evaluating conversational AI. ROUGE-L receives the highest weight among the ROUGE metrics.

# Why ROUGE Metrics Are Given Significant Weight?

Since ROUGE metrics evaluate both fluency (via ROUGE-2 and ROUGE-L) and coverage of relevant content (via ROUGE-1), they are essential in determining how well the chatbot maintains meaning, logical structure, and content relevance, all of which are highly relevant for conversational agents.

•	BERT Score: Assesses semantic similarity based on embeddings, making it particularly suited for LLM evaluation where meaning over exact words is significant.

**Reason for the Weight:** BERTScore evaluates semantic similarity by comparing the contextual embeddings of words in the generated response and reference response. It uses BERT embeddings to assess meaningful content that goes beyond exact word matches, making it highly relevant for tasks like text generation and dialogue generation.

**Weight Justification:** While BERTScore provides semantic accuracy, it’s not as important as ROUGE in terms of capturing fluency and logical structure. Hence, it receives a moderate weight of 0.15, reflecting the importance of meaning in comparison to fluency and recall metrics.

•	Cosine Similarity (Embedding): Calculates similarity in the embedding space, typically utilizing Sentence Transformers or similar embeddings, providing insights into overall alignment with intended answers.

**Reason for the Weight:** Cosine Similarity measures the overall semantic similarity between the vector representations of two sentences. It compares the angle between the two vectors in high-dimensional space, allowing it to capture general similarities in meaning.

**Weight Justification:** Cosine similarity is more of a secondary metric that supplements other metrics but doesn’t carry as much importance in this particular evaluation. It is given the lowest weight (0.05) to acknowledge its role in providing a broad sense of similarity but not as a primary evaluator for fluency or structure.

# Comparative Evaluation of Chatbot Configurations
# Comprehensive Discription of Six Chatbot Codes

This explains the six codes by grouping common functionalities and highlighting the differences. The explanation starts with the base models and then transitions to Retrieval-Augmented Generation (RAG) models. Repeated code is explained once, with differences emphasized later.

Common Libraries

Below are the libraries used across the codes, with a brief explanation of their purpose:

1.	langchain_groq: Provides the interface to interact with Groq-enabled large language models (LLMs).

2.	os: Used to interact with the operating system for managing environment variables or file paths.

3.	langchain.chains.RetrievalQA: Enables Retrieval-Augmented Generation (RAG) for question answering using external documents.

4.	langchain_huggingface: Integrates HuggingFace embeddings for vectorization of text.

5.	langchain_community.vectorstores.FAISS: Implements a FAISS-based vector store to store and retrieve document embeddings.

6.	langchain_community.document_loaders.DirectoryLoader: Loads documents from a specified directory.

7.	langchain.text_splitter: Splits large documents into smaller, manageable text chunks.

8.	re: Provides regular expressions for cleaning responses.

Base models focus on conversational AI without external document retrieval. These include ChatbotGemma_Base.py, ChatbotLlama_Base.py, and ChatbotMixtral_Base.py.

Code Explanation for Base Models

Common Code for Base Models:
from langchain_groq import ChatGroq
import os

# Initialize the model
llm = ChatGroq(
    temperature=0,  # Controls response randomness
    groq_api_key='YOUR_API_KEY_HERE',  # Replace with your API key
    model_name="MODEL_NAME_HERE"  # Replace with specific model name
    )
# List to store conversation history
conversation_history = []
# Function to interact with the chatbot
def chat_with_model():
    print("Welcome to the chatbot! Type 'exit' to quit.")

    while True:
        # Capture user input
        user_input = input("You: ")
        
        if user_input.lower() == "exit":  # Exit condition
            print("Goodbye!")
            break
        
        # Add user input to conversation history
        conversation_history.append(f"User: {user_input}")
        
        # Use the conversation history as context for the LLM
        context = "\n".join(conversation_history)
        
        # Get response from the LLM
        response = llm.invoke(context).content
        
        # Add bot's response to history and print it
        conversation_history.append(f"Bot: {response}")
        print(f"Bot: {response}")

# Start the chatbot
chat_with_model()


Explanation:
1.	Model Initialization:
o	Uses the ChatGroq class to initialize a model with a specific temperature and API key.
o	The temperature controls randomness: a lower value gives deterministic responses.
o	The model_name specifies the exact model to use (e.g., gemma2-9b-it).
2.	Conversation History:
o	Maintains a list of all interactions (conversation_history) to provide context for subsequent responses.
3.	Chat Functionality:
o	The chatbot runs in a loop, taking user inputs and appending them to the conversation history.
o	The model generates a response based on the full conversation context.
Differences in Base Models
Model Name	Changes in Code
Gemma RAG	model_name="gemma2-9b-it"
Llama RAG	model_name="llama-3.1-70b-versatile"
Mistral RAG	model_name="mixtral-8x7b-32768"

Steps to Execute Base Models
1.	Install langchain_groq using pip install langchain-groq.
2.	Replace YOUR_API_KEY_HERE with your Groq API key in the script.
3.	Run the script (python Chatbot<Model>_Base.py) and interact via the terminal.
Retrieval-Augmented Generation (RAG) Models
RAG integrates document retrieval to enhance chatbot responses with contextually relevant information. These include ChatbotGemma_RAG.py, ChatbotLlama_RAG.py, and ChatbotMistralRAG.py.
How RAG Works
1.	Document Loading:
o	Loads files from a specified directory.
o	Splits the documents into chunks for efficient retrieval.
2.	Embedding Generation:
o	Converts text chunks into vector embeddings using HuggingFace Sentence Transformers.
3.	FAISS Vector Store:
o	Stores the embeddings in a FAISS database for fast retrieval.
4.	Query Handling:
o	Matches user queries with the most relevant document chunks using similarity search.
o	Combines the retrieved chunks with the user query for LLM processing.
5.	Fallback Mechanism:
o	If no relevant information is found in the documents, the model defaults to generating a response based on its training.

Code Explanation for RAG Models
Common Code for RAG Models:
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import re

# Initialize the model
llm = ChatGroq(
    temperature=0,
    groq_api_key='YOUR_API_KEY_HERE',
    model_name="MODEL_NAME_HERE"
)

# Load documents
docs_folder = '/path/to/documents'
loader = DirectoryLoader(docs_folder)
documents = loader.load()

# Split documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create embeddings for the chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(texts, embeddings)

# Initialize Retrieval QA system
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Clean RAG responses
def clean_response(response):
    response = re.sub(r"(according to.*?|based on the provided.*?|the text mentions that)", "", response, flags=re.IGNORECASE)
    return response.strip()

# Chat function
conversation_history = []

def chat_with_model():
    print("Welcome to the RAG-enhanced Chatbot! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        conversation_history.append(f"User: {user_input}")
        context = "\n".join(conversation_history)

        try:
            doc_response = retrieval_qa.invoke({"query": user_input})['result']
            response = clean_response(doc_response)
        except:
            response = llm.invoke(user_input).content

        conversation_history.append(f"Bot: {response}")
        print(f"Bot: {response}")

# Start the chatbot
chat_with_model()



# Explanation:

1.	Document Preprocessing:
o	DirectoryLoader loads documents.
o	CharacterTextSplitter breaks documents into overlapping chunks for better retrieval accuracy.
2.	Embedding and Retrieval:
o	Generates embeddings using HuggingFaceEmbeddings.
o	Stores embeddings in a FAISS vector database.
3.	Retrieval QA System:
o	The RetrievalQA system matches user queries with document embeddings to provide contextually relevant responses.
4.	Fallback Mechanism:
o	If the RAG system fails to find relevant content, the base LLM generates a response.


# Steps to Execute RAG Models

1.	Install dependencies: langchain_groq, faiss-cpu, transformers.

2.	Place relevant documents in a folder and update docs_folder.

3.	Replace YOUR_API_KEY_HERE with the API key.

4.	Run the script (python Chatbot<Model>_RAG.py) and test queries with document-backed responses.


We are collecting responses from all the models listed above and running the metrics on these models using Colab.

After performing the metric evaluation we got the following weighted averages for models.

![image](https://github.com/user-attachments/assets/6b4916ed-2a7b-40f7-aa2a-3f9e701d369a)


The evaluation results for the weighted average scores reveal the highest and lowest performing configurations among the models. The Llama_Base model recorded the lowest score at 0.2385, while the Llama_RAG configuration achieved the highest score at 0.2993. Among the other models, the Gemma_Base scored 0.2636, and the Gemma_RAG slightly improved to 0.2684. Similarly, the Mistral_Base configuration obtained a weighted average of 0.2761, with the Mistral_RAG configuration enhancing its performance to 0.2921. These scores demonstrate the variations in performance across the models and configurations.

To evaluate whether the models are significantly different, a statistical test such as the t-test can be employed. This test assesses whether the means of two groups (in this case, the performance scores of different chatbot configurations) are statistically different from each other.

![image](https://github.com/user-attachments/assets/56d083b8-13d3-46bd-a48e-e118e99fab37)


The table compares various chatbot models using statistical tests to determine which model performs best based on weighted average scores. For each comparison, the "Better Model" is selected when the p-value is less than 0.05, indicating a statistically significant difference. Below, we explain why each selected model is the best:

1.	Llama Base vs. Llama RAG:
o	Better Model: Llama RAG
o	The Llama RAG configuration incorporates retrieval-augmented generation (RAG), which enriches responses by retrieving relevant information from external sources. This significantly improves performance compared to the Llama Base model, reflected in the t-statistic (-9.03) and the highly significant p-value (<0.0001).

2.	Gemma Base vs. Gemma RAG:
o	Better Model: Gemma RAG
o	Although the p-value (0.436) is greater than 0.05, suggesting no statistically significant difference, the RAG component in Gemma RAG is expected to enhance its ability to provide more accurate and contextually relevant answers, making it the slightly better choice.

3.	Mistral Base vs. Mistral RAG:
o	Better Model: Mistral RAG
o	Mistral RAG's integration of RAG improves its retrieval and generation capabilities, outperforming the base model. This is confirmed by the t-statistic (-2.11) and the significant p-value (0.036).

4.	Llama Base vs. Gemma Base:
o	Better Model: Llama Base
o	Llama Base outperforms Gemma Base with a statistically significant difference (p-value <0.0001), likely due to better optimization or model architecture that results in superior baseline performance.

5.	Gemma Base vs. Mistral Base:
o	Better Model: Mistral Base
o	Mistral Base exhibits better baseline performance, as evidenced by a significant p-value (0.0231). Its architecture or training strategy might be more effective compared to Gemma Base.

6.	Mistral Base vs. Llama Base:
o	Better Model: Mistral Base
o	Mistral Base significantly outperforms Llama Base (p-value <0.0001) due to its superior ability to generalize and generate better responses in a base configuration.

7.	Llama RAG vs. Gemma RAG:
o	Better Model: Llama RAG
o	Llama RAG performs better than Gemma RAG, as shown by a significant p-value (0.00012). The superior implementation of RAG in Llama RAG may contribute to its enhanced retrieval and response quality.

8.	Gemma RAG vs. Mistral RAG:
o	Better Model: Mistral RAG
o	Mistral RAG significantly outperforms Gemma RAG (p-value = 0.0035). This indicates that Mistral RAG's architecture or integration with RAG is more effective in leveraging external information.

9.	Mistral RAG vs. Llama RAG:
o	Better Model: Llama RAG
o	The p-value (0.402) indicates no significant difference between the two models. However, Llama RAG's retrieval capabilities might slightly edge out Mistral RAG in terms of consistency or relevance.



# Final Selection and Deployment Preparations

Based on the comparative evaluation, the configuration demonstrating the highest accuracy and consistency with standard responses will be selected for deployment. This finalized chatbot will be integrated into a user-friendly interface (UI) designed for accessibility across Ananda Exchange’s platform, enabling efficient, 24/7 support. Final deployment preparations focused on ensuring seamless performance across devices, aligning with the goal of delivering reliable user experience for cryptocurrency support.

This methodology outlines each step of the approach, including model selection, configuration development, and performance assessment, ensuring clarity and reproducibility for stakeholders. Each section provides insights into not only the actions taken but also the rationale, making this approach adaptable to similar applications in cryptocurrency and beyond.
 
# Evaluation metrics used:

	BLEU (Bilingual Evaluation Understudy Score): Measures word overlap between the chatbot’s response and standard answers, where higher scores indicate closer lexical similarity.

	ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Evaluates recall of n-grams within responses, utilizing ROUGE-1, ROUGE-2, and ROUGE-L scores to capture different n-gram recalls.

	BERT Score: Assesses semantic similarity based on embeddings, making it particularly suited for LLM evaluation where meaning over exact words is significant.

	Cosine Similarity (Embedding): Calculates similarity in the embedding space, typically utilizing Sentence Transformers or similar embeddings, providing insights into overall alignment with intended answers.

We are collecting responses from all the models listed above and running the metrics on these models using Colab.


