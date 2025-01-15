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



