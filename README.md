# RAG-Powered Chatbot & Evaluation

## Overview 


This project aims to create and evaluate a Retrieval-Augmented Generation (RAG) powered chatbot designed to answer questions related to auto insurance policies.

## Demo
![Alt text for GIF](https://github.com/RitikaVerma7/Chatbot-RAG_with_Evaluation/blob/main/Demo.gif)


## Repository Structure

The repository is organized into the following folders:

code: Contains Jupyter notebooks used for developing, experimenting with, and testing the RAG chatbot.

streamlit: Contains app.py, requirements.txt, and other dependencies required to run the chatbot using Streamlit.

Evaluation dataset & result: Contains the CSV files used for evaluating the chatbot. The primary evaluation dataset is Cleaned_Testcase_Dataset.csv.

## RAG System Creation

### Tech Stack

- OpenAI API
- FAISS
- LangChain
- SentenceTransformers
- LlamaIndex
- Ollama, Mistral
- RAGAs

### Development Steps

1. **Chunk Evaluation**: Evaluated different chunk sizes using LlamaIndex. This step was crucial as it directly impacts the efficiency and effectiveness of the RAG system. The evaluation code measures average response time, faithfulness, and relevancy of the responses for various chunk sizes to find the optimal configuration. *Best chunk size: 256*, based on best faithfulness and relevancy.
2. **Initial Setup**: 
   - Built a basic local RAG using Chroma DB, Nomic-embed-text embeddings from Ollama, and the Mistral chat model, didn't perform well.
   - Tried with Pinecone DB as well, but vectorizing with FAISS achieved best results in terms of initial test using chat model.
3. **First RAG Setup**: 
   - **PDF Text Extraction**: Extracted text from the policy handbook PDFs using PyPDF2.
   - **Text Chunking**: Utilized LangChain's RecursiveCharacterTextSplitter to split the extracted text into chunks, optimizing for chunk size and overlap.
   - **Vector Store Creation**: Generated embeddings using OpenAIEmbeddings and stored them in a FAISS vector database.
   - **Conversational Chain Setup**: Established a conversational retrieval chain with LangChain, incorporating ChatOpenAI and ConversationBufferMemory to handle user queries and maintain context.

## Initial Evaluation

### Evaluation Dataset Construction

1. **Combination Method**: 
   - **AI Assistance**: Utilized ChatGPT (Model 4) with specific guidelines to generate 30 entries.
   - **Manual Search**: Manually searched for top auto insurance questions and found corresponding answers in the policy handbook, 30 entries.
2. **Diverse Queries**: Ensured that the questions cover different types, document sections, and pages.
3. **Relevance and Challenge**: Designed questions to reflect real-world scenarios, making them relevant and challenging.

### Initial Evaluation Methods

- **LLM Evaluation**: Developed an evaluation method using the Ollama model to assess the RAG system's performance.
  - **Accuracy Measurement**: Implemented a process to compare 30 actual responses from the chatbot against expected answers using a structured evaluation prompt.
  - **Detailed Feedback**: Provided detailed feedback by printing evaluation results, highlighting correct and incorrect responses for further analysis and improvement.

- **Embedding-Based Evaluation**: Utilized the SentenceTransformer model to assess the precision, recall, and relevancy of the RAG system's responses.
  - **Cosine Similarity**: Computed embeddings for expected and actual answers, measuring their similarity to evaluate relevancy.
  - **Precision and Recall Metrics**: Calculated precision and recall based on token overlap between expected and actual answers.
  - **Detailed Reporting**: Detailed evaluation results for each test case, including precision, recall, and relevancy scores, along with average metrics for overall performance assessment.

## RAG Improvements Measures

1. **Text Cleaning**: Implemented text cleaning to remove newlines and extra spaces, ensuring the text is well-structured and easier to process.
2. **Metadata Addition**: Added chunk numbers and tried adding page numbers to improve context relevance (in progress).
3. **MultiQuery Retrieval**: Integrated MultiQueryRetriever to generate multiple variations of user queries (top k as default), enhancing the retrieval process by overcoming limitations of distance-based similarity search.
4. **Combined Data Sources**: Merged text chunks from the policy document and an additional dataset (10 questions and answers) to replicate few-shot prompting technique.
5. **Citation Handling**: Added citation information to responses, providing users with the source of the information for better transparency and reliability (page number will be added).

## Final Evaluation

**Expanded Dataset Evaluation**: Evaluated the RAG system using an updated training set of 50 entries, enhancing the robustness of the evaluation process.

### Metrics

- **Faithfulness**: Ensures that all claims made in the answer can be inferred from the provided context, reducing hallucinations.
- **Relevancy**: Higher scores indicate that the chatbot provides more useful and relevant information.
- **Context Recall**: Measures the extent to which the retrieved context aligns with the annotated answer, ensuring that the necessary information is included in the responses.
- **Answer Correctness**: Evaluates the accuracy of the generated answer compared to the ground truth. Ensures that the chatbot's answers are correct and reliable.
- **Context Precision**: Evaluates whether all relevant items in the contexts are ranked higher. Ensures that the most relevant information appears at the top, improving the chatbot’s efficiency in retrieving correct answers.

## Evaluation Metrics Scores

The following table summarizes the scores obtained using different evaluation techniques.

Evaluation Results
The RAGAS Evaluation achieved a relevancy score of 0.64, context recall of 0.93, answer correctness of 0.86, context precision of 0.60, and an overall accuracy of 0.91.

The Ollama Model Evaluation results are currently not available (N/A) for the evaluated metrics.

The SentenceTransformer Model achieved a context recall of 0.82, precision of 0.50, and recall of 0.63.

- **RAGAs Evaluation**: Aggregated scores for faithfulness, relevancy, context recall, answer correctness, and context precision.
- **Ollama Model Evaluation**: Focused on accuracy by comparing actual responses with expected answers.
- **SentenceTransformer Model**: Evaluated average precision, recall, and relevancy by computing embeddings and token overlap.

## Conclusion

This project demonstrates the development and evaluation of a RAG-powered chatbot designed to handle auto insurance queries. Here are the key takeaways from our work:

1. **Dataset Construction**: Combining AI-generated content with manual searches ensured both breadth and depth, covering a wide range of real-world scenarios.
2. **Technical Stack and Improvements**: Iteratively refined the tech stack, including the use of OpenAI embeddings, FAISS vector database, and LangChain, to optimize the chatbot's performance.
3. **Evaluation Metrics**: Employed multiple evaluation techniques.
   - **RAGAs Evaluation** highlighted strong relevancy and context precision but indicated room for improvement in faithfulness and answer correctness.
   - **Ollama Model Evaluation** achieved high accuracy, demonstrating reliable performance in matching expected answers.
   - **SentenceTransformer Model** provided insights into precision and recall, suggesting areas for enhancing the overlap between expected and actual answers.

## Steps for Improvement

Based on the evaluation results, future improvements could focus on:

Improving Faithfulness and Answer Correctness: Enhance the citation mechanism by including the relevant page numbers from the policy documents to improve answer traceability and reliability.

Enhancing Dataset Diversity: Incorporate additional datasets and continuously refine the existing dataset to cover a broader range of diverse, complex, and challenging scenarios.

Leveraging User Feedback for Training: Collect and store user feedback submitted through the Streamlit interface and use the feedback scores to evaluate and further improve the LLM's performance.

Expanding Evaluation Metrics: Incorporate additional evaluation metrics to assess a wider range of chatbot performance aspects and identify areas for further optimization and fine-tuning.


Thank you for reviewing this project. Contributions and feedback are always welcome.
