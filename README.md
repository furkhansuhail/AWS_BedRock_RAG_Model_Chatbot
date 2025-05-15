# PDF_RAG_Model_AWS_Chatbot


## In this particular video, we are going to discuss an end-to-end LLM project using AWS Bedrock and LangChain. 

## Project Overview
This project implements a Document Q&A Chatbot that enables users to ask questions about PDF documents using 
state-of-the-art Retrieval-Augmented Generation (RAG). It leverages Amazon Bedrock, fully managed service offering 
multiple foundation models (FMs) like Claude, LLaMA 2, and Amazon Titan — with no infrastructure to manage.
    Topics: 
        
        Q) What Amazon Bedrock is
            Amazon Bedrock is a fully managed generative AI service by AWS that allows you to build and scale generative
            AI applications using foundation models (FMs) from leading AI companies (like Anthropic, Meta, Mistral, 
            Stability AI, Cohere, and Amazon) — without managing infrastructure. You can access these models via simple 
            API calls without needing to train or host them yourself. It supports text, image, embedding, chat, 
            and agent-based tasks. 
                
            Key Features:
                No model training or hosting needed.
                API-based access to multiple model providers.
                Use for text generation, embeddings, agents, image generation, etc.        

            
            Practical Use Cases Using Various Models            
                | Use Case                       | Model                               | Description                                                                                                                        |
                | ------------------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
                | **Chatbot**                    | Claude (Anthropic) / Titan (Amazon) | Conversational agents that can handle customer service, HR queries, etc.                                                           |
                | **Content Summarization**      | Claude / Mistral                    | Summarize lengthy legal, financial, or academic documents.                                                                         |
                | **Text Generation**            | Meta’s LLaMA / Amazon Titan         | Create marketing copy, product descriptions, emails, or creative stories.                                                          |
                | **Image Generation**           | Stability AI                        | Create product images, art, or marketing visuals from text prompts.                                                                |
                | **Embedding for Search**       | Cohere Embed                        | Generate vector representations of text to power semantic search or recommendations.                                               |
                | **RAG-Based Search**           | Titan + Amazon Kendra or OpenSearch | Combine model generation with internal documents using retrieval augmented generation.                                             |
                | **Agents for Task Automation** | Bedrock Agents                      | Chain multiple API calls, functions, and reasoning steps to perform structured tasks (like booking, summarizing + emailing, etc.). |
                |________________________________|_____________________________________|____________________________________________________________________________________________________________________________________|


        Q) Why it matters
            _________________________________________________________________________________________________________________________________________________________
            | Reason                     | Description                                                                                                               |
            | -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
            | Multi-model Access         | Gives you flexibility to choose the best FM from multiple providers (e.g., Claude, Titan, LLaMA, etc.) for your use case. |
            | No Infrastructure Overhead | AWS handles all backend scaling, security, and updates. You just consume the model.                                       |
            | Easy Integration           | Native integration with AWS services (e.g., Lambda, SageMaker, API Gateway, S3, etc.).                                    |
            | Security & Compliance      | Runs inside your VPC, supports IAM, KMS encryption, audit logs via CloudTrail, etc.                                       |
            | Customization Support      | Supports fine-tuning (via Custom Models) and RAG (Retrieval Augmented Generation) with Agents.                            |
            |Supports RAG & Agents       | Easily build Retrieval-Augmented Generation applications and intelligent agents using Bedrock's built-in tools.
            |____________________________|___________________________________________________________________________________________________________________________|
        
        
            

We aim to:

    1. Use RAG (Retrieval-Augmented Generation) architecture.
    2. Store multiple PDFs as vector embeddings in a Vector Store.
    3. Retrieve answers from PDFs using LLMs from AWS Bedrock.

## Requirements & Setup
Libraries Required
    Install the following:

    1. boto3
    2. AWS CLI
    3. PyPDF
    4. LangChain
    5. Streamlit
    6. faiss-cpu

## Setup Steps


    Create a virtual environment.
    
    Run:
        pip install -r requirements.txt
        Configure AWS CLI.
    
    Q) How to code with it
            Step 1: Set up AWS SDK

                pip install boto3



## Project Architecture
    1. Data Ingestion
        - Read all PDFs from a folder.
        - Split documents into chunks.
        - Create embeddings using Amazon Titan.
        - Store embeddings in a vector database like FAISS.
    
    2. Embedding & Vector Store
        - Use LangChain to interact with AWS Bedrock for embedding generation.
        - Supported models: Amazon Titan, OpenAI, Google Generative AI.
    3. Querying
        - Perform similarity search from vector store.
        - Send chunks + query to LLM with a prompt template.
        - Return detailed answers to user.


            User Uploads PDF
                    ↓
            Extract Text from PDF
                    ↓
            Chunk & Embed Text
                    ↓
            Store in Vector DB (e.g., FAISS, Pinecone)
                    ↓
            User Asks Question
                    ↓
            Retrieve Relevant Chunks
                    ↓
            Send Prompt + Chunks to Bedrock Model (Claude / LLaMA 2 / Titan)
                    ↓
            Return Answer to User
