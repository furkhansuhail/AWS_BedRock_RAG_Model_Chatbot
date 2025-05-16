import json
import os
import sys
import boto3
import streamlit as st
import urllib.request as request
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from Model_Ids import *

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrock

# Data ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector store
from langchain_community.vectorstores import FAISS

# LLM utilities
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Bedrock clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Config class for ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list

config = DataIngestionConfig(
    root_dir=Path("Dataset"),
    source_URL="https://raw.githubusercontent.com/furkhansuhail/ProjectData/main/RagModel_Bedrock_Data/Machine%20Learning%20with%20TensorFlow.pdf",
    local_data_file=Path("Dataset/Machine Learning with TensorFlow.pdf"),
    STATUS_FILE="Dataset/status.txt",
    ALL_REQUIRED_FILES=[]
)

def download_project_file(source_URL, local_data_file):
    local_data_file.parent.mkdir(parents=True, exist_ok=True)
    if local_data_file.exists():
        print(f"✅ File already exists at: {local_data_file}")
    else:
        print(f"⬇ Downloading file from {source_URL}...")
        file_path, _ = request.urlretrieve(url=source_URL, filename=local_data_file)
        print(f"✅ File downloaded and saved to: {file_path}")

def data_ingestion():
    loader = PyPDFDirectoryLoader("Dataset")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_documents(documents)

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")





def get_claude_llm():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")

    return ChatBedrock(
        model_id=Haiku_Id,
        model_provider="anthropic",  # ✅ Must be a top-level argument
        client=bedrock_client,
        model_kwargs={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 0.999
        }
    )








def get_llama2_llm():
    llm = Bedrock(
        model_id=Llama_Id,
        client=bedrock,
        model_kwargs={"max_gen_len": 512}
    )
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a
concise answer to the question at the end but use at least summarize with
250 words with detailed explanations. If you don't know the answer,
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


def get_response_llm_Haiku(llm, faiss_index, query):
    from langchain.chains import RetrievalQA
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.prompts import PromptTemplate
    from langchain.chains.llm import LLMChain

    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    prompt_template = PromptTemplate(
        template="Use the following context to answer the question.\n\n{context}\n\nQuestion: {question}",
        input_variables=["context", "question"]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    qa = RetrievalQA(combine_documents_chain=stuff_chain, retriever=retriever)

    # ✅ Use .invoke() instead of deprecated __call__()
    answer = qa.invoke({"query": query})

    return answer["result"]

def main():
    download_project_file(config.source_URL, config.local_data_file)
    st.set_page_config("Machine Learning with TensorFlow.pdf")
    st.header("Chat with PDF using AWS Bedrock💁")
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
