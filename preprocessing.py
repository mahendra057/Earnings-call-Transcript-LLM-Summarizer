from langchain_openai import AzureOpenAI  # Import Azure OpenAI
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import tiktoken
from langchain.schema import Document # Wrap each chunk in a Document object


def initialize_azure_openai_llm():

    load_dotenv()

    AZURE_OPENAI_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')

    llm = AzureChatOpenAI(
            deployment_name="gpt-35-turbo-16k",  # or your deployment
            api_version="2024-07-01-preview",  # or your api version This line was uncommented to provide the api_version
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint= AZURE_OPENAI_ENDPOINT,
            model_name="gpt-3.5-turbo"  # Explicitly set the model name
                )

    return llm
    


def split_text_by_tokens(input_string, chunk_size, overlap=0):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(input_string)  # Tokenize the input string

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):  # Create chunks with overlap
        chunk_tokens = tokens[i:i + chunk_size]  # Extract tokens for the current chunk
        decoded_chunk = tokenizer.decode(chunk_tokens)  # Decode the chunk back to text

        #print(f"Chunk {len(chunks) + 1} Length_Tokens: {len(chunk_tokens)} Tokens: {chunk_tokens}")  # Print the token IDs in the chunk
        #print(f"Chunk {len(chunks) + 1} Text: {decoded_chunk}\n")  # Print the decoded text

        chunks.append(decoded_chunk)  # Add the chunk to the list of chunks

    return chunks

