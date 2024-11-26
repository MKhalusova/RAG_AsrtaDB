import os
from dotenv import load_dotenv
import textwrap
from typing import List

from openai import OpenAI
from astrapy import DataAPIClient


def load_environment_variables() -> None:
    """
    Load environment variables from .env file.
    Raises an error if critical environment variables are missing.
    """
    load_dotenv()
    required_vars = [
        "ASTRA_DB_APPLICATION_TOKEN",
        "ASTRA_DB_API_ENDPOINT",
        "ASTRA_DB_COLLECTION_NAME",
        "ASTRA_DB_NAMESPACE",
        "OPENAI_API_KEY"
    ]

    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")


def get_collection(collection_name: str, keyspace: str):
    """
    Establish connection to Astra DB and retrieve specified collection.
    Args:
        collection_name (str): Name of the collection to retrieve
        keyspace (str): Database keyspace
    Returns:
        Collection object from Astra DB
    """

    astra_client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
    database = astra_client.get_database(os.getenv("ASTRA_DB_API_ENDPOINT"))

    # Get the collection
    astradb_collection = database.get_collection(name=collection_name,
                                                 keyspace=keyspace)

    print(f"Collection: {astradb_collection.full_name}\n")
    return astradb_collection


def get_embedding(text: str, openai_client: OpenAI):
    """
    Generate embedding for given text using OpenAI's embedding model.

    Args:
        text (str): Input text to embed
        openai_client (OpenAI): Configured OpenAI client

    Returns:
        Embedding vector for the input text
    """
    return openai_client.embeddings.create(
        input=text, model="text-embedding-3-large"
    ).data[0].embedding


def generate_answer(question: str, documents: List[str], openai_client: OpenAI):
    """
    Generate an answer based on retrieved documents and user question.

    Args:
        question (str): User's input question
        documents (List[str]): List of retrieved documents
        openai_client (OpenAI): Configured OpenAI client

    Returns:
        LLM-generated answer
    """

    prompt = (
        "You are an assistant that can answer user questions given provided context. "
        "Provide a conversational answer. "
        "If you don't know the answer, or no documents are provided, "
        "say 'I do not have enough context to answer the question.'"
    )
    relevant_documents = ""
    for doc in documents:
        relevant_documents += f"Document: \n\n{doc}\n\n"

    augmented_prompt = (
        f"{prompt}"
        f"User question: {question}\n\n"
        f"Retrieved documents to use as context:\n\n {relevant_documents}"
    )
    response = openai_client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': 'You answer users questions.'},
            {'role': 'user', 'content': augmented_prompt},
        ],
        model="gpt-3.5-turbo-0125",
        temperature=0,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Load and validate environment variables
    load_environment_variables()

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Get AstraDB collection
    collection = get_collection(os.getenv("ASTRA_DB_COLLECTION_NAME"), os.getenv("ASTRA_DB_NAMESPACE"))

    # Get user query
    user_input = input("What would you like to know? ")

    # Generate query embedding
    query_vector = get_embedding(user_input, client)

    # Perform similarity search and get 5 documents
    results = collection.find(sort={"$vector": query_vector}, limit=5)
    retrieved_documents = [doc["content"] for doc in results]

    # Pass the documents and the user query to the LLM to generate an answer
    answer = generate_answer(user_input, retrieved_documents, client)
    print(textwrap.fill(answer, width=150))
    print("\n\nContext used:")

    # Optionally, display the documents that were used to generate the answer
    for index, item in enumerate(retrieved_documents):
        print(f"DOCUMENT #{index+1}: \n{item}\n\n")

    # examples: What are Costco's core merchandise categories? Tell me about incentive plans at Chevron?