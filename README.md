# Basic RAG with AstraDB

* UnstructuredIO used to preprocess the data into a AstraDB collection. 
* Embedding model: `text-embedding-3-large` from OpenAI
* Generation LLM: `gpt-3.5-turbo-0125`

To run the script:
1. Populate an AstraDB collection with Unstructured Platform's workflow
2. Create a `.env` file and populate it with the following environment variables: `ASTRA_DB_APPLICATION_TOKEN`, `ASTRA_DB_API_ENDPOINT`, `ASTRA_DB_COLLECTION_NAME`, `ASTRA_DB_NAMESPACE`, `OPENAI_API_KEY`
3. In terminal, run `python retrieve.py` to execute the script. 
