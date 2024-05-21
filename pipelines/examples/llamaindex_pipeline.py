from typing import List, Union, Generator
from schemas import OpenAIChatMessage

import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)


def get_response(
    user_message: str, messages: List[OpenAIChatMessage]
) -> Union[str, Generator]:
    # This is where you can add your custom RAG pipeline.
    # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

    print(messages)
    print(user_message)

    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query(user_message)

    return response.response_gen


async def on_startup():
    # This function is called when the server is started.
    pass


async def on_shutdown():
    # This function is called when the server is stopped.
    pass