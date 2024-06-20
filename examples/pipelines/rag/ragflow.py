"""
title: RAGFlow Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the RAGFlow API.
requirements: requests
configuration:
  - name: api_key
    type: str
    default: ""
    description: Your RAGFlow API key.
  - name: base_url
    type: str
    default: "http://localhost:8000/v1/"
    description: The base URL of the RAGFlow API endpoint.
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import requests


class Pipeline:
    def __init__(self, api_key: str, base_url: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.conversation_id = None

    async def on_startup(self):
        # Create a new conversation
        response = requests.get(
            f"{self.base_url}api/new_conversation", headers=self.headers
        )
        if response.status_code == 200:
            self.conversation_id = response.json()["data"]["id"]
        else:
            raise Exception("Failed to create a new conversation")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Send the user message to RAGFlow and retrieve the answer
        data = {
            "conversation_id": self.conversation_id,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}api/completion", json=data, headers=self.headers
        )

        if response.status_code == 200:
            answer = response.json()["data"]["answer"]
            return answer
        else:
            raise Exception("Failed to retrieve the answer from RAGFlow")
