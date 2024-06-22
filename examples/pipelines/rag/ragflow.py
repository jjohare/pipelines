"""
title: RAGFlow Pipeline
author: open-webui
date: 2024-10-01
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the RAGFlow API.
requirements: requests
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests

class Pipeline:
    class Valves(BaseModel):
        """
        Configuration options for the pipeline.
        These options can be set through the OpenWebUI interface.
        """
        ragflow_base_url: str = "http://192.168.0.51/v1/api/"
        ragflow_api_key: str = "ragflow-g3NzY5MDQ2MmU4NDExZWZiZTcwMDI0M"

    def __init__(self):
        self.valves = self.Valves()
        self.conversation_id = None
        self.headers = {"Authorization": f"Bearer {self.valves.ragflow_api_key}"}
        self.user_id = "user_123"

    async def on_startup(self):
        # Create a new conversation
        response = requests.get(
            f"{self.valves.ragflow_base_url}api/new_conversation",
            headers=self.headers,
            params={"user_id": self.user_id}
        )
        if response.status_code == 200:
            try:
                data = response.json()
                self.conversation_id = data["data"]["id"]
            except ValueError as e:
                raise Exception(f"Failed to parse JSON response: {str(e)}, {response.text}")
        else:
            raise Exception(f"Failed to create a new conversation: {response.status_code}, {response.text}")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # Send the user message to RAGFlow and retrieve the answer
        data = {
            "conversation_id": self.conversation_id,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
        }
        response = requests.post(
            f"{self.valves.ragflow_base_url}api/completion", json=data, headers=self.headers
        )

        if response.status_code == 200:
            try:
                data = response.json()
                answer = data["data"]["answer"]
                return answer
            except ValueError as e:
                raise Exception(f"Failed to parse JSON response: {str(e)}, {response.text}")
        else:
            raise Exception(f"Failed to retrieve the answer from RAGFlow: {response.status_code}, {response.text}")

    def configure(self, config: dict):
        """
        Configure the pipeline with the provided settings in the admin panel.
        """
        self.valves.ragflow_base_url = config.get("ragflow_base_url", self.valves.ragflow_base_url)
        self.valves.ragflow_api_key = config.get("ragflow_api_key", self.valves.ragflow_api_key)
        self.headers = {"Authorization": f"Bearer {self.valves.ragflow_api_key}"}
        # Ensure the base URL ends with a slash
        if not self.valves.ragflow_base_url.endswith("/"):
            self.valves.ragflow_base_url += "/"

pipeline = Pipeline()
