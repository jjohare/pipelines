"""
title: RAGFlow Pipeline
author: jjohare
date: 2024-10-01
version: 1.1
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the RAGFlow API.
requirements: requests
"""

import logging
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests

# logging.basicConfig(level=logging.DEBUG)

class Pipeline:
    class Valves(BaseModel):
        """
        Configuration options for the pipeline.
        These options can be set through the OpenWebUI interface.
        """
        ragflow_base_url: str = "https://demo.ragflow.io/v1/"  # Updated default base URL
        ragflow_api_key: str = ""  # Empty by default, to be entered by the user

    def __init__(self):
        self.valves = self.Valves()
        self.conversation_id = None
        self.headers = {"Authorization": f"Bearer {self.valves.ragflow_api_key}"}
        self.user_id = "user_123"

    async def on_startup(self):
        # Create a new conversation with GET method and URL-encoded parameters
        url = f"{self.valves.ragflow_base_url}api/new_conversation"
        params = {"user_id": self.user_id}
        logging.debug(f"Requesting new conversation: URL = {url}, Params = {params}, Headers = {self.headers}")

        response = requests.get(url, headers=self.headers, params=params)
        
        logging.debug(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                logging.debug(f"Response JSON: {data}")
                if not isinstance(data, dict):
                    raise ValueError("Invalid response format")
                self.conversation_id = data.get("data", {}).get("id")
                if not self.conversation_id:
                    raise ValueError("Missing conversation ID in response")
            except (ValueError, KeyError, AttributeError) as e:
                raise Exception(f"Failed to parse JSON response: {str(e)}, {response.text}")
        else:
            raise Exception(f"Failed to create a new conversation: {response.status_code}, {response.text}")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # Send the user message to RAGFlow and retrieve the answer
        if not self.conversation_id:
            raise Exception("Conversation ID is not set. Ensure the pipeline has been properly initialized.")
        
        url = f"{self.valves.ragflow_base_url}api/completion"
        data = {
            "conversation_id": self.conversation_id,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
        }
        logging.debug(f"Requesting completion: URL = {url}, Data = {data}, Headers = {self.headers}")

        response = requests.post(url, headers=self.headers, json=data)
        
        logging.debug(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                logging.debug(f"Response JSON: {data}")
                if not isinstance(data, dict):
                    raise ValueError("Invalid response format")
                answer = data.get("data", {}).get("answer", "No answer found.")
                if answer is None:
                    raise ValueError("Missing answer in response")
                return answer
            except (ValueError, KeyError, AttributeError) as e:
                raise Exception(f"Failed to parse JSON response: {str(e)}, {response.text}")
        elif response.status_code == 404:
            # Handle the case when the API returns a 404 error
            logging.error(f"RAGFlow API returned a 404 error: {response.text}")
            return "Sorry, the requested resource was not found."
        else:
            raise Exception(f"Failed to retrieve the answer from RAGFlow: {response.status_code}, {response.text}")

    def configure(self, config: dict):
        """
        Configure the pipeline with the provided settings in the admin panel.
        """
        self.valves.ragflow_base_url = config.get("ragflow_base_url", self.valves.ragflow_base_url)
        self.valves.ragflow_api_key = config.get("ragflow_api_key", self.valves.ragflow_api_key)
        self.headers = {"Authorization": f"Bearer {self.valves.ragflow_api_key}"}
        logging.debug(f"Pipeline configured: Base URL = {self.valves.ragflow_base_url}, API Key = {self.valves.ragflow_api_key}")

pipeline = Pipeline()
