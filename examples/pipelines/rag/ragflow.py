"""
title: RAGFlow Pipeline
author: open-webui
date: 2024-10-01
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the RAGFlow API.
requirements: requests
"""

import os
import requests
import logging

logging.basicConfig(level=logging.DEBUG)

class Pipeline:
    class Valves(BaseModel):
        ragflow_base_url: str = "http://192.168.0.51/v1/"
        ragflow_api_key: str = "ragflow-g3NzY5MDQ2MmU4NDExZWZiZTcwMDI0Mm"

    def __init__(self):
        self.valves = self.Valves()
        self.conversation_id = None
        self.headers = {"Authorization": f"Bearer {self.valves.ragflow_api_key}"}
        self.user_id = "user_123"

    async def on_startup(self):
        # Log environment info
        logging.debug(f"Environment variables: {os.environ}")
        try:
            # Network connectivity test
            logging.debug("Testing network connectivity...")
            connectivity_response = requests.get("http://192.168.0.51/v1/api/new_conversation", headers=self.headers, params={"user_id": "test"})
            logging.debug(f"Connectivity test status code: {connectivity_response.status_code}")

            # Create a new conversation
            url = f"{self.valves.ragflow_base_url}api/new_conversation"
            params = {"user_id": self.user_id}
            logging.debug(f"Requesting new conversation: URL = {url}, Params = {params}, Headers = {self.headers}")

            response = requests.get(url, headers=self.headers, params=params)
            logging.debug(f"Response status code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                logging.debug(f"Response JSON: {data}")
                self.conversation_id = data.get("data", {}).get("id")
                if not self.conversation_id:
                    raise ValueError("Missing conversation ID in response")
            else:
                raise Exception(f"Failed to create a new conversation: {response.status_code}, {response.text}")
        except Exception as e:
            logging.error(f"Error during on_startup: {str(e)}")
            raise

    async def on_shutdown(self):
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        try:
            url = f"{self.valves.ragflow_base_url}api/completion"
            data = {
                "conversation_id": self.conversation_id,
                "messages": [{"role": "user", "content": user_message}],
                "stream": False,
            }
            logging.debug(f"Requesting completion: URL = {url}, Data = {data}, Headers = {self.headers}")

            response = requests.post(url, headers=self.headers, json=data)
            logging.debug(f"Response status code: {response.status_code}")

            # Print the raw response content to console for inspection
            print(f"Raw response content: {response.text}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    logging.debug(f"Response JSON: {data}")
                    answer = data.get("data", {}).get("answer", "No answer found.")
                    if answer is None:
                        raise ValueError("Missing answer in response")
                    return answer
                except ValueError as e:
                    logging.error(f"Failed to parse JSON response: {str(e)}, {response.text}")
                    raise
            elif response.status_code == 404:
                logging.error(f"RAGFlow API returned a 404 error: {response.text}")
                return "Sorry, the requested resource was not found."
            else:
                raise Exception(f"Failed to retrieve the answer from RAGFlow: {response.status_code}, {response.text}")
        except Exception as e:
            logging.error(f"Error during pipe execution: {str(e)}")
            raise

    def configure(self, config: dict):
        try:
            self.valves.ragflow_base_url = config.get("ragflow_base_url", self.valves.ragflow_base_url)
            self.valves.ragflow_api_key = config.get("ragflow_api_key", self.valves.ragflow_api_key)
            self.headers = {"Authorization": f"Bearer {self.valves.ragflow_api_key}"}
            logging.debug(f"Pipeline configured: Base URL = {self.valves.ragflow_base_url}, API Key = {self.valves.ragflow_api_key}")
        except Exception as e:
            logging.error(f"Error during configuration: {str(e)}")
            raise

pipeline = Pipeline()
