from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import requests
import os

class Pipeline:
    def __init__(self):
        self.base_url = "http://192.168.0.51/v1/api/"
        self.api_key = "ragflow-g3NzY5MDQ2MmU4NDExZWZiZTcwMDI0M"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.conversation_id = None
        self.user_id = "user_123"  # You may want to generate this dynamically

    async def on_startup(self):
        # Create a new conversation
        response = requests.get(
            f"{self.base_url}new_conversation",
            headers=self.headers,
            params={"user_id": self.user_id}
        )
        if response.status_code == 200:
            self.conversation_id = response.json()["data"]["id"]
        else:
            raise Exception(f"Failed to create a new conversation: {response.text}")

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
            f"{self.base_url}completion", json=data, headers=self.headers
        )
        if response.status_code == 200:
            answer = response.json()["data"]["answer"]
            return answer
        else:
            raise Exception(f"Failed to retrieve the answer from RAGFlow: {response.text}")

    def configure(self, config: dict):
        """
        Configure the pipeline with the provided settings.
        This method will be called by OpenWebUI with the settings from the admin panel.
        """
        self.base_url = config.get("ragflow_base_url", self.base_url)
        self.api_key = config.get("ragflow_api_key", self.api_key)
        
        # Ensure the base URL ends with a slash
        if self.base_url and not self.base_url.endswith("/"):
            self.base_url += "/"
        
        # Update headers with new API key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

pipeline = Pipeline()

def get_config_schema():
    """
    Define the configuration schema for the admin panel.
    """
    return {
        "ragflow_base_url": {
            "type": "string",
            "required": False,
            "label": "RAGFlow Base URL",
            "placeholder": "http://192.168.0.xx/v1/api/",
            "default": "http://192.168.0.xx/v1/api/"
        },
        "ragflow_api_key": {
            "type": "string",
            "required": False,
            "label": "RAGFlow API Key",
            "placeholder": "ragflow-default",
            "default": "ragflow-default"
        }
    }
