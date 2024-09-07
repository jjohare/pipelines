from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import requests

class Pipeline:
    class Valves(BaseModel):
        PERPLEXITY_API_BASE_URL: str = "https://api.perplexity.ai"
        PERPLEXITY_API_KEY: str = ""
        
        # Store the models queried from the API
        models: List[dict] = []
        selected_model: str = ""

        # Switches for citations, images, and JSON return
        return_citations: bool = True
        return_images: bool = False
        stream: bool = True

        def select_model(self, model_id: str):
            available_model_ids = [model['id'] for model in self.models]
            if model_id in available_model_ids:
                self.selected_model = model_id
                print(f"Model set to: {model_id}")
            else:
                raise ValueError(f"Model {model_id} not found. Available models: {available_model_ids}")
        
    def __init__(self):
        self.type = "manifold"
        self.name = "Perplexity: "

        self.valves = self.Valves(
            **{
                "PERPLEXITY_API_KEY": os.getenv(
                    "PERPLEXITY_API_KEY", "your-perplexity-api-key-here"
                )
            }
        )

        # Debugging: print the API key to ensure it's loaded
        print(f"Loaded API Key: {self.valves.PERPLEXITY_API_KEY}")

    def query_models(self):
        """
        This function queries the Perplexity API to get the available models.
        """
        headers = {
            "Authorization": f"Bearer {self.valves.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

        model_url = f"{self.valves.PERPLEXITY_API_BASE_URL}/models"

        try:
            r = requests.get(url=model_url, headers=headers)
            r.raise_for_status()

            models = r.json()
            self.valves.models = models  # Save the models in the valves
            print(f"Available models: {models}")
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        print(f"on_valves_updated:{__name__}")
        pass

    def pipe(
        self, user_message: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        headers = {
            "Authorization": f"Bearer {self.valves.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

        # Enhanced payload with dynamic switches for citations and images
        payload = {
            "model": self.valves.selected_model,
            "messages": [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": f"{user_message} Please include references in your response."}
            ],
            "stream": body.get("stream", self.valves.stream),  # Set stream mode dynamically
            "return_citations": self.valves.return_citations,  # Use valve setting for citations
            "return_images": self.valves.return_images  # Use valve setting for images
        }

        print(payload)

        try:
            r = requests.post(
                url=f"{self.valves.PERPLEXITY_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=self.valves.stream,  # Stream switch controlled by valve setting
            )

            r.raise_for_status()

            if self.valves.stream:
                return r.iter_lines()
            else:
                response = r.json()
                formatted_response = {
                    "id": response["id"],
                    "model": response["model"],
                    "created": response["created"],
                    "usage": response["usage"],
                    "object": response["object"],
                    "choices": [
                        {
                            "index": choice["index"],
                            "finish_reason": choice["finish_reason"],
                            "message": {
                                "role": choice["message"]["role"],
                                "content": choice["message"]["content"]
                            },
                            "delta": {"role": "assistant", "content": ""}
                        } for choice in response["choices"]
                    ]
                }
                return formatted_response
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perplexity API Client")
    parser.add_argument("--api-key", type=str, required=True, help="API key for Perplexity")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the Perplexity API")
    parser.add_argument("--model-id", type=str, required=False, help="Model ID to use")
    parser.add_argument("--return-citations", action="store_true", help="Return citations")
    parser.add_argument("--return-images", action="store_true", help="Return images")
    parser.add_argument("--stream", action="store_true", help="Stream the response")

    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.valves.PERPLEXITY_API_KEY = args.api_key

    # Query the available models
    pipeline.query_models()

    # Set model, citations, images, and stream based on input args
    if args.model_id:
        try:
            pipeline.valves.select_model(args.model_id)
        except ValueError as e:
            print(e)
    else:
        # Default to the first model if no model is specified
        pipeline.valves.selected_model = pipeline.valves.models[0]["id"]

    pipeline.valves.return_citations = args.return_citations
    pipeline.valves.return_images = args.return_images
    pipeline.valves.stream = args.stream

    response = pipeline.pipe(
        user_message=args.prompt,
        messages=[],
        body={"stream": pipeline.valves.stream}
    )

    print("Response:", response)
