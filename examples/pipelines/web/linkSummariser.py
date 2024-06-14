"""
Web Summary Pipeline for OpenWebUI and Pipelines

This pipeline script integrates with OpenWebUI and Pipelines to generate summaries of web pages using the Scrapegraph AI library and OpenAI API.

To use this pipeline, add the following line to the PIPELINES_URLS environment variable when running the Docker container:

https://raw.githubusercontent.com/your-username/your-repo/main/linkSummariser.py

Replace "https://raw.githubusercontent.com/your-username/your-repo/main/linkSummariser.py" with the actual URL of this script hosted on GitHub.

Once the pipeline is running, you can access it through the OpenWebUI interface. Provide the following inputs:
- OPENAI_API_KEY (in the admin panel): Your OpenAI API key.
- TOPICS (in the admin panel): A comma-separated list of topics to be considered when generating summaries.
- URL(s) (in the main interaction panel): A single URL or a line-separated list of URLs to be summarized.

The pipeline will process the provided URLs, generate summaries considering the specified topics, and return the summaries to the OpenWebUI interface.
.
"""


from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import sys
import subprocess
import re
import asyncio

def install(package):
    """
    Install the specified package using pip.
    This function is used to install the required dependencies within the pipeline script.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install the required dependencies
install("requests")
install("scrapegraphai>=0.7.0")
install("playwright")  # Install Playwright

# Install Playwright browsers and dependencies
subprocess.run(["playwright", "install"], check=True)
subprocess.run(["playwright", "install-deps"], check=True)

async def setup_playwright():
    """
    Set up Playwright by installing the required browsers.
    """
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            await p.chromium.install()
    except Exception as e:
        print(f"Error setting up Playwright: {e}")

from scrapegraphai.graphs import SmartScraperGraph

class Pipeline:
    class Valves(BaseModel):
        """
        Configuration options for the pipeline.
        These options can be set through the OpenWebUI interface.
        """
        OPENAI_API_KEY: str = ""  # OpenAI API key
        TOPICS: str = ""  # Comma-separated list of topics to be considered when generating summaries

    def __init__(self):
        self.name = "Web Summary Pipeline"
        self.valves = self.Valves()

    async def on_startup(self):
        """
        Async function called when the pipeline is started.
        """
        print(f"on_startup:{__name__}")
        await setup_playwright()  # Set up Playwright in the on_startup method

    async def on_shutdown(self):
        """
        Async function called when the pipeline is shut down.
        """
        print(f"on_shutdown:{__name__}")

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """
        Main pipeline function that processes the user input and generates summaries.
        """
        print(f"pipe:{__name__}")
        openai_key = self.valves.OPENAI_API_KEY
        topics = [topic.strip() for topic in self.valves.TOPICS.split(",")]

        urls = user_message.split("\n")

        summaries = []
        for url in urls:
            summary = process_link(url, openai_key, topics)
            if summary:
                summaries.append(summary)

        return "\n".join(summaries)

def create_prompt(url, topics):
    """
    Create a prompt for generating a summary of the specified URL considering the given topics.
    """
    topics_str = ", ".join(topics)
    prompt = (
        f"Please create a short summary of the web page at the provided URL, unless it is a 404 or similar failure. "
        f"The response should follow these guidelines:\n\n"
        f"- Start the summary with a hyphen followed by a space ('- ').\n"
        f"- If bullet points are appropriate, use a tab followed by a hyphen and a space ('\\t- ') for each point, which is compliant with logseq markdown.\n"
        f"- Embed the web URL inline within the descriptive text, selecting a word sequence of high relevance to the summary.\n"
        f"- Check the provided list of topics and try to find the most relevant ones. If multiple relevant topics are found, include each of them inline within the summary, surrounded by double square brackets (e.g., [[topic1]], [[topic2]]).\n"
        f"- Each relevant topic should be tagged only once in the summary.\n"
        f"- Use UK English spelling throughout.\n"
        f"- If the web page is a 404 or otherwise inaccessible, do not return a summary.\n\n"
        f"List of topics to consider: {topics_str}\n\n"
        f"URL to summarize: {url}"
    )
    return prompt

def process_link(url, openai_key, topics):
    """
    Process a single URL to generate a summary using Scrapegraph AI and OpenAI API.
    """
    graph_config = {
        "llm": {
            "api_key": openai_key,
            "model": "gpt-3.5-turbo-0125",
        },
        "headless": True,  # Set headless to True
    }

    prompt = create_prompt(url, topics)

    smart_scraper_graph = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=graph_config
    )

    try:
        result = smart_scraper_graph.run()
        print("Result:", result)  # Debugging line to check the structure of result
        summary = result.get('summary', '').strip().replace('"', '')

        if not summary or '404' in summary:
            print(f"No summary found for URL: {url}")
            return None

        for topic in topics:
            if topic.lower() in summary.lower():
                summary = re.sub(r'(\b{}\b)'.format(re.escape(topic)), r'[[{}]]'.format(topic), summary, count=1, flags=re.IGNORECASE)

        return summary
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None
        
