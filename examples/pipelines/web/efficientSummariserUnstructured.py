"""
Efficient Web Summary Pipeline for OpenWebUI and Pipelines

This pipeline script integrates with OpenWebUI and Pipelines to extract URLs from unstructured text
and generate summaries of web pages using the OpenAI API.

Key features:
- URL extraction from unstructured text using regex
- Efficient web scraping using Playwright
- Content filtering to reduce irrelevant data
- Batched processing of URLs
- Topic highlighting in summaries
- Customizable summary length and batch size
- Model selection from available OpenAI models
- Integration of summaries back into the original text

Usage:
1. Set the OPENAI_API_KEY in the Valves configuration.
2. Set the TOPICS (comma-separated) in the Valves configuration.
3. Select the desired model from the dropdown in the Valves configuration.
4. Optionally adjust MAX_TOKENS and BATCH_SIZE in the Valves configuration.
5. Input unstructured text containing URLs in the user message.
6. The pipeline will extract URLs, scrape the web pages, generate summaries, and integrate them back into the text.
"""

import re
import sys
import subprocess
from typing import List
from pydantic import BaseModel
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from openai import AsyncOpenAI

def extract_urls(text):
    """
    Extract URLs from unstructured text using regex.

    Args:
        text (str): Unstructured text potentially containing URLs.

    Returns:
        List[str]: A list of extracted URLs.
    """
    print("Extracting URLs from text...")
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+(?=[.,;:!?)\]}\s]|$)'
    urls = list(set(re.findall(url_pattern, text)))
    print(f"Extracted URLs: {urls}")
    return urls

def setup_playwright():
    """
    Set up Playwright by launching a browser instance.
    This function is called during the pipeline startup process.
    """
    print("Setting up Playwright...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            browser.close()
        print("Playwright setup completed successfully")
    except Exception as e:
        print(f"Error setting up Playwright: {e}")
        sys.exit(1)

def filter_content(html_content):
    """
    Filter the HTML content to extract relevant text and limit its length.

    Args:
        html_content (str): The raw HTML content of the web page.

    Returns:
        str: Filtered and limited text content.
    """
    print("Filtering content from HTML...")
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    main_content = soup.select_one('main, #content, .main-content, article')
    if main_content:
        text = main_content.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)
    
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()[:1000]
    filtered_text = ' '.join(words)
    print(f"Filtered content: {filtered_text[:200]}... (truncated)")
    return filtered_text

def create_prompt(urls, contents, topics, max_tokens):
    """
    Create a prompt for generating summaries of the specified URLs considering the given topics.

    Args:
        urls (List[str]): List of URLs to summarize.
        contents (List[str]): List of corresponding contents from the scraped URLs.
        topics (List[str]): List of topics to consider in the summaries.
        max_tokens (int): Maximum number of tokens for each summary.

    Returns:
        str: The generated prompt for the OpenAI API.
    """
    print("Creating prompt for OpenAI API...")
    topics_str = ", ".join(topics)
    prompt = (
        f"Please create concise summaries of the following web pages, unless they are 404 or similar failures. "
        f"For each summary:\n"
        f"- Start with the URL enclosed in brackets, like this: [URL]\n"
        f"- Follow these guidelines:\n"
        f"  - Start each summary with a hyphen followed by a space ('- ').\n"
        f"  - If bullet points are appropriate, use a tab followed by a hyphen and a space ('\\t- ') for each point.\n"
        f"  - Check the provided list of topics and include the most relevant ones inline within the summary.\n"
        f"  - Each relevant topic should be marked only once in the summary.\n"
        f"  - Use UK English spelling throughout.\n"
        f"  - If a web page is inaccessible, mention that instead of providing a summary.\n"
        f"- Keep each summary to approximately {max_tokens} tokens.\n\n"
        f"List of topics to consider: {topics_str}\n\n"
        f"Web pages to summarize:\n"
    )
    for url, content in zip(urls, contents):
        prompt += f"[{url}]\n{content}\n\n"
    print(f"Generated prompt: {prompt[:500]}... (truncated)")
    return prompt

def scrape_url(url):
    """
    Scrape the content of a given URL using Playwright.

    Args:
        url (str): The URL to scrape.

    Returns:
        str or None: The filtered content of the web page, or None if an error occurred.
    """
    print(f"Scraping URL: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            content = page.content()
            filtered_content = filter_content(content)
            print(f"Successfully scraped {url}")
            return filtered_content
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        finally:
            browser.close()

async def summarize_batch(client, urls, topics, max_tokens, model):
    """
    Summarize a batch of URLs using the OpenAI API.

    Args:
        client (AsyncOpenAI): The OpenAI API client.
        urls (List[str]): List of URLs to summarize.
        topics (List[str]): List of topics to consider in the summaries.
        max_tokens (int): Maximum number of tokens for each summary.
        model (str): The OpenAI model to use for summarization.

    Returns:
        str: The generated summaries for the batch of URLs.
    """
    print(f"Summarizing batch of {len(urls)} URLs...")
    scraped_contents = [scrape_url(url) for url in urls]
    prompt = create_prompt(urls, scraped_contents, topics, max_tokens)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web pages."},
        {"role": "user", "content": prompt},
    ]
    
    try:
        response = await client.chat_completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens * len(urls)
        )
        summaries = response.choices[0].message['content']
        print(f"Successfully generated summaries: {summaries[:500]}... (truncated)")
        return summaries
    except Exception as e:
        print(f"Error generating summaries: {e}")
        return ""

def post_process_summaries(summaries, topics):
    """
    Post-process the generated summaries to ensure proper topic highlighting.

    Args:
        summaries (str): The combined summaries generated by the OpenAI API.
        topics (List[str]): List of topics to highlight in the summaries.

    Returns:
        List[str]: The processed summaries with proper topic highlighting.
    """
    print("Post-processing summaries...")
    processed_summaries = []
    for summary in summaries.split("[http"):
        if not summary.strip():
            continue
        summary = "[http" + summary
        parts = summary.split("]", 1)
        if len(parts) < 2:
            url = "Unknown URL"
            content = summary.strip()
        else:
            url = parts[0][1:]  # Remove the opening bracket
            content = parts[1].strip()
        
        for topic in topics:
            if topic.lower() in content.lower():
                content = re.sub(r'\b{}\b'.format(re.escape(topic)), r'[[{}]]'.format(topic), content, count=1, flags=re.IGNORECASE)
        
        processed_summaries.append(f"{url}\n{content}")
    print(f"Processed summaries: {processed_summaries}")
    return processed_summaries

async def get_available_models(api_key):
    """
    Fetch the list of available models from OpenAI.

    Args:
        api_key (str): OpenAI API key.

    Returns:
        List[str]: List of available model names.
    """
    print("Fetching available models from OpenAI...")
    client = AsyncOpenAI(api_key=api_key)
    try:
        models = await client.models.list()
        available_models = [model.id for model in models.data if model.id.startswith(("gpt-3.5", "gpt-4"))]
        print(f"Available models: {available_models}")
        return available_models
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["gpt-3.5-turbo", "gpt-4"]  # Fallback to default models

class Pipeline:
    class Valves(BaseModel):
        """
        Configuration options for the pipeline.
        These options can be set through the OpenWebUI interface.
        """
        OPENAI_API_KEY: str = ""  # OpenAI API key
        TOPICS: str = ""  # Comma-separated list of topics to be considered when generating summaries
        MAX_TOKENS: int = 2000  # Maximum number of tokens for each summary
        BATCH_SIZE: int = 10  # Number of URLs to process in each batch
        MODEL: str = "gpt-3.5-turbo"  # Default model

    def __init__(self):
        self.name = "Efficient Web Summary Pipeline"
        self.valves = self.Valves()
        self.available_models = []

    async def on_startup(self):
        """
        Async function called when the pipeline is started.
        """
        print(f"Starting up {self.name}")
        setup_playwright()  # Set up Playwright in the on_startup method
        self.available_models = await get_available_models(self.valves.OPENAI_API_KEY)
        self.valves.MODEL = self.available_models[0] if self.available_models else "gpt-3.5-turbo"
        print(f"Startup complete. Using model: {self.valves.MODEL}")

    async def on_shutdown(self):
        """
        Async function called when the pipeline is shut down.
        """
        print(f"Shutting down {self.name}")

    async def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        print(f"Processing input in {self.name}")
        try:
            openai_key = self.valves.OPENAI_API_KEY
            topics = [topic.strip() for topic in self.valves.TOPICS.split(",")]
            max_tokens = self.valves.MAX_TOKENS
            batch_size = self.valves.BATCH_SIZE
            model = self.valves.MODEL
            
            print("Extracting URLs from user message...")
            urls = extract_urls(user_message)
            
            if not urls:
                print("No valid URLs found in the input text.")
                return "No valid URLs found in the input text."
            
            print(f"Found {len(urls)} URLs to process")
            client = AsyncOpenAI(api_key=openai_key)
            
            all_summaries = {}
            
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1} of {len(urls)//batch_size + 1}")
                batch_summaries = await summarize_batch(client, batch, topics, max_tokens, model)
                if batch_summaries:
                    processed_batch_summaries = post_process_summaries(batch_summaries, topics)
                    
                    for url, summary in zip(batch, processed_batch_summaries):
                        all_summaries[url] = summary
                else:
                    print(f"Failed to generate summaries for batch {i//batch_size + 1}")
            
            result = user_message
            for url, summary in all_summaries.items():
                if "404 error" in summary.lower() or "not available" in summary.lower():
                    print(f"Skipping summary for unavailable page: {url}")
                    continue
                
                summary_text = f"\n\n### Web Summary - Auto Generated\n{summary}\n"
                result = result.replace(url, f"{url}{summary_text}")
            
            print("Processing complete. Returning result.")
            return result
        except Exception as e:
            print(f"Error in pipe function: {e}")
            return f"An error occurred while processing the request: {str(e)}"

    def get_config(self):
        """
        Return the configuration options for the pipeline, including the model dropdown.
        """
        print("Getting configuration options...")
        return {
            "OPENAI_API_KEY": {"type": "string", "value": self.valves.OPENAI_API_KEY},
            "TOPICS": {"type": "string", "value": self.valves.TOPICS},
            "MAX_TOKENS": {"type": "number", "value": self.valves.MAX_TOKENS},
            "BATCH_SIZE": {"type": "number", "value": self.valves.BATCH_SIZE},
            "MODEL": {"type": "select", "value": self.valves.MODEL, "options": self.available_models}
        }

# Expose the Pipeline class
pipeline = Pipeline()
print("Pipeline instance created and exposed.")