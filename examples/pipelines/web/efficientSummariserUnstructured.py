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
import asyncio
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

def install_packages(packages):
    """
    Install the specified packages using pip.
    This function is used to install the required dependencies within the pipeline script.

    Args:
        packages (List[str]): The list of packages to install.
    """
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            sys.exit(1)

# Install the required dependencies
install_packages(["requests", "playwright", "beautifulsoup4", "openai"])

# Install Playwright browsers and dependencies
try:
    subprocess.run(["playwright", "install"], check=True)
    subprocess.run(["playwright", "install-deps"], check=True)
    print("Playwright and dependencies installed successfully")
except subprocess.CalledProcessError as e:
    print(f"Error installing Playwright: {e}")
    sys.exit(1)

def extract_urls(text):
    """
    Extract URLs from unstructured text using regex.

    Args:
        text (str): Unstructured text potentially containing URLs.

    Returns:
        List[str]: A list of extracted URLs.
    """
    # This regex pattern matches most common URL formats and excludes trailing punctuation
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+(?=[.,;:!?)\]}\s]|$)'
    return list(set(re.findall(url_pattern, text)))

async def setup_playwright():
    """
    Set up Playwright by launching a browser instance.
    This function is called during the pipeline startup process.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            await browser.close()
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
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove scripts, styles, and other unnecessary elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    # Extract main content (adjust selectors based on common website structures)
    main_content = soup.select_one('main, #content, .main-content, article')
    if main_content:
        text = main_content.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)
    
    # Remove extra whitespace and limit to first 1000 words
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()[:1000]
    return ' '.join(words)

def create_prompt(urls, topics, max_tokens):
    """
    Create a prompt for generating summaries of the specified URLs considering the given topics.

    Args:
        urls (List[str]): List of URLs to summarize.
        topics (List[str]): List of topics to consider in the summaries.
        max_tokens (int): Maximum number of tokens for each summary.

    Returns:
        str: The generated prompt for the OpenAI API.
    """
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
        f"Web pages to summarize:\n" + "\n".join(urls)
    )
    return prompt

async def scrape_url(url):
    """
    Scrape the content of a given URL using Playwright.

    Args:
        url (str): The URL to scrape.

    Returns:
        str or None: The filtered content of the web page, or None if an error occurred.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            content = await page.content()
            filtered_content = filter_content(content)
            print(f"Successfully scraped {url}")
            return filtered_content
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        finally:
            await browser.close()

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
        List[str]: The generated summaries for the batch of URLs.
    """
    scraped_contents = await asyncio.gather(*[scrape_url(url) for url in urls])
    prompt = create_prompt(urls, topics, max_tokens)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web pages."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I understand. I'll summarize the web pages and highlight relevant topics as requested."},
        {"role": "user", "content": "Here are the contents of the web pages:\n\n" + "\n\n".join([f"[{url}]\n{content}" for url, content in zip(urls, scraped_contents) if content])}
    ]
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens * len(urls)
        )
        print(f"Successfully generated summaries for batch of {len(urls)} URLs")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summaries: {e}")
        return []

def post_process_summaries(summaries, topics):
    """
    Post-process the generated summaries to ensure proper topic highlighting.

    Args:
        summaries (str): The combined summaries generated by the OpenAI API.
        topics (List[str]): List of topics to highlight in the summaries.

    Returns:
        List[str]: The processed summaries with proper topic highlighting.
    """
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
    
    return processed_summaries

async def get_available_models(api_key):
    """
    Fetch the list of available models from OpenAI.

    Args:
        api_key (str): OpenAI API key.

    Returns:
        List[str]: List of available model names.
    """
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
        await setup_playwright()  # Set up Playwright in the on_startup method
        self.available_models = await get_available_models(self.valves.OPENAI_API_KEY)
        self.valves.MODEL = self.available_models[0] if self.available_models else "gpt-3.5-turbo"
        print(f"Startup complete. Using model: {self.valves.MODEL}")

    async def on_shutdown(self):
        """
        Async function called when the pipeline is shut down.
        """
        print(f"Shutting down {self.name}")

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """
        Main pipeline function that processes the user input, extracts URLs, and generates summaries.

        Args:
            user_message (str): The user's input message containing unstructured text with URLs.
            model_id (str): The ID of the model to use (not used in this implementation).
            messages (List[dict]): Previous messages in the conversation (not used in this implementation).
            body (dict): Additional request body information (not used in this implementation).

        Returns:
            str: The original text with integrated summaries for the extracted URLs.
        """
        print(f"Processing input in {self.name}")
        try:
            openai_key = self.valves.OPENAI_API_KEY
            topics = [topic.strip() for topic in self.valves.TOPICS.split(",")]
            max_tokens = self.valves.MAX_TOKENS
            batch_size = self.valves.BATCH_SIZE
            model = self.valves.MODEL
            
            # Extract URLs from the unstructured text
            urls = extract_urls(user_message)
            
            if not urls:
                print("No valid URLs found in the input text.")
                return "No valid URLs found in the input text."
            
            print(f"Found {len(urls)} URLs to process")
            client = AsyncOpenAI(api_key=openai_key)
            
            # Process URLs in batches
            all_summaries = {}
            
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1} of {len(urls)//batch_size + 1}")
                batch_summaries = asyncio.run(summarize_batch(client, batch, topics, max_tokens, model))
                if batch_summaries:
                    processed_batch_summaries = post_process_summaries(batch_summaries, topics)
                    
                    for url, summary in zip(batch, processed_batch_summaries):
                        all_summaries[url] = summary
                else:
                    print(f"Failed to generate summaries for batch {i//batch_size + 1}")
            
            # Integrate summaries back into the original text
            result = user_message
            for url, summary in all_summaries.items():
                if "404 error" in summary.lower() or "not available" in summary.lower():
                    # If it's a 404 or unavailable, keep the original URL
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
