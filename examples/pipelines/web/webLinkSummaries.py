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

Usage:
1. Set the OPENAI_API_KEY in the Valves configuration.
2. Set the TOPICS (comma-separated) in the Valves configuration.
3. Select the desired model from the dropdown in the Valves configuration.
4. Optionally adjust MAX_TOKENS and BATCH_SIZE in the Valves configuration.
5. Input unstructured text containing URLs in the user message.
6. The pipeline will extract URLs, scrape the web pages, and generate summaries.
"""

import re
from typing import List, Union, Generator, Iterator, Tuple
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import sys
import subprocess
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

def install(package):
    """
    Install the specified package using pip.
    This function is used to install the required dependencies within the pipeline script.

    Args:
        package (str): The name of the package to install.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install the required dependencies
install("requests")
install("playwright")
install("beautifulsoup4")
install("openai")

# Install Playwright browsers and dependencies
subprocess.run(["playwright", "install"], check=True)
subprocess.run(["playwright", "install-deps"], check=True)

def extract_urls(text: str) -> List[Tuple[str, str]]:
    """
    Extract all URLs from the text.
    Returns a list of tuples containing (url, url).

    Args:
        text (str): Unstructured text potentially containing URLs.

    Returns:
        List[Tuple[str, str]]: A list of extracted URLs.
    """
    # Pattern for URLs
    url_pattern = r'(https?://\S+)'
    
    # Extract all URLs
    urls = re.findall(url_pattern, text)
    
    # Return list of tuples (url, url)
    return [(url, url) for url in urls]

async def setup_playwright():
    """
    Set up Playwright by installing the required browsers.
    This function is called during the pipeline startup process.
    """
    try:
        async with async_playwright() as p:
            await p.chromium.install()
    except Exception as e:
        print(f"Error setting up Playwright: {e}")

def filter_content(html_content: str) -> str:
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
    
    # Remove extra whitespace and limit to first 32000 words
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()[:32000]
    return ' '.join(words)

def create_prompt(url: str, topics: List[str], max_tokens: int) -> str:
    """
    Create a prompt for generating a summary of the specified URL considering the given topics.

    Args:
        url (str): The URL to summarize.
        topics (List[str]): List of topics to consider in the summaries.
        max_tokens (int): Maximum number of tokens for the summary.

    Returns:
        str: The generated prompt for the OpenAI API.
    """
    topics_str = ", ".join(topics)
    prompt = (
        f"Please create a concise summary of the following web page, based on up to the first 32000 words of the page. "
        f"Follow these guidelines:\n"
        f"- Start the summary with a hyphen followed by a space ('- ').\n"
        f"- If bullet points are appropriate, use a tab followed by a hyphen and a space ('\\t- ') for each point.\n"
        f"- Check the provided list of topics and include the most relevant ones inline within the summary.\n"
        f"- Each relevant topic should be marked only once in the summary.\n"
        f"- Use UK English spelling throughout.\n"
        f"- If the web page is inaccessible or empty, mention that instead of providing a summary.\n"
        f"- Keep the summary to approximately {max_tokens} tokens.\n\n"
        f"List of topics to consider: {topics_str}\n\n"
        f"Web page to summarize: {url}"
    )
    return prompt

async def scrape_url(url: str) -> str:
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
            await page.goto(url, wait_until="domcontentloaded")
            content = await page.content()
            filtered_content = filter_content(content)
            return filtered_content
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        finally:
            await browser.close()

async def summarize_url(client: AsyncOpenAI, url: str, topics: List[str], max_tokens: int, model: str) -> str:
    """
    Summarize a single URL using the OpenAI API.

    Args:
        client (AsyncOpenAI): The OpenAI API client.
        url (str): The URL to summarize.
        topics (List[str]): List of topics to consider in the summaries.
        max_tokens (int): Maximum number of tokens for the summary.
        model (str): The OpenAI model to use for summarization.

    Returns:
        str: The generated summary.
    """
    scraped_content = await scrape_url(url)
    if not scraped_content:
        return f"Unable to access or summarize the content at {url}"

    prompt = create_prompt(url, topics, max_tokens)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web pages."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I understand. I'll summarize the web page based on up to the first 32000 words and highlight relevant topics as requested."},
        {"role": "user", "content": f"Here is the content of the web page (up to 32000 words):\n\n{scraped_content}"}
    ]
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content

def post_process_summary(summary: str, topics: List[str]) -> str:
    """
    Post-process the generated summary to ensure proper topic highlighting.

    Args:
        summary (str): The generated summary.
        topics (List[str]): List of topics to highlight in the summary.

    Returns:
        str: The processed summary with proper topic highlighting.
    """
    for topic in topics:
        if topic.lower() in summary.lower():
            summary = re.sub(r'\b{}\b'.format(re.escape(topic)), r'[[{}]]'.format(topic), summary, count=1, flags=re.IGNORECASE)
    
    return summary

def insert_summaries(original_text: str, summaries: List[Tuple[str, str, str]]) -> str:
    """
    Insert the generated summaries back into the original text.

    Args:
        original_text (str): The original text containing URLs.
        summaries (List[Tuple[str, str, str]]): The summaries to insert, each as a tuple of (url, url, summary).

    Returns:
        str: The text with summaries inserted.
    """
    lines = original_text.split('\n')
    new_lines = []
    summary_index = 0

    for line in lines:
        new_lines.append(line)
        if summary_index < len(summaries):
            _, url, summary = summaries[summary_index]
            if url in line:
                new_lines.append(f"\n{summary}\n")
                summary_index += 1

    return '\n'.join(new_lines)

async def get_available_models(api_key: str) -> List[str]:
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
        return [model.id for model in models.data if model.id.startswith(("gpt-3.5", "gpt-4"))]
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
        MAX_TOKENS: int = 32000  # Maximum number of tokens for each summary
        BATCH_SIZE: int = 10  # Number of URLs to process in each batch
        MODEL: str = "gpt-4-turbo"  # Default model

    def __init__(self):
        self.name = "Efficient Web Summary Pipeline"
        self.valves = self.Valves()
        self.available_models = []

    async def on_startup(self):
        """
        Async function called when the pipeline is started.
        """
        await setup_playwright()  # Set up Playwright in the on_startup method
        self.available_models = await get_available_models(self.valves.OPENAI_API_KEY)
        self.valves.MODEL = self.available_models[0] if self.available_models else "gpt-4-turbo"

    async def on_shutdown(self):
        """
        Async function called when the pipeline is shut down.
        """
        print(f"on_shutdown:{__name__}")

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """
        Main pipeline function that processes the user input, extracts URLs, and generates summaries.

        Args:
            user_message (str): The user's input message containing unstructured text with URLs.
            model_id (str): The ID of the model to use (not used in this implementation).
            messages (List[dict]): Previous messages in the conversation (not used in this implementation).
            body (dict): Additional request body information (not used in this implementation).

        Returns:
            str: The generated summaries for the extracted URLs.
        """
        try:
            openai_key = self.valves.OPENAI_API_KEY
            topics = [topic.strip() for topic in self.valves.TOPICS.split(",")]
            max_tokens = self.valves.MAX_TOKENS
            batch_size = self.valves.BATCH_SIZE
            model = self.valves.MODEL
            
            # Extract URLs from the unstructured text
            urls = extract_urls(user_message)
            
            if not urls:
                return "No valid URLs found in the input text."
            
            client = AsyncOpenAI(api_key=openai_key)
            
            # Process URLs in batches
            summaries = []
            for url, _ in urls:
                summary = asyncio.run(summarize_url(client, url, topics, max_tokens, model))
                processed_summary = post_process_summary(summary, topics)
                summaries.append((url, url, processed_summary))
            
            result = insert_summaries(user_message, summaries)
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
