"""
Web Summary Pipeline for OpenWebUI and Pipelines

This pipeline script integrates with OpenWebUI and Pipelines to generate summaries of web pages using OpenAI API.
It includes optimizations for efficient scraping, content filtering, and batched processing to reduce API token usage.

Key features:
- Efficient web scraping using Playwright
- Content filtering to reduce irrelevant data
- Batched processing of URLs
- Topic highlighting in summaries
- Customizable summary length and batch size
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import sys
import subprocess
import re
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
            await page.goto(url, wait_until="domcontentloaded")
            content = await page.content()
            filtered_content = filter_content(content)
            return filtered_content
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        finally:
            await browser.close()

async def summarize_batch(client, urls, topics, max_tokens):
    """
    Summarize a batch of URLs using the OpenAI API.

    Args:
        client (AsyncOpenAI): The OpenAI API client.
        urls (List[str]): List of URLs to summarize.
        topics (List[str]): List of topics to consider in the summaries.
        max_tokens (int): Maximum number of tokens for each summary.

    Returns:
        str: The generated summaries for the batch of URLs.
    """
    scraped_contents = await asyncio.gather(*[scrape_url(url) for url in urls])
    prompt = create_prompt(urls, topics, max_tokens)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web pages."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I understand. I'll summarize the web pages and highlight relevant topics as requested."},
        {"role": "user", "content": "Here are the contents of the web pages:\n\n" + "\n\n".join([f"[{url}]\n{content}" for url, content in zip(urls, scraped_contents) if content])}
    ]
    
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        max_tokens=max_tokens * len(urls)
    )
    
    return response.choices[0].message.content

def post_process_summaries(summaries, topics):
    """
    Post-process the generated summaries to ensure proper topic highlighting.

    Args:
        summaries (str): The combined summaries generated by the OpenAI API.
        topics (List[str]): List of topics to highlight in the summaries.

    Returns:
        str: The processed summaries with proper topic highlighting.
    """
    processed_summaries = []
    for summary in summaries.split("[http"):
        if not summary.strip():
            continue
        summary = "[http" + summary
        url = summary.split("]")[0][1:]
        content = summary.split("]", 1)[1].strip()
        
        for topic in topics:
            if topic.lower() in content.lower():
                content = re.sub(r'\b{}\b'.format(re.escape(topic)), r'[[{}]]'.format(topic), content, count=1, flags=re.IGNORECASE)
        
        processed_summaries.append(f"{url}\n{content}")
    
    return "\n\n".join(processed_summaries)

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

        Args:
            user_message (str): The user's input message containing URLs to summarize.
            model_id (str): The ID of the model to use (not used in this implementation).
            messages (List[dict]): Previous messages in the conversation (not used in this implementation).
            body (dict): Additional request body information (not used in this implementation).

        Returns:
            str: The generated summaries for the provided URLs.
        """
        print(f"pipe:{__name__}")
        openai_key = self.valves.OPENAI_API_KEY
        topics = [topic.strip() for topic in self.valves.TOPICS.split(",")]
        urls = user_message.split("\n")
        
        client = AsyncOpenAI(api_key=openai_key)
        
        # Process URLs in batches of 5
        batch_size = 5
        all_summaries = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            batch_summaries = asyncio.run(summarize_batch(client, batch, topics, 150))
            all_summaries.append(batch_summaries)
        
        combined_summaries = "\n".join(all_summaries)
        processed_summaries = post_process_summaries(combined_summaries, topics)
        
        return processed_summaries
