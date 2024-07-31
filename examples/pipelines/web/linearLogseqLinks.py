"""
Enhanced Web Summary Pipeline for OpenWebUI and Pipelines

This pipeline script integrates with OpenWebUI and Pipelines to process markdown-style text blocks,
extract URLs, and generate summaries of web pages using the OpenAI API.

Key features:
- Processing of text blocks starting with "- "
- URL extraction from the beginning of text blocks
- Efficient web scraping using Playwright
- Content filtering to reduce irrelevant data
- Skipping of blocks with embedded links and lots of text
- Topic highlighting in summaries
- Customizable summary length
- Incorporation of original links in summaries
- JSON-structured responses for better parsing
- Logseq-compatible output format

Usage:
1. Set the OPENAI_API_KEY in the Valves configuration.
2. Set the TOPICS (comma-separated) in the Valves configuration.
3. Optionally adjust MAX_TOKENS in the Valves configuration.
4. Input markdown-style text with blocks starting with "- " in the user message.
5. The pipeline will process eligible blocks, generate summaries, and return the modified text.
"""

import re
import json
from typing import List, Union, Tuple
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

def extract_blocks(text: str) -> List[str]:
    """
    Extract text blocks that start with "- " from the input text.

    Args:
        text (str): Input text containing markdown-style blocks.

    Returns:
        List[str]: List of extracted text blocks.
    """
    return re.split(r'\n(?=- )', text)

def extract_url_from_block(block: str) -> Tuple[str, str, str]:
    """
    Extract URL from the beginning of a text block.

    Args:
        block (str): Text block potentially starting with a URL.

    Returns:
        Tuple[str, str, str]: Tuple containing (link_text, url, remaining_text).
    """
    # Remove "- " prefix
    content = block[2:].strip()
    
    # Check for Markdown-style link
    markdown_match = re.match(r'\[([^\]]+)\]\((https?://\S+)\)(.*)', content)
    if markdown_match:
        return markdown_match.group(1), markdown_match.group(2), markdown_match.group(3)
    
    # Check for regular URL
    url_match = re.match(r'(https?://\S+)(.*)', content)
    if url_match:
        return url_match.group(1), url_match.group(1), url_match.group(2)
    
    return "", "", content

def should_process_block(block: str) -> bool:
    """
    Determine if a block should be processed or skipped.

    Args:
        block (str): Text block to analyze.

    Returns:
        bool: True if the block should be processed, False otherwise.
    """
    # Skip blocks with embedded links and lots of text
    if len(block.split()) > 50 and len(re.findall(r'(https?://\S+)', block)) > 1:
        return False
    return True

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

def create_prompt(link_text: str, url: str, topics: List[str], max_tokens: int) -> str:
    """
    Create a prompt for generating a summary of the specified URL considering the given topics.

    Args:
        link_text (str): The text of the link, if available.
        url (str): The URL to summarize.
        topics (List[str]): List of topics to consider in the summaries.
        max_tokens (int): Maximum number of tokens for the summary.

    Returns:
        str: The generated prompt for the OpenAI API.
    """
    topics_str = ", ".join(topics)
    prompt = (
        f"Summarize the following web page and return the result in JSON format. "
        f"Follow these guidelines:\n"
        f"1. If the page is accessible and contains content:\n"
        f"   a. Create a brief descriptive heading (max 50 characters).\n"
        f"   b. Summarize the content in approximately {max_tokens} tokens.\n"
        f"   c. Explicitly incorporate at least 3 relevant topics from this list: {topics_str}.\n"
        f"   d. Format the summary using Logseq-style indentation (tab, dash, space).\n"
        f"2. If the page is inaccessible or empty, return the original link without commentary.\n"
        f"3. Return the result in this JSON format:\n"
        f"   {{\"status\": \"success\" or \"failure\",\n"
        f"    \"heading\": \"Brief descriptive heading\",\n"
        f"    \"summary\": \"Formatted summary\",\n"
        f"    \"used_topics\": [\"topic1\", \"topic2\", \"topic3\"]}}\n\n"
        f"Link text: '{link_text}'\n"
        f"Web page to summarize: {url}"
    )
    return prompt

async def summarize_url(client: AsyncOpenAI, link_text: str, url: str, topics: List[str], max_tokens: int, model: str) -> dict:
    """
    Summarize a single URL using the OpenAI API.

    Args:
        client (AsyncOpenAI): The OpenAI API client.
        link_text (str): The text of the link, if available.
        url (str): The URL to summarize.
        topics (List[str]): List of topics to consider in the summaries.
        max_tokens (int): Maximum number of tokens for the summary.
        model (str): The OpenAI model to use for summarization.

    Returns:
        dict: A dictionary containing the summarization result or failure information.
    """
    scraped_content = await scrape_url(url)
    if not scraped_content:
        return {
            "status": "failure",
            "original_text": f"- [{link_text}]({url})"
        }

    prompt = create_prompt(link_text, url, topics, max_tokens)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web pages and returns results in JSON format."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I understand. I'll summarize the web page and return the result in the specified JSON format."},
        {"role": "user", "content": f"Here is the content of the web page (up to 32000 words):\n\n{scraped_content}"}
    ]
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    
    try:
        # Attempt to parse the JSON response
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        # If JSON parsing fails, return a failure status with the original link
        return {
            "status": "failure",
            "original_text": f"- [{link_text}]({url})"
        }

async def process_block(client: AsyncOpenAI, block: str, topics: List[str], max_tokens: int, model: str) -> str:
    """
    Process a single block, generating a summary if it contains a URL at the beginning.

    Args:
        client (AsyncOpenAI): The OpenAI API client.
        block (str): The text block to process.
        topics (List[str]): List of topics to consider in the summaries.
        max_tokens (int): Maximum number of tokens for the summary.
        model (str): The OpenAI model to use for summarization.

    Returns:
        str: The processed block, either summarized or original.
    """
    link_text, url, remaining_text = extract_url_from_block(block)
    
    if url:
        result = await summarize_url(client, link_text, url, topics, max_tokens, model)
        if result["status"] == "success":
            # Format the successful summary in Logseq-compatible format
            formatted_summary = (
                f"- ### {result['heading']}\n"
                f"\t[This web link has been automatically summarised]({url})\n"
                f"{result['summary']}\n"
                f"\tTopics: {', '.join(result['used_topics'])}"
            )
            return formatted_summary
        else:
            # Return the original text if summarization failed
            return result["original_text"]
    else:
        # Return the original block if no URL was found
        return block

class Pipeline:
    class Valves(BaseModel):
        """
        Configuration options for the pipeline.
        These options can be set through the OpenWebUI interface.
        """
        OPENAI_API_KEY: str = ""  # OpenAI API key
        TOPICS: str = ""  # Comma-separated list of topics to be considered when generating summaries
        MAX_TOKENS: int = 300  # Maximum number of tokens for each summary
        MODEL: str = "gpt-4o-mini"  # Default model (hardcoded)

    def __init__(self):
        self.name = "Linear Web Summary Pipeline"
        self.valves = self.Valves()

    async def on_startup(self):
        """
        Async function called when the pipeline is started.
        """
        await setup_playwright()

    async def on_shutdown(self):
        """
        Async function called when the pipeline is shut down.
        """
        print(f"on_shutdown:{__name__}")

    async def process_blocks(self, blocks: List[str], client: AsyncOpenAI, topics: List[str], max_tokens: int, model: str) -> List[str]:
        """
        Process all blocks, either summarizing or skipping them based on content.

        Args:
            blocks (List[str]): List of text blocks to process.
            client (AsyncOpenAI): The OpenAI API client.
            topics (List[str]): List of topics to consider in the summaries.
            max_tokens (int): Maximum number of tokens for each summary.
            model (str): The OpenAI model to use for summarization.

        Returns:
            List[str]: List of processed blocks.
        """
        processed_blocks = []
        for block in blocks:
            if should_process_block(block):
                processed_block = await process_block(client, block, topics, max_tokens, model)
            else:
                processed_block = block  # Skip processing, keep original
            processed_blocks.append(processed_block)
        return processed_blocks

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        """
        Main pipeline function that processes the user input, extracts blocks, and generates summaries.

        Args:
            user_message (str): The user's input message containing markdown-style blocks.
            model_id (str): The ID of the model to use (not used in this implementation).
            messages (List[dict]): Previous messages in the conversation (not used in this implementation).
            body (dict): Additional request body information (not used in this implementation).

        Returns:
            str: The processed text with summaries for eligible blocks.
        """
        try:
            openai_key = self.valves.OPENAI_API_KEY
            topics = [topic.strip() for topic in self.valves.TOPICS.split(",")]
            max_tokens = self.valves.MAX_TOKENS
            model = self.valves.MODEL
            
            client = AsyncOpenAI(api_key=openai_key)
            
            # Extract blocks from the user message
            blocks = extract_blocks(user_message)
            
            # Process blocks
            processed_blocks = asyncio.run(self.process_blocks(blocks, client, topics, max_tokens, model))
            
            # Combine processed blocks into final output
            result = "\n".join(processed_blocks)
            
            return result
        except Exception as e:
            print(f"Error in pipe function: {e}")
            return user_message  # Return the original message if an error occurs

    def get_config(self):
        """
        Return the configuration options for the pipeline.
        """
        return {
            "OPENAI_API_KEY": {"type": "string", "value": self.valves.OPENAI_API_KEY},
            "TOPICS": {"type": "string", "value": self.valves.TOPICS},
            "MAX_TOKENS": {"type": "number", "value": self.valves.MAX_TOKENS},
            "MODEL": {"type": "string", "value": self.valves.MODEL}
        }
