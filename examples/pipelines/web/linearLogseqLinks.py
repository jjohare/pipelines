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
- Debug logging with easy enable/disable option

Usage:
1. Set the OPENAI_API_KEY in the Valves configuration.
2. Set the TOPICS (comma-separated) in the Valves configuration.
3. Optionally adjust MAX_TOKENS in the Valves configuration.
4. Input markdown-style text with blocks starting with "- " in the user message.
5. The pipeline will process eligible blocks, generate summaries, and return the modified text.
"""

import re
import json
import logging
from typing import List, Union, Tuple
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import sys
import subprocess
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

# Debug logging configuration
DEBUG_MODE = True  # Set to False to disable debug logging

if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())

def debug_log(message):
    if DEBUG_MODE:
        logger.debug(message)

def install(package):
    """
    Install the specified package using pip.
    This function is used to install the required dependencies within the pipeline script.

    Args:
        package (str): The name of the package to install.
    """
    debug_log(f"Installing package: {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install the required dependencies
install("requests")
install("playwright")
install("beautifulsoup4")
install("openai")

# Install Playwright browsers and dependencies
debug_log("Installing Playwright browsers and dependencies")
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
    blocks = re.split(r'\n(?=- )', text)
    debug_log(f"Extracted {len(blocks)} blocks from input text")
    return blocks

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
        link_text, url, remaining = markdown_match.groups()
        debug_log(f"Extracted Markdown-style link: {link_text} - {url}")
        return link_text, url, remaining
    
    # Check for regular URL
    url_match = re.match(r'(https?://\S+)(.*)', content)
    if url_match:
        url, remaining = url_match.groups()
        debug_log(f"Extracted regular URL: {url}")
        return url, url, remaining
    
    debug_log("No URL found in block")
    return "", "", content

def should_process_block(block: str) -> bool:
    """
    Determine if a block should be processed or skipped.

    Args:
        block (str): Text block to analyze.

    Returns:
        bool: True if the block should be processed, False otherwise.
    """
    if len(block.split()) > 50 and len(re.findall(r'(https?://\S+)', block)) > 1:
        debug_log("Skipping block due to length and multiple URLs")
        return False
    return True

async def setup_playwright():
    """
    Set up Playwright by installing the required browsers.
    This function is called during the pipeline startup process.
    """
    try:
        debug_log("Setting up Playwright")
        async with async_playwright() as p:
            await p.chromium.install()
        debug_log("Playwright setup completed successfully")
    except Exception as e:
        logger.error(f"Error setting up Playwright: {e}")

def filter_content(html_content: str) -> str:
    """
    Filter the HTML content to extract relevant text and limit its length.

    Args:
        html_content (str): The raw HTML content of the web page.

    Returns:
        str: Filtered and limited text content.
    """
    debug_log("Filtering HTML content")
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
    filtered_text = ' '.join(words)
    debug_log(f"Filtered content length: {len(filtered_text)} characters")
    return filtered_text

async def scrape_url(url: str) -> str:
    """
    Scrape the content of a given URL using Playwright.

    Args:
        url (str): The URL to scrape.

    Returns:
        str or None: The filtered content of the web page, or None if an error occurred.
    """
    debug_log(f"Scraping URL: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            content = await page.content()
            filtered_content = filter_content(content)
            debug_log(f"Successfully scraped and filtered content from {url}")
            return filtered_content
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
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
    debug_log(f"Created prompt for URL: {url}")
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
    debug_log(f"Attempting to summarize URL: {url}")
    scraped_content = await scrape_url(url)
    if not scraped_content:
        debug_log(f"Failed to scrape content from {url}")
        return {
            "status": "failure",
        }

    prompt = create_prompt(link_text, url, topics, max_tokens)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web pages and returns results in JSON format."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I understand. I'll summarize the web page and return the result in the specified JSON format."},
        {"role": "user", "content": f"Here is the content of the web page (up to 32000 words):\n\n{scraped_content}"}
    ]

    
try:
    debug_log(f"Sending request to OpenAI API for {url}")
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    
    # Check if the response contains choices and is not empty
    if not response or not response.choices or not response.choices[0].message.content:
        debug_log(f"Empty or invalid response from OpenAI API for {url}")
        return {
            "status": "failure",
        }
    
    # Attempt to parse the JSON response
    debug_log(f"Response from OpenAI: {response.choices[0].message.content}")
    result = json.loads(response.choices[0].message.content)
    result["status"] = "success"
    debug_log(f"Successfully summarized {url}")
    return result
except (json.JSONDecodeError, Exception) as e:
    # If JSON parsing fails or any other exception occurs, return a failure status
    logger.error(f"Error in summarize_url for {url}: {e}")
    return {
        "status": "failure",
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
    debug_log(f"Processing block: {block[:50]}...")
    link_text, url, remaining_text = extract_url_from_block(block)
    
    if url:
        result = await summarize_url(client, link_text, url, topics, max_tokens, model)
        if result.get("status") == "success":
            # Format the successful summary in Logseq-compatible format
            formatted_summary = (
                f"- ### {result.get('heading', 'Summary')}\n"
                f"\t[This web link has been automatically summarised]({url})\n"
                f"{result.get('summary', 'No summary available.')}\n"
                f"\tTopics: {', '.join(result.get('used_topics', []))}"
            )
            debug_log(f"Successfully summarized URL: {url}")
            return formatted_summary
        else:
            debug_log(f"Failed to summarize URL: {url}. Returning original block.")
            return block
    else:
        debug_log("No URL found in block. Returning original block.")
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
        debug_log("Pipeline initialized")

    async def on_startup(self):
        """
        Async function called when the pipeline is started.
        """
        debug_log("Pipeline startup initiated")
        await setup_playwright()
        debug_log("Pipeline startup completed")

    async def on_shutdown(self):
        """
        Async function called when the pipeline is shut down.
        """
        debug_log(f"Pipeline shutdown: {__name__}")

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
        debug_log(f"Processing {len(blocks)} blocks")
        processed_blocks = []
        for block in blocks:
            if should_process_block(block):
                processed_block = await process_block(client, block, topics, max_tokens, model)
            else:
                debug_log("Skipping block processing")
                processed_block = block  # Skip processing, keep original
            processed_blocks.append(processed_block)
        debug_log(f"Completed processing {len(processed_blocks)} blocks")
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
            debug_log("Starting pipe function")
            openai_key = self.valves.OPENAI_API_KEY
            topics = [topic.strip() for topic in self.valves.TOPICS.split(",")]
            max_tokens = self.valves.MAX_TOKENS
            model = self.valves.MODEL
            
            debug_log(f"Using model: {model}, max_tokens: {max_tokens}, topics: {topics}")
            
            client = AsyncOpenAI(api_key=openai_key)
            
            # Extract blocks from the user message
            blocks = extract_blocks(user_message)
            
            # Process blocks
            debug_log("Processing blocks")
            processed_blocks = asyncio.run(self.process_blocks(blocks, client, topics, max_tokens, model))
            
            # Combine processed blocks into final output
            result = "\n".join(processed_blocks)
            
            debug_log("Pipe function completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in pipe function: {e}")
            debug_log(f"Returning original user message due to error")
            return user_message  # Return the original message if an error occurs

    def get_config(self):
        """
        Return the configuration options for the pipeline.
        """
        debug_log("Retrieving pipeline configuration")
        return {
            "OPENAI_API_KEY": {"type": "string", "value": self.valves.OPENAI_API_KEY},
            "TOPICS": {"type": "string", "value": self.valves.TOPICS},
            "MAX_TOKENS": {"type": "number", "value": self.valves.MAX_TOKENS},
            "MODEL": {"type": "string", "value": self.valves.MODEL}
        }
