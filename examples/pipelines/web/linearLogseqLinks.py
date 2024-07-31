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
- Customizable summary length and model selection
- Incorporation of original links in summaries

Usage:
1. Set the OPENAI_API_KEY in the Valves configuration.
2. Set the TOPICS (comma-separated) in the Valves configuration.
3. Select the desired model from the dropdown in the Valves configuration.
4. Optionally adjust MAX_TOKENS in the Valves configuration.
5. Input markdown-style text with blocks starting with "- " in the user message.
6. The pipeline will process eligible blocks, generate summaries, and return the modified text.
"""

import re
from typing import List, Union, Tuple
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import sys
import subprocess
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

# (Keep the existing installation and setup code)

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

# (Keep the existing setup_playwright, filter_content, and scrape_url functions)

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
        f"Please create a concise summary of the following web page, based on up to the first 32000 words of the page. "
        f"Follow these guidelines:\n"
        f"- Start the summary with the original link in markdown format: [<link_text>](<url>).\n"
        f"- Continue the summary immediately after the link.\n"
        f"- If bullet points are appropriate, use a tab followed by a hyphen and a space for each point.\n"
        f"- Check the provided list of topics and include the most relevant ones inline within the summary.\n"
        f"- Each relevant topic should be marked only once in the summary using double square brackets, e.g., [[topic]].\n"
        f"- Use UK English spelling throughout.\n"
        f"- If the web page is inaccessible or empty, mention that instead of providing a summary.\n"
        f"- Keep the summary to approximately {max_tokens} tokens.\n\n"
        f"List of topics to consider: {topics_str}\n\n"
        f"Link text: '{link_text}'\n"
        f"Web page to summarize: {url}"
    )
    return prompt

async def summarize_url(client: AsyncOpenAI, link_text: str, url: str, topics: List[str], max_tokens: int, model: str) -> str:
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
        str: The generated summary.
    """
    scraped_content = await scrape_url(url)
    if not scraped_content:
        return f"[{link_text}]({url}) - Unable to access or summarize the content at this URL."

    prompt = create_prompt(link_text, url, topics, max_tokens)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web pages."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I understand. I'll summarize the web page based on up to the first 32000 words, include the original link, and highlight relevant topics as requested."},
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
    Post-process the generated summary to ensure proper formatting and topic highlighting.

    Args:
        summary (str): The generated summary.
        topics (List[str]): List of topics to highlight in the summary.

    Returns:
        str: The processed summary with proper formatting and topic highlighting.
    """
    # Ensure the summary starts with a markdown link
    if not summary.startswith("["):
        summary = re.sub(r'^(https?://\S+)', r'[\1](\1)', summary)
    
    # Highlight topics
    for topic in topics:
        if topic.lower() in summary.lower():
            summary = re.sub(r'\b{}\b'.format(re.escape(topic)), r'[[{}]]'.format(topic), summary, count=1, flags=re.IGNORECASE)
    
    return summary

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
        summary = await summarize_url(client, link_text, url, topics, max_tokens, model)
        processed_summary = post_process_summary(summary, topics)
        return f"- {processed_summary}"
    else:
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
        MODEL: str = "gpt-4o-mini"  # Default model

    def __init__(self):
        self.name = "Linear Web Summary Pipeline"
        self.valves = self.Valves()
        self.available_models = []

    async def on_startup(self):
        """
        Async function called when the pipeline is started.
        """
        await setup_playwright()
        self.available_models = await get_available_models(self.valves.OPENAI_API_KEY)
        self.valves.MODEL = self.available_models[0] if self.available_models else "gpt-4o-mini"

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
            return f"An error occurred while processing the request: {str(e)}"

    def get_config(self):
        """
        Return the configuration options for the pipeline, including the model dropdown.
        """
        return {
            "OPENAI_API_KEY": {"type": "string", "value": self.valves.OPENAI_API_KEY},
            "TOPICS": {"type": "string", "value": self.valves.TOPICS},
            "MAX_TOKENS": {"type": "number", "value": self.valves.MAX_TOKENS},
            "MODEL": {"type": "select", "value": self.valves.MODEL, "options": self.available_models}
        }
