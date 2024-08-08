"""
Enhanced Web Summary Pipeline for OpenWebUI and Pipelines

This pipeline script integrates with OpenWebUI and Pipelines to process markdown-style text blocks,
extract URLs, and generate summaries of web pages using the OpenAI API. It includes Reddit API
integration and improved web scraping capabilities.

Key features:
- Processing of text blocks starting with "- "
- URL extraction from the beginning of text blocks
- Efficient web scraping using Playwright with stealth mode
- Reddit API integration for Reddit URLs
- Content filtering to reduce irrelevant data
- Skipping of blocks with embedded links and lots of text
- Topic highlighting in summaries
- Customizable summary length
- Incorporation of original links in summaries
- JSON-structured responses for better parsing
- Logseq-compatible output format
- Random user agent selection for web scraping

Usage:
1. Set the OPENAI_API_KEY in the Valves configuration.
2. Set the TOPICS (comma-separated) in the Valves configuration.
3. Optionally adjust MAX_TOKENS in the Valves configuration.
4. Set Reddit API credentials if needed.
5. Input markdown-style text with blocks starting with "- " in the user message.
6. The pipeline will process eligible blocks, generate summaries, and return the modified text.
"""

import re
import json
import random
import requests
from typing import List, Union, Tuple
# from schemas import OpenAIChatMessage  # Commented out as it's unclear where this schema comes from
from pydantic import BaseModel
import sys
import subprocess
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
from openai import AsyncOpenAI

# --- Package Installation ---
def install(package):
    """
    Installs the specified package using pip.
    This function is used to install the required dependencies within the pipeline script.

    Args:
        package (str): The name of the package to install.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install("requests")
install("playwright")
install("beautifulsoup4")
install("openai")
install("playwright-stealth")

# Install Playwright browsers and dependencies
subprocess.run(["playwright", "install"], check=True)
subprocess.run(["playwright", "install-deps"], check=True)

# --- Helper Functions ---

def extract_blocks(text: str) -> List[str]:
    """
    Extracts text blocks that start with "- " from the input text.

    Args:
        text (str): Input text containing markdown-style blocks.

    Returns:
        List[str]: A list of extracted text blocks.
    """
    return re.split(r'\n(?=- )', text)


def extract_url_from_block(block: str) -> Tuple[str, str, str]:
    """
    Extracts the URL from the beginning of a text block.

    Args:
        block (str): A text block potentially starting with a URL.

    Returns:
        Tuple[str, str, str]: A tuple containing (link_text, url, remaining_text).
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
    Determines if a block should be processed or skipped.

    Args:
        block (str): The text block to analyze.

    Returns:
        bool: True if the block should be processed, False otherwise.
    """
    # Skip blocks with embedded links and lots of text
    if len(block.split()) > 50 and len(re.findall(r'(https?://\S+)', block)) > 1:
        return False
    return True


async def setup_playwright():
    """
    Sets up Playwright by installing the required browsers.
    This function is called during the pipeline startup process.
    """
    try:
        async with async_playwright() as p:
            await p.chromium.install()
    except Exception as e:
        print(f"Error setting up Playwright: {e}")


def filter_content(html_content: str) -> str:
    """
    Filters the HTML content to extract relevant text and limit its length.

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


async def scrape_url(url: str, random_user_agent: bool) -> str:
    """
    Scrapes the content of a given URL using Playwright with stealth mode.

    Args:
        url (str): The URL to scrape.
        random_user_agent (bool): Whether to use a random user agent.

    Returns:
        str or None: The filtered content of the web page, or None if an error occurred.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()

        # Apply stealth settings
        await stealth_async(context)

        if random_user_agent:
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
            ]
            await context.set_extra_http_headers({"User-Agent": random.choice(user_agents)})

        page = await context.new_page()
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
    Creates a prompt for generating a summary of the specified URL considering the given topics.

    Args:
        link_text (str): The text of the link, if available.
        url (str): The URL to summarize.
        topics (List[str]): A list of topics to consider in the summaries.
        max_tokens (int): The maximum number of tokens for the summary.

    Returns:
        str: The generated prompt for the OpenAI API.
    """
    topics_str = ", ".join(topics)
    prompt = (
        f"You are a helpful assistant that summarizes web pages and returns results in JSON format.\n"
        f"Summarize the following web page, ensuring appropriate UK English spelling, handling concatenated words, and returning the result in JSON format. "
        f"Follow these guidelines:\n"
        f"1. If the page is accessible and contains content:\n"
        f"   a. Create a brief descriptive heading (max 50 characters).\n"
        f"   b. Summarize the content in approximately {max_tokens} tokens, focusing on key information and insights.\n"
        f"   c. Explicitly incorporate and mention at least 3 relevant topics from this list: {topics_str}. Use the format [[topic]] to mention them within the summary text.\n"
        f"   d. Format the summary using Logseq-style nested indentation (multiple tabs, dash, space) for better readability within a document.\n"
        f"   e. Ensure proper UK English spelling and separate mistakenly concatenated words.\n"
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


async def summarize_url(client: AsyncOpenAI, link_text: str, url: str, topics: List[str], max_tokens: int, model: str, random_user_agent: bool) -> dict:
    """
    Summarizes a single URL using the OpenAI API.

    Args:
        client (AsyncOpenAI): The OpenAI API client.
        link_text (str): The text of the link, if available.
        url (str): The URL to summarize.
        topics (List[str]): A list of topics to consider in the summaries.
        max_tokens (int): The maximum number of tokens for the summary.
        model (str): The OpenAI model to use for summarization.
        random_user_agent (bool): Whether to use a random user agent for scraping.

    Returns:
        dict: A dictionary containing the summarization result or failure information.
    """
    scraped_content = await scrape_url(url, random_user_agent)
    if not scraped_content:
        return {
            "status": "failure",
            "original_text": f"- [{link_text}]({url})"
        }

    prompt = create_prompt(link_text, url, topics, max_tokens)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web pages and returns results in JSON format."},
        # In-Context Learning Examples
        {"role": "user", "content": "Example 1:\nLink text: 'Example Website'\nWeb page to summarize: https://www.example.com\nTopics: Technology, Artificial Intelligence, Machine Learning\nExpected JSON Output:\n{\n  \"status\": \"success\",\n  \"heading\": \"Example Website Summary\",\n  \"summary\": \"\\t- This is a summary of the Example Website. It discusses various aspects of [[Technology]], including advancements in [[Artificial Intelligence]] and the applications of [[Machine Learning]].\",\n  \"used_topics\": [\"Technology\", \"Artificial Intelligence\", \"Machine Learning\"]\n}"},
        {"role": "user", "content": "Example 2:\nLink text: 'Page Not Found'\nWeb page to summarize: https://www.example.com/404\nTopics: Technology, Artificial Intelligence, Machine Learning\nExpected JSON Output:\n{\n  \"status\": \"failure\",\n  \"original_text\": \"- [Page Not Found](https://www.example.com/404)\"\n}"},
        {"role": "user", "content": "Example 3:\nLink text: 'TheImportanceofGrammar'\nWeb page to summarize: https://www.example.com/grammar\nTopics: Grammar, Linguistics, Education\nExpected JSON Output:\n{\n  \"status\": \"success\",\n  \"heading\": \"The Importance of Grammar\",\n  \"summary\": \"\\t- This article highlights the importance of [[Grammar]] in effective communication. It explores the role of [[Linguistics]] in understanding grammar and its impact on [[Education]].\",\n  \"used_topics\": [\"Grammar\", \"Linguistics\", \"Education\"]\n}"},
        {"role": "user", "content": "Example 4:\nLink text: 'ColourfulWebsite'\nWeb page to summarize: https://www.example.com/colours\nTopics: Colour, Design, Art\nExpected JSON Output:\n{\n  \"status\": \"success\",\n  \"heading\": \"Colourful Website Summary\",\n  \"summary\": \"\\t- This website showcases the use of [[Colour]] in design. It explores various colour palettes and their application in [[Design]] and [[Art]].\",\n  \"used_topics\": [\"Colour\", \"Design\", \"Art\"]\n}"},
        {"role": "user", "content": "Example 5:\nLink text: 'Interesting Reddit Discussion'\nWeb page to summarize: https://www.reddit.com/r/example/comments/123456/interesting_discussion/\nTopics: Reddit, Discussion, Community\nExpected JSON Output:\n{\n  \"status\": \"success\",\n  \"heading\": \"Reddit: Interesting Discussion\",\n  \"summary\": \"Title: Interesting Reddit Discussion\\nAuthor: u/exampleuser\\nScore: 100\\nNumber of comments: 50\\n\\nContent:\\nThis is the content of the Reddit post...\\n\",\n  \"used_topics\": [\"Reddit\", \"Discussion\", \"Community\"]\n}"},
        # Actual Prompt
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

async def process_block(client: AsyncOpenAI, block: str, topics: List[str], max_tokens: int, model: str, random_user_agent: bool, reddit_client: 'RedditClient') -> str:
    """
    Processes a single block, generating a summary if it contains a URL at the beginning.

    Args:
        client (AsyncOpenAI): The OpenAI API client.
        block (str): The text block to process.
        topics (List[str]): A list of topics to consider in the summaries.
        max_tokens (int): The maximum number of tokens for the summary.
        model (str): The OpenAI model to use for summarization.
        random_user_agent (bool): Whether to use a random user agent for scraping.
        reddit_client (RedditClient): The Reddit API client.

    Returns:
        str: The processed block, either summarized or original.
    """
    link_text, url, remaining_text = extract_url_from_block(block)

    if url:
        if reddit_client.is_reddit_url(url):
            result = reddit_client.get_reddit_content(url)
        else:
            result = await summarize_url(client, link_text, url, topics, max_tokens, model, random_user_agent)

        if result["status"] == "success":
            # Integrate Logseq topics as mentions
            summary_with_mentions = result["summary"]
            for topic in result["used_topics"]:
                summary_with_mentions = summary_with_mentions.replace(topic, f"[[{topic}]]")

            # Format the successful summary in Logseq-compatible format with nested bullets
            formatted_summary = (
                f"- ### {result['heading']}\n"
                f"\t- [This web link has been automatically summarised]({url})\n"
                f"{summary_with_mentions}\n"  # Use summary with mentions here
                f"\t- Topics: {', '.join(result['used_topics'])}"
            )
            return formatted_summary
        else:
            # Return the original text if summarization failed
            return result["original_text"]
    else:
        # Return the original block if no URL was found
        return block


class RedditClient:
    def __init__(self, client_id, client_secret, user_agent, username, password):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.username = username
        self.password = password
        self.token = None

    def get_reddit_token(self):
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        data = {
            'grant_type': 'password',
            'username': self.username,
            'password': self.password
        }
        headers = {'User-Agent': self.user_agent}
        res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
        self.token = res.json()['access_token']

    def is_reddit_url(self, url: str) -> bool:
        reddit_pattern = r'^https?://(?:www\.)?reddit\.com/r/[^/]+/comments/'
        return bool(re.match(reddit_pattern, url))

    def get_reddit_content(self, url: str) -> dict:
        if not self.token:
            self.get_reddit_token()

        headers = {"Authorization": f"bearer {self.token}", "User-Agent": self.user_agent}

        # Extract the Reddit post ID from the URL
        post_id = url.split('/')[-3]

        # Fetch the post data
        response = requests.get(f"https://oauth.reddit.com/api/info?id=t3_{post_id}", headers=headers)
        post_data = response.json()['data']['children'][0]['data']

        # Construct a summary of the post
        summary = f"Title: {post_data['title']}\n"
        summary += f"Author: u/{post_data['author']}\n"
        summary += f"Score: {post_data['score']}\n"
        summary += f"Number of comments: {post_data['num_comments']}\n\n"
        summary += f"Content:\n{post_data['selftext']}"

        return {
            "status": "success",
            "heading": f"Reddit: {post_data['title'][:50]}",
            "summary": summary,
            "used_topics": ["Reddit", "Social Media", "User-generated Content"]
        }


# --- Main Pipeline Class ---

class Pipeline:
    class Valves(BaseModel):
        """
        Configuration options for the pipeline.
        These options can be set through the OpenWebUI interface.
        """
        OPENAI_API_KEY: str = ""
        TOPICS: str = ""
        MAX_TOKENS: int = 300
        MODEL: str = "gpt-4o-mini"  # Use gpt-4o-mini model
        RANDOM_USER_AGENT: bool = True
        REDDIT_CLIENT_ID: str = ""
        REDDIT_SECRET: str = ""
        REDDIT_USER_AGENT: str = "openwebui reddit lookup for logseq"
        REDDIT_USERNAME: str = ""
        REDDIT_PASSWORD: str = ""

    def __init__(self):
        self.name = "Enhanced Linear Web Summary Pipeline"
        self.valves = self.Valves()
        self.reddit_client = None

    async def on_startup(self):
        """
        Async function called when the pipeline is started.
        """
        await setup_playwright()
        self.reddit_client = RedditClient(
            self.valves.REDDIT_CLIENT_ID,
            self.valves.REDDIT_SECRET,
            self.valves.REDDIT_USER_AGENT,
            self.valves.REDDIT_USERNAME,
            self.valves.REDDIT_PASSWORD
        )

    async def on_shutdown(self):
        """
        Async function called when the pipeline is shut down.
        """
        print(f"on_shutdown:{__name__}")

    async def process_blocks(self, blocks: List[str], client: AsyncOpenAI, topics: List[str], max_tokens: int, model: str, random_user_agent: bool) -> List[str]:
        """
        Processes all blocks, either summarizing or skipping them based on content.

        Args:
            blocks (List[str]): A list of text blocks to process.
            client (AsyncOpenAI): The OpenAI API client.
            topics (List[str]): A list of topics to consider in the summaries.
            max_tokens (int): The maximum number of tokens for each summary.
            model (str): The OpenAI model to use for summarization.
            random_user_agent (bool): Whether to use a random user agent for web scraping.

        Returns:
            List[str]: A list of processed blocks.
        """
        processed_blocks = []
        for block in blocks:
            if should_process_block(block):
                processed_block = await process_block(client, block, topics, max_tokens, model, random_user_agent, self.reddit_client)
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
            random_user_agent = self.valves.RANDOM_USER_AGENT

            client = AsyncOpenAI(api_key=openai_key)

            # Extract blocks from the user message
            blocks = extract_blocks(user_message)

            # Process blocks
            processed_blocks = asyncio.run(self.process_blocks(blocks, client, topics, max_tokens, model, random_user_agent))

            # Combine processed blocks into final output
            result = "\n".join(processed_blocks)

            return result
        except Exception as e:
            print(f"Error in pipe function: {e}")
            return user_message  # Return the original message if an error occurs

    def get_config(self):
        """
        Returns the configuration options for the pipeline.
        """
        return {
            "OPENAI_API_KEY": {"type": "string", "value": self.valves.OPENAI_API_KEY},
            "TOPICS": {"type": "string", "value": self.valves.TOPICS},
            "MAX_TOKENS": {"type": "number", "value": self.valves.MAX_TOKENS},
            "MODEL": {"type": "string", "value": self.valves.MODEL},
            "RANDOM_USER_AGENT": {"type": "boolean", "value": self.valves.RANDOM_USER_AGENT},
            "REDDIT_CLIENT_ID": {"type": "string", "value": self.valves.REDDIT_CLIENT_ID},
            "REDDIT_SECRET": {"type": "string", "value": self.valves.REDDIT_SECRET},
            "REDDIT_USER_AGENT": {"type": "string", "value": self.valves.REDDIT_USER_AGENT},
            "REDDIT_USERNAME": {"type": "string", "value": self.valves.REDDIT_USERNAME},
            "REDDIT_PASSWORD": {"type": "string", "value": self.valves.REDDIT_PASSWORD}
        }
