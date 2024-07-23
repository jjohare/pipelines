"""
Efficient Web Summary Pipeline for OpenWebUI and Pipelines

This pipeline script integrates with OpenWebUI and Pipelines to extract URLs from unstructured text,
generate summaries of web pages using the OpenAI API, and insert summaries back into the original text.

Key features:
- URL extraction from specific Markdown format
- Linear processing of URLs in document order
- Web scraping using Playwright
- Content filtering to reduce irrelevant data
- Topic highlighting in summaries
- Customizable summary length
- Model selection from available OpenAI models
- Extensive logging and debug information
"""

import re
from typing import List, Tuple
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import sys
import subprocess
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install(package):
    """
    Install the specified package using pip.
    This function is used to install the required dependencies within the pipeline script.
    """
    logger.info(f"Installing package: {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install the required dependencies
install("requests")
install("playwright")
install("beautifulsoup4")
install("openai")

# Install Playwright browsers and dependencies
logger.info("Installing Playwright browsers and dependencies")
subprocess.run(["playwright", "install"], check=True)
subprocess.run(["playwright", "install-deps"], check=True)

def extract_urls(text: str) -> List[Tuple[str, str]]:
    """
    Extract URLs from the specific Markdown format in the text.
    Returns a list of tuples containing (link_text, url).
    """
    pattern = r'\[(.*?)\]\((.*?)\)(?=\n|\r|\r\n)'
    matches = re.findall(pattern, text)
    logger.debug(f"Extracted {len(matches)} URLs")
    return matches

def filter_content(html_content: str) -> str:
    """
    Filter the HTML content to extract relevant text and limit it to the first 32000 words.
    """
    logger.debug("Filtering HTML content")
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    main_content = soup.select_one('main, #content, .main-content, article')
    if main_content:
        text = main_content.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)
    
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()[:32000]  # Limit to first 32000 words
    filtered_text = ' '.join(words)
    logger.debug(f"Filtered content (first 100 chars): {filtered_text[:100]}...")
    return filtered_text

def create_prompt(url: str, topics: List[str], max_tokens: int) -> str:
    """
    Create a prompt for generating a summary of the specified URL considering the given topics.
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
    logger.debug(f"Created prompt for URL: {url}")
    return prompt

def scrape_url(url: str) -> str:
    """
    Scrape the content of a given URL using Playwright.
    """
    logger.info(f"Scraping URL: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)  # 30 seconds timeout
            content = page.content()
            logger.debug(f"Raw content for {url} (first 100 chars): {content[:100]}...")
            filtered_content = filter_content(content)
            logger.debug(f"Filtered content word count: {len(filtered_content.split())}")
            return filtered_content
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
        finally:
            browser.close()

def summarize_url(client: OpenAI, url: str, topics: List[str], max_tokens: int, model: str) -> str:
    """
    Summarize a single URL using the OpenAI API.
    """
    logger.info(f"Summarizing URL: {url}")
    scraped_content = scrape_url(url)
    if not scraped_content:
        return f"Unable to access or summarize the content at {url}"

    prompt = create_prompt(url, topics, max_tokens)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web pages."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I understand. I'll summarize the web page based on up to the first 32000 words and highlight relevant topics as requested."},
        {"role": "user", "content": f"Here is the content of the web page (up to 32000 words):\n\n{scraped_content}"}
    ]
    
    logger.debug(f"OpenAI API request - Model: {model}, Max tokens: {max_tokens}")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    
    summary = response.choices[0].message.content
    logger.debug(f"Generated summary (first 100 chars): {summary[:100]}...")
    return summary

def post_process_summary(summary: str, topics: List[str]) -> str:
    """
    Post-process the generated summary to ensure proper topic highlighting.
    """
    logger.info("Post-processing summary")
    for topic in topics:
        if topic.lower() in summary.lower():
            summary = re.sub(r'\b{}\b'.format(re.escape(topic)), r'[[{}]]'.format(topic), summary, count=1, flags=re.IGNORECASE)
    
    logger.debug(f"Processed summary (first 100 chars): {summary[:100]}...")
    return summary

def insert_summaries(original_text: str, summaries: List[Tuple[str, str, str]]) -> str:
    """
    Insert the generated summaries back into the original text.
    """
    logger.info("Inserting summaries into original text")
    lines = original_text.split('\n')
    new_lines = []
    summary_index = 0

    for line in lines:
        new_lines.append(line)
        if summary_index < len(summaries):
            link_text, url, summary = summaries[summary_index]
            if f"[{link_text}]({url})" in line:
                new_lines.append(f"\t\t- {summary}")
                summary_index += 1

    return '\n'.join(new_lines)

def get_available_models(api_key: str) -> List[str]:
    """
    Fetch the list of available models from OpenAI.
    """
    logger.info("Fetching available OpenAI models")
    if not api_key:
        logger.warning("API key is not set. Returning default models.")
        return ["gpt-4-0125-preview", "gpt-3.5-turbo"]
    
    client = OpenAI(api_key=api_key)
    try:
        models = client.models.list()
        available_models = [model.id for model in models.data if model.id.startswith(("gpt-3.5", "gpt-4"))]
        logger.debug(f"Available models: {available_models}")
        return available_models
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return ["gpt-4-0125-preview", "gpt-3.5-turbo"]  # Fallback to default models

class Pipeline:
    class Valves(BaseModel):
        """
        Configuration options for the pipeline.
        """
        OPENAI_API_KEY: str = ""
        TOPICS: str = ""
        MAX_TOKENS: int = 8000
        MODEL: str = "gpt-4-0125-preview"

    def __init__(self):
        self.name = "Efficient Web Summary Pipeline"
        self.valves = self.Valves()
        self.available_models = []

    def on_startup(self):
        """
        Function called when the pipeline is started.
        """
        logger.info(f"Starting up {self.name}")
        if not self.valves.OPENAI_API_KEY:
            logger.warning("OpenAI API key is not set. Please set it before using the pipeline.")
            self.available_models = ["gpt-4o-mini", "gpt-4o"]  # Default models
        else:
            self.available_models = get_available_models(self.valves.OPENAI_API_KEY)
        self.valves.MODEL = self.available_models[0] if self.available_models else "gpt-4-0125-preview"
        logger.info(f"Startup complete. Using model: {self.valves.MODEL}")

    def on_shutdown(self):
        """
        Function called when the pipeline is shut down.
        """
        logger.info(f"Shutting down {self.name}")

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        """
        Main pipeline function that processes the user input, extracts URLs, generates summaries,
        and inserts them back into the original text.
        """
        logger.info("Processing user message")
        try:
            openai_key = self.valves.OPENAI_API_KEY
            topics = [topic.strip() for topic in self.valves.TOPICS.split(",")]
            max_tokens = self.valves.MAX_TOKENS
            model = self.valves.MODEL
            
            logger.debug(f"Pipeline configuration: Topics: {topics}, Max tokens: {max_tokens}, Model: {model}")
            
            url_matches = extract_urls(user_message)
            
            if not url_matches:
                logger.warning("No valid URLs found in the input text")
                return user_message

            client = OpenAI(api_key=openai_key)
            
            summaries = []
            for link_text, url in url_matches:
                summary = summarize_url(client, url, topics, max_tokens, model)
                processed_summary = post_process_summary(summary, topics)
                summaries.append((link_text, url, processed_summary))
            
            result = insert_summaries(user_message, summaries)
            
            logger.info("Pipeline processing complete")
            return result
        except Exception as e:
            logger.error(f"Error in pipe function: {e}")
            return f"An error occurred while processing the request: {str(e)}"

    def get_config(self):
        """
        Return the configuration options for the pipeline, including the model dropdown.
        """
        logger.debug("Returning pipeline configuration")
        return {
            "OPENAI_API_KEY": {"type": "string", "value": self.valves.OPENAI_API_KEY},
            "TOPICS": {"type": "string", "value": self.valves.TOPICS},
            "MAX_TOKENS": {"type": "number", "value": self.valves.MAX_TOKENS},
            "MODEL": {"type": "select", "value": self.valves.MODEL, "options": self.available_models}
        }
