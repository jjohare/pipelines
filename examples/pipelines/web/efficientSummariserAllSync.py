"""
Efficient Web Summary Pipeline for OpenWebUI and Pipelines

This pipeline script integrates with OpenWebUI and Pipelines to extract URLs from unstructured text
and generate summaries of web pages using the OpenAI API with GPT-4-0-mini model.

Key features:
- URL extraction from unstructured text using regex
- Efficient web scraping using Playwright
- Content filtering to reduce irrelevant data
- Topic highlighting in summaries
- Integration of summaries back into the original text

Usage:
1. Set the OPENAI_API_KEY in the Valves configuration.
2. Set the TOPICS (comma-separated) in the Valves configuration.
3. Input unstructured text containing URLs in the user message.
4. The pipeline will extract URLs, scrape the web pages, generate summaries, and integrate them back into the text.
"""

import re
import sys
import logging
from typing import List
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Pipeline:
    class Valves(BaseModel):
        """
        Configuration options for the pipeline.
        These options can be set through the OpenWebUI interface.
        """
        OPENAI_API_KEY: str = ""  # OpenAI API key
        TOPICS: str = ""  # Comma-separated list of topics to be considered when generating summaries
        MAX_TOKENS: int = 2000  # Maximum number of tokens for each summary

    def __init__(self):
        self.name = "Efficient Web Summary Pipeline"
        self.valves = self.Valves()

    def setup_playwright(self):
        """
        Set up Playwright by launching a browser instance.
        This function is called during the pipeline startup process.
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto('https://example.com')  # Test navigation
                logging.info("Playwright setup successful.")
                browser.close()
        except Exception as e:
            logging.error(f"Playwright setup failed: {e}")
            sys.exit(1)  # Exit if setup fails

    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from unstructured text using regex.

        Args:
            text (str): Unstructured text potentially containing URLs.

        Returns:
            List[str]: A list of extracted URLs.
        """
        logging.info("Extracting URLs from input text.")
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+(?=[.,;:!?)\]}\s]|$)'
        urls = list(set(re.findall(url_pattern, text)))
        logging.debug(f"Extracted URLs: {urls}")
        return urls

    def scrape_and_filter_content(self, url: str) -> str:
        """
        Scrape the content of a given URL using Playwright and filter it.

        Args:
            url (str): The URL to scrape.

        Returns:
            str: Filtered text content of the web page.
        """
        logging.info(f"Scraping URL: {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove scripts, styles, and other unnecessary elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Extract main content
                main_content = soup.select_one('main, #content, .main-content, article')
                if main_content:
                    text = main_content.get_text(separator=' ', strip=True)
                else:
                    text = soup.get_text(separator=' ', strip=True)
                
                # Limit to first 1000 words
                words = text.split()[:1000]
                filtered_text = ' '.join(words)
                logging.info(f"Successfully scraped and filtered content for {url}")
                return filtered_text
            except Exception as e:
                logging.error(f"Error scraping {url}: {e}")
                return f"Error scraping content: {str(e)}"
            finally:
                browser.close()

    def generate_summary(self, url: str, content: str) -> str:
        """
        Generate a summary for the given URL and content using OpenAI's GPT-4-0-mini model.

        Args:
            url (str): The URL of the web page.
            content (str): The filtered content of the web page.

        Returns:
            str: The generated summary.
        """
        logging.info(f"Generating summary for {url}")
        client = OpenAI(api_key=self.valves.OPENAI_API_KEY)
        topics = [topic.strip() for topic in self.valves.TOPICS.split(",")]
        topics_str = ", ".join(topics)
        
        prompt = (
            f"Please summarize the following web page content, considering these topics: {topics_str}\n\n"
            f"URL: {url}\n\n"
            f"Content: {content}\n\n"
            f"Guidelines:\n"
            f"- Start with a hyphen and space ('- ').\n"
            f"- Use bullet points if appropriate, starting with a tab, hyphen, and space ('\\t- ').\n"
            f"- Include relevant topics inline within the summary.\n"
            f"- Use UK English spelling.\n"
            f"- If the content is irrelevant or seems to be an error page, mention that instead.\n"
            f"- Keep the summary concise, about {self.valves.MAX_TOKENS} tokens."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4-0-mini",  # Hard-coded model as requested
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes web pages."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.valves.MAX_TOKENS
            )
            summary = response.choices[0].message.content.strip()
            logging.info(f"Successfully generated summary for {url}")
            return summary
        except Exception as e:
            logging.error(f"Error generating summary for {url}: {e}")
            return f"Error generating summary: {str(e)}"

    def integrate_summaries(self, text: str, summaries: dict) -> str:
        """
        Integrate summaries back into the original text next to the respective URLs.

        Args:
            text (str): Original text containing URLs.
            summaries (dict): Summaries associated with each URL.

        Returns:
            str: Updated text with summaries integrated.
        """
        logging.info("Integrating summaries into the original text")
        for url, summary in summaries.items():
            summary_text = f"\n\n### Web Summary - Auto Generated\n{summary}\n"
            text = text.replace(url, f"{url}{summary_text}")
        return text

    def pipe(self, user_message: str, model_id: str = None, messages: List[dict] = None, body: dict = None) -> str:
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
        logging.info(f"Processing input in {self.name}")
        try:
            urls = self.extract_urls(user_message)
            
            if not urls:
                logging.warning("No valid URLs found in the input text.")
                return "No valid URLs found in the input text."
            
            logging.info(f"Found {len(urls)} URLs to process")
            
            summaries = {}
            for url in urls:
                content = self.scrape_and_filter_content(url)
                summary = self.generate_summary(url, content)
                summaries[url] = summary
            
            result = self.integrate_summaries(user_message, summaries)
            
            logging.info("Processing complete. Returning result.")
            return result
        except Exception as e:
            logging.error(f"Error in pipe function: {e}")
            return f"An error occurred while processing the request: {str(e)}"

    def get_config(self):
        """
        Return the configuration options for the pipeline.
        """
        return {
            "OPENAI_API_KEY": {"type": "string", "value": self.valves.OPENAI_API_KEY},
            "TOPICS": {"type": "string", "value": self.valves.TOPICS},
            "MAX_TOKENS": {"type": "number", "value": self.valves.MAX_TOKENS}
        }

# Expose the Pipeline class
pipeline = Pipeline()
pipeline.setup_playwright()
logging.info("Pipeline instance created and exposed.")
