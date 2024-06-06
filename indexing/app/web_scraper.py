import requests
from bs4 import BeautifulSoup
from typing import List, Optional
import logging
from lxml import etree
import re

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

if __name__ != '__main__':
    import queries


class WebScraper:
    def __init__(self, base_url: str, proxy: str = ''):
        self.base_url = base_url
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if proxy:
            disable_warnings(category=InsecureRequestWarning)
            self.session.verify = False
            self.session.proxies.update({"http": proxy})
            self.session.proxies.update({"https": proxy})

        self.answer = {
            'de': 'Antwort\n',
            'it': 'Rispondi\n',
            'fr': 'Réponse\n'
        }

    async def run(self, test: int = 0):
        """
        Asynchronously retrieves and processes FAQ data from 'https://faq.bsv.admin.ch' to insert into the database.

        The endpoint 'https://faq.bsv.admin.ch/sitemap.xml' is utilized to discover all relevant FAQ URLs. For each URL,
        the method extracts the primary question (denoted by the 'h1' tag) and its corresponding answer (within an
        'article' tag).
        Unnecessary boilerplate text will be removed for clarity and conciseness.

        Each extracted FAQ entry is then upserted (inserted or updated if already exists) into the database, with
        detailed logging to track the operation's progress and identify any errors.

        Returns a confirmation message upon successful completion of the process.

        TODO:
        - Consider implementing error handling at a more granular level to retry failed insertions or updates, enhancing
        the robustness of the data ingestion process.
        - Explore optimization opportunities in text extraction and processing to improve efficiency and reduce runtime,
        especially for large sitemaps.
        """
        self.logger.info(f"Beginne Datenextraktion für: {self.base_url}")
        urls = self.get_sitemap_urls()

        if test:
            urls = urls[:test]

        for url in urls:
            lang, h1, article = self.extract_article(url)

            if h1 and test:
                self.logger.info("--------------------")
                self.logger.info(f"url: {url}")
                self.logger.info(f"question: {h1}")
                self.logger.info(f"answer: {article}")
                self.logger.info(f"language: {lang}")

            elif h1 and article:
                self.logger.info(f"extract: {url}")
                info, rid = await queries.update_or_insert(url, h1, article, lang)
                self.logger.info(f"{info}: {url}")

        self.logger.info(f"Done! {len(urls)} wurden verarbeitet.")
        return urls

    def _get_response(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        """Send a GET request and return the response object."""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None

    def get_sitemap_urls(self) -> List[str]:
        """Extract URLs from the sitemap."""
        response = self._get_response(self.base_url)

        path = []
        if response is not None:
            root = etree.fromstring(response.content)
            namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            path = root.xpath("//sitemap:loc", namespaces=namespace)
        urls = [url.text for url in path]

        return urls

    def extract_article(self, url: str):
        response = self._get_response(url)
        if not response:
            return ''

        soup = BeautifulSoup(response.text, 'lxml')

        extracted = []
        for tag in ['html', 'h1', 'article']:
            element = soup.find(tag)
            
            if not element:
                extracted.append('')
            elif tag == 'html':
                extracted.append(element['lang'] if element.has_attr('lang') else '')
            else:
                text = element.get_text()
                if tag == 'article':
                    text = text.replace(self.answer.get(extracted[0], ''), '')
                text = re.sub(r"((\r\n|\r|\n)\s*){2,}", "\n\n", text.strip())
                extracted.append(text)

        return extracted[0], extracted[1], extracted[2]


if __name__ == '__main__':
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description='Run WebScraper demo')
    parser.add_argument('--sitemap', type=str,
                        default='https://faq.bsv.admin.ch/sitemap.xml',
                        help='The sitemap URL of the website to scrape (default: https://faq.bsv.admin.ch/sitemap.xml)')
    # noinspection HttpUrlsUsage
    parser.add_argument('--proxy', type=str,
                        default='',
                        help='The proxy address if you are using one (example: http://your-proxy-url.com:0000')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    scraper = WebScraper(args.sitemap, args.proxy)
    asyncio.run(scraper.run(9))