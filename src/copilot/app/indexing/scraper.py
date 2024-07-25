import requests
from bs4 import BeautifulSoup
from typing import List, Optional
import logging
from lxml import etree
import re

if __name__ != '__main__':
    from . import queries

SITEMAP_URL = 'http://www.sitemaps.org/schemas/sitemap/0.9'

ANSWER = {
    'en': 'Answer',
    'de': 'Antwort',
    'it': 'Rispondi',
    'fr': 'Réponse'
}


class Scraper:
    """
    Scraper class that can extract FAQ questions from a specified website.

    Parameters
    ==========
    base_url : str
        sitemap URL of the website to scrap
    proxy : str, optional
        Proxy URL if necessary
    """

    def __init__(self, base_url: str, proxy: str = None):
        self.base_url = base_url
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)

        if proxy:
            self.session.proxies.update({"http": proxy})
            self.session.proxies.update({"https": proxy})

    async def run(self, test: int = 0):
        """
        Retrieves and processes FAQ data from `base_url` to insert into the database.

        Each extracted FAQ entry is then upserted (inserted or updated if already exists) into the database, with
        detailed logging to track the operation's progress and identify any errors.

        If `test`>0, then extract only the specified number of articles and log them instead of upserting them.

        Log a confirmation message upon successful completion of the process.

        .. todo::
            - Consider implementing error handling at a more granular level to retry failed insertions or updates, enhancing the robustness of the data ingestion process.
            - Explore optimization opportunities in text extraction and processing to improve efficiency and reduce runtime, especially for large sitemaps.

        Parameters
        ==========
        test : int, default 0
            Number of articles to extract as a test

        Returns
        =======
        list of str
            list of urls which got extracted
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
        """
        Send a GET request and return the response object.
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None

    def get_sitemap_urls(self) -> List[str]:
        """
        Extract URLs from the sitemap. The endpoint '/sitemap.xml' is used to discover all relevant FAQ URLs.
        """
        response = self._get_response(self.base_url)

        path = []
        if response is not None:
            root = etree.fromstring(response.content)
            namespace = {'sitemap': SITEMAP_URL}
            path = root.xpath("//sitemap:loc", namespaces=namespace)
        urls = [url.text for url in path]

        return urls

    def extract_article(self, url: str):
        """
        Given an url, extracts the primary question (denoted by the 'h1' tag) and its corresponding answer (within an
        'article' tag).

        Unnecessary boilerplate text will be removed for clarity and conciseness.

        Parameters
        ----------
        url : str
            URL of the website where the article needs to be extracted.

        Returns
        -------
        str, str, str
            The article language, its question and its answer.
        """
        response = self._get_response(url)
        if not response:
            return '', '', ''

        soup = BeautifulSoup(response.text, 'lxml')

        extracted = []
        for tag in ['html', 'h1', 'article']:
            element = soup.find(tag)

            # tag not found
            if not element:
                extracted.append(None)

            # extract lang
            elif tag == 'html':
                extracted.append(element.get('lang'))

            # extract text
            else:
                text = element.get_text()
                if tag == 'article':
                    text = text.replace(ANSWER.get(extracted[0]), '')  # remove ANSWER title
                    text = re.sub(r"((\r\n|\r|\n)\s*){2,}", "\n\n", text)  # remove double linebreak
                extracted.append(text.strip())

        return extracted[0], extracted[1], extracted[2]  # lang, h1, article


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

    scraper = Scraper(args.sitemap, args.proxy)
    asyncio.run(scraper.run(9))
