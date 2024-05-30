import requests
from bs4 import BeautifulSoup
from typing import List, Optional
import logging
from lxml import etree

class WebScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_response(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
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
        response = self.get_response(self.base_url)
        if not response:
            return []

        root = etree.fromstring(response.content)
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [url.text for url in root.xpath("//sitemap:loc", namespaces=namespace)]

        return urls

    def extract_text_from_tag(self, url: str, tag: str) -> str:
        """Extract text from a specific tag in a web page."""
        response = self.get_response(url)
        if not response:
            return ''

        soup = BeautifulSoup(response.text, 'lxml')
        body = soup.find(tag)
        return '\n'.join(line.strip() for line in body.get_text().splitlines() if line) if body else ''

    def detect_language(self, url: str) -> str:
        """Detect the language of a website based on the 'lang' attribute in the HTML tag."""
        response = self.get_response(url)
        if not response:
            return ''

        soup = BeautifulSoup(response.text, 'lxml')
        html_tag = soup.find('html')
        return html_tag['lang'] if html_tag and html_tag.has_attr('lang') else ''

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    sitemap_url = 'https://faq.bsv.admin.ch/sitemap.xml'
    scraper = WebScraper(sitemap_url)
    
    scraper.logger.info(f"Beginne Datenextraktion für: {sitemap_url}")
    urls = scraper.get_sitemap_urls()
    
    for url in urls[:9]:
        extracted_h1 = scraper.extract_text_from_tag(url, 'h1')
        extracted_article = scraper.extract_text_from_tag(url, 'article')
        language = scraper.detect_language(url)

        # Efficient text processing
        for term in ['Antwort\n', 'Rispondi\n', 'Réponse\n']:
            extracted_article = extracted_article.replace(term, '')

        if extracted_h1:
            scraper.logger.info("--------------------")
            scraper.logger.info(f"url: {url}")
            scraper.logger.info(f"question: {extracted_h1}")
            scraper.logger.info(f"answer: {extracted_article}")
            scraper.logger.info(f"language: {language}")
