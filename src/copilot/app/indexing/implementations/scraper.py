import logging
from typing import List

from haystack.dataclasses import ByteStream
from haystack.components.fetchers import LinkContentFetcher

from indexing.base import BaseScraper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Scraper(BaseScraper):
    """
    A class used to scrap URLs from *.admin.ch websites.

    Attributes
    ----------
    fetcher : LinkContentFetcher
        An instance of LinkContentFetcher to fetch the content of URLs.

    Methods
    -------
    scrap_urls(url_list: List[str]) -> List[ByteStream]
        Scrapes the given URLs and returns the content as a list of ByteStreams.

    """

    def __init__(self):
        super().__init__()
        self.fetcher = LinkContentFetcher()

    def scrap_urls(self, urls: List[str]) -> List[ByteStream]:
        """
        Scrapes the given URLs and returns the content as a list of ByteStreams.

        Parameters
        ----------
        urls : List[str]
            A list of URLs to scrape.

        Returns
        -------
        List[ByteStream]
            A list of ByteStreams containing the content of the scraped URLs.
        """

        streams = self.fetcher.run(urls=urls)
        return streams["streams"]


scraper = Scraper()
