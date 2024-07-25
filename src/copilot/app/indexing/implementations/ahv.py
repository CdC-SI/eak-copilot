import logging
from typing import List, Any
from urllib.parse import unquote

from haystack.components.converters import PyPDFToDocument

from bs4 import BeautifulSoup

from indexing.base import BaseParser, BaseIndexer
from utils.embedding import get_embedding

from indexing import queries
from .scraper import scraper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AHVParser(BaseParser):
    """
    A class used to parse and clean documents.

    Attributes
    ----------
    pdf_converter : PyPDFToDocument
        An instance of PyPDFToDocument to convert PDF content to Document objects.

    cleaner : DocumentCleaner
        An instance of DocumentCleaner to clean documents.

    splitter : DocumentSplitter
        An instance of DocumentSplitter to split documents into chunks.

    Methods
    -------
    parse_html(html: bytes) -> List[str]
        Extracts URLs from the given HTML content.

    convert_pdf_to_documents(content: List[ByteStream]) -> List[Document]
        Converts PDF content to Document objects.

    clean_documents(documents: List[Document]) -> List[Document]
        Cleans the given documents.

    split_documents(documents: List[Document]) -> List[Document]
        Splits the given documents into chunks.
    """
    def __init__(self):
        super().__init__()

    def contains_tag(self, tag):
        """
        Checks if a tag contains a memento URL.

        Parameters
        ----------
        tag : bs4.element.Tag
            The tag to check.

        Returns
        -------
        bool
            True if the tag contains a memento URL, False otherwise.
        """
        if tag.name == "a" and "href" in tag.attrs:
            href = tag["href"]
            decoded_href = unquote(href)
            return "Merkblätter/" in decoded_href
        return False

    def get_pdf_paths(self, soup):
        """
        Extracts the paths of PDF documents from a BeautifulSoup object.

        Parameters
        ----------
        soup : BeautifulSoup
            The BeautifulSoup object to extract PDF paths from.

        Returns
        -------
        list of str
            The list of PDF paths.
        """
        # TO DO: re-scrap links which are not PDFs such as:
        # pdf_paths = [a["href"] for a in soup.find_all("a", {"class": "co-document-content"})]
        pdf_paths = [a["href"] for a in soup.find_all("a", {"class": "co-document-content"}) if "/p/" in a["href"]]
        return pdf_paths

    def parse_urls(self, content: str) -> List[str]:
        soup = BeautifulSoup(content, features="html.parser")

        # Find all "a" tags with href containing "Merkblätter/" (and subsequent path)
        links = soup.find_all(self.contains_tag)
        links = self.remove_duplicate_links(links)

        url_list = [link["href"] for link in links]

        return url_list

    def convert_to_documents(self, content: List[Any]) -> List[Any]:
        return PyPDFToDocument().run(sources=content)


class AHVIndexer(BaseIndexer):
    """
    A class used to index PDF Merkblätter documents into a VectorDB.

    Attributes
    ----------
    scraper : Scraper
        An instance of Scraper to scrape URLs and extract content from them.
    parser : Parser
        An instance of Parser to parse and clean documents.

    Methods
    -------
    index(sitemap_url: str) -> dict
        Scraps, parses and indexes PDF content from the given sitemap URL into the VectorDB.
    """
    async def index(self, sitemap_url: str) -> dict:
        # Get sitemap
        sitemap = await self.scraper.fetch(sitemap_url)

        # Extract URLs of memento sections from sitemap (Allgemeines, Beiträge, etc.)
        url_list = self.parser.parse_urls(sitemap)

        # Get HTML from each memento section link
        content = self.scraper.scrap_urls(url_list)

        soups = []
        for page in content:
            soups.append(BeautifulSoup(page.data, features="html.parser"))

        # Get PDF paths from each memento section
        pdf_paths = []
        for soup in soups:
            pdf_paths.extend(self.parser.get_pdf_paths(soup))

        # Scrap PDFs from each memento section
        pdf_urls = ["https://ahv-iv.ch" + pdf_path for pdf_path in pdf_paths]

        # Add "it", "fr" pdf paths
        pdf_urls.extend([pdf_url.replace(".d", ".f") for pdf_url in pdf_urls])
        pdf_urls.extend([pdf_url.replace(".d", ".i") for pdf_url in pdf_urls])

        content = self.scraper.scrap_urls(pdf_urls)

        # Convert PDF content to Document objects
        documents = self.parser.convert_to_documents(content)

        # Remove empty documents
        documents = self.parser.remove_empty_documents(documents["documents"])

        # Clean documents
        documents = self.parser.clean_documents(documents)

        # Split documents into chunks
        chunks = self.parser.split_documents(documents["documents"])

        # TO DO: refactor embedding logic to embed from documents (add from_documents method)
        # Upsert documents into VectorDB
        for doc in chunks["documents"]:
            text = doc.content
            embedding = get_embedding(text)
            url = doc.meta["url"]
            await queries.insert_rag(embedding, text, url)

        return {"content": f"{sitemap_url}: PDF RAG data indexed successfully"}


ahv_indexer = AHVIndexer(
    scraper=scraper,
    parser=AHVParser()
)
