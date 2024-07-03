import logging
from typing import List
from datetime import datetime

from haystack.dataclasses import Document, ByteStream
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument, PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter

from components.indexing.base import BaseScraper, BaseParser, BaseIndexer
from components.embedding.factory import EmbeddingFactory

from components.indexing import parsing
from indexing import queries

# Load env variables
from config.base_config import rag_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HaystackScraper(BaseScraper):
    """
    Class used to scrape URLs and extract content from them.

    Attributes
    ----------
    fetcher : LinkContentFetcher
        An instance of LinkContentFetcher to fetch the content of URLs.

    Methods
    -------
    scrap_urls(url_list: List[str]) -> List[ByteStream]
        Scrapes the given URLs and returns the content as a list of ByteStreams.

    async html_content_from_sitemap(sitemap_url: str) -> List[ByteStream]
        Extracts HTML content from the given sitemap URL. It fetches the sitemap, extracts URLs from it,
        and then scrapes the HTML content from these URLs.

    async pdf_content_from_sitemap(sitemap_url: str) -> List[ByteStream]
        Extracts PDF content from the given sitemap URL. It fetches the sitemap, extracts URLs from it,
        scrapes the HTML content from these URLs, extracts PDF paths from the HTML content, and then
        scrapes the PDF content from these paths.
    """

    def __init__(self):
        super().__init__()
        self.fetcher = LinkContentFetcher()

    def scrap_urls(self, url_list: List[str]) -> List[ByteStream]:
        """
        Scrapes the given URLs and returns the content as a list of ByteStreams.

        Parameters
        ----------
        url_list : List[str]
            A list of URLs to scrape.

        Returns
        -------
        List[ByteStream]
            A list of ByteStreams containing the content of the scraped URLs.
        """
        streams = self.fetcher.run(urls=url_list)
        return streams["streams"]

    async def html_content_from_sitemap(self, sitemap_url: str) -> List[ByteStream]:
        """
        Extracts HTML content from the given sitemap URL. It fetches the sitemap, extracts URLs from it,
        and then scrapes the HTML content from these URLs.

        Parameters
        ----------
        sitemap_url : str
            The URL of the sitemap to extract HTML content from.

        Returns
        -------
        List[ByteStream]
            A list of ByteStreams containing the HTML content extracted from the sitemap.
        """
        # Get sitemap
        sitemap = await self.fetch(sitemap_url)

        # Extract URLs from sitemap
        url_list = HaystackParser.extract_urls_from_xml(sitemap)

        # Scrap HTML from URLs
        content = self.scrap_urls(url_list)

        return content

    async def pdf_content_from_sitemap(self, sitemap_url: str) -> List[ByteStream]:
        """
        Extracts PDF content from the given sitemap URL. It fetches the sitemap, extracts URLs from it,
        scrapes the HTML content from these URLs, extracts PDF paths from the HTML content, and then
        scrapes the PDF content from these paths.

        Parameters
        ----------
        sitemap_url : str
            The URL of the sitemap to extract PDF content from.

        Returns
        -------
        List[ByteStream]
            A list of ByteStreams containing the PDF content extracted from the sitemap.
        """
        # Get sitemap
        sitemap = await self.fetch(sitemap_url)

        # Extract URLs of memento sections from sitemap (Allgemeines, Beiträge, etc.)
        url_list = HaystackParser.extract_urls_from_html(sitemap)

        # Get HTML from each memento section link
        response = self.scrap_urls(url_list)

        soups = []
        for res in response:
            soups.append(parsing.get_soup(res.data, "html.parser"))

        # Get PDF paths from each memento section
        pdf_paths = []
        for soup in soups:
            pdf_paths.extend(parsing.get_pdf_paths(soup))

        # Scrap PDFs from each memento section
        pdf_urls = ["https://ahv-iv.ch" + pdf_path for pdf_path in pdf_paths]

        # Add "it", "fr" pdf paths
        pdf_urls.extend([pdf_url.replace(".d", ".f") for pdf_url in pdf_urls])
        pdf_urls.extend([pdf_url.replace(".d", ".i") for pdf_url in pdf_urls])

        content = self.scrap_urls(pdf_urls)

        return content


class HaystackParser(BaseParser):
    """
    Class used to parse, clean and split documents.

    Attributes
    ----------
    html_converter : HTMLToDocument
        An instance of HTMLToDocument to convert HTML content to Document objects.

    pdf_converter : PyPDFToDocument
        An instance of PyPDFToDocument to convert PDF content to Document objects.

    cleaner : DocumentCleaner
        An instance of DocumentCleaner to clean documents.

    splitter : DocumentSplitter
        An instance of DocumentSplitter to split documents into chunks.

    Methods
    -------
    extract_urls_from_xml(sitemap: bytes) -> List[str]
        Extracts URLs from the given XML sitemap.

    extract_urls_from_html(html: bytes) -> List[str]
        Extracts URLs from the given HTML content.

    convert_html_to_documents(content: List[ByteStream]) -> List[Document]
        Converts HTML content to Document objects.

    convert_pdf_to_documents(content: List[ByteStream]) -> List[Document]
        Converts PDF content to Document objects.

    clean_documents(documents: List[Document]) -> List[Document]
        Cleans the given documents.

    split_documents(documents: List[Document]) -> List[Document]
        Splits the given documents into chunks.
    """
    def __init__(self):
        """
        Initializes the HaystackParser with HTMLToDocument, PyPDFToDocument, DocumentCleaner, and DocumentSplitter instances.
        """
        self.html_converter = HTMLToDocument()
        self.pdf_converter = PyPDFToDocument()
        self.cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=False
        )
        self.splitter = DocumentSplitter(
            split_by="passage",
            split_length=1,
            split_overlap=0
        )

    @staticmethod
    def extract_urls_from_xml(sitemap: bytes) -> List[str]:
        """
        Extracts URLs from the given XML sitemap.

        Parameters
        ----------
        sitemap : bytes
            The XML sitemap to extract URLs from.

        Returns
        -------
        List[str]
            A list of URLs extracted from the sitemap.
        """
        soup = parsing.get_soup(sitemap, "xml")

        url_list = []

        # Iterate over all 'url' elements and extract 'loc' URLs
        for url_element in soup.find_all('url'):
            loc_element = url_element.find('loc')
            if loc_element and loc_element.text:
                url_list.append(loc_element.text)

        return url_list

    @staticmethod
    def extract_urls_from_html(html: bytes) -> List[str]:
        """
        Extracts URLs from the given HTML content.

        Parameters
        ----------
        html : bytes
            The HTML content to extract URLs from.

        Returns
        -------
        List[str]
            A list of URLs extracted from the HTML content.
        """
        soup = parsing.get_soup(html, "html.parser")

        # Find all "a" tags with href containing "Merkblätter/" (and subsequent path)
        links = soup.find_all(parsing.contains_memento_url)

        # Remove duplicate links
        links = parsing.remove_duplicate_links(links)

        links = [link["href"] for link in links]

        return links

    def convert_html_to_documents(self, content: List[ByteStream]) -> List[Document]:
        """
        Converts HTML content to Document objects.

        Parameters
        ----------
        content : List[ByteStream]
            The HTML content to convert to Document objects.

        Returns
        -------
        List[Document]
            A list of Document objects created from the HTML content.
        """
        return self.html_converter.run(sources=content)

    def convert_pdf_to_documents(self, content: List[ByteStream]) -> List[Document]:
        """
        Converts PDF content to Document objects.

        Parameters
        ----------
        content : List[ByteStream]
            The PDF content to convert to Document objects.

        Returns
        -------
        List[Document]
            A list of Document objects created from the PDF content.
        """
        return self.pdf_converter.run(sources=content)

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Cleans the given documents.

        Parameters
        ----------
        documents : List[Document]
            The documents to clean.

        Returns
        -------
        List[Document]
            A list of cleaned documents.
        """
        # Remove empty documents
        documents = super().clean_documents(documents)

        return self.cleaner.run(documents=documents)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits the given documents into chunks.

        Parameters
        ----------
        documents : List[Document]
            The documents to split into chunks.

        Returns
        -------
        List[Document]
            A list of documents split into chunks.
        """
        # Remove empty documents
        documents = super().clean_documents(documents)

        return self.splitter.run(documents=documents)


class HaystackIndexer(BaseIndexer):
    """
    Class used to index documents into a VectorDB.

    Attributes
    ----------
    scraper : HaystackScraper
        An instance of HaystackScraper to scrape URLs and extract content from them.
    parser : HaystackParser
        An instance of HaystackParser to parse and clean documents.
    embedding_client : EmbeddingClient
        An instance of EmbeddingClient to embed documents.

    Methods
    -------
    index_html_from_sitemap(sitemap_url: str, language: str = "de") -> dict
        Indexes HTML content from the given sitemap URL into the VectorDB.
    index_pdfs_from_sitemap(sitemap_url: str, language: str = "de") -> dict
        Indexes PDF content from the given sitemap URL into the VectorDB.
    """
    def __init__(self):
        self.scraper = HaystackScraper()
        self.parser = HaystackParser()
        self.embedding_client = EmbeddingFactory.get_embedding_client(rag_config["embedding"]["model"])

    # TO DO: index multiple languages: ["de", "fr", "it"]
    async def index_html_from_sitemap(self, sitemap_url: str, language: str = "de") -> dict:
        """
        Indexes HTML content from the given sitemap URL into the VectorDB.

        Parameters
        ----------
        sitemap_url : str
            The URL of the sitemap to index HTML content from.
        language : str, optional
            The language of the HTML content, by default "de".

        Returns
        -------
        dict
            A dictionary containing a message indicating the success of the indexing operation.
        """
        # Get HTML content from all sitemap URLs
        content = await self.scraper.html_content_from_sitemap(sitemap_url)

        # Convert HTML content to Document objects
        documents = self.parser.convert_html_to_documents(content)

        # Clean documents
        documents = self.parser.clean_documents(documents["documents"])

        # Split documents into chunks
        chunks = self.parser.split_documents(documents["documents"])

        # TO DO: refactor embedding logic to embed from documents (add from_documents method)
        # Embed documents
        str_chunks = [doc.content for doc in chunks["documents"]]
        document_embeddings = self.embedding_client.embed_documents(str_chunks)
        document_embeddings = [e.embedding for e in document_embeddings]
        urls = [doc.meta["url"] for doc in chunks["documents"]]

        # Upsert documents into VectorDB
        for embedding, doc, url in zip(document_embeddings, str_chunks, urls):
            date = datetime.now()
            await queries.insert_rag(str(embedding), doc, url, date, date)

        return {"content": f"{sitemap_url}: RAG data indexed successfully"}

    async def index_pdfs_from_sitemap(self, sitemap_url: str, language: str = "de") -> dict:
        """
        Indexes PDF content from the given sitemap URL into the VectorDB.

        Parameters
        ----------
        sitemap_url : str
            The URL of the sitemap to index PDF content from.
        language : str, optional
            The language of the PDF content, by default "de".

        Returns
        -------
        dict
            A dictionary containing a message indicating the success of the indexing operation.
        """
        # Get PDF content from all sitemap memento links
        content = await self.scraper.pdf_content_from_sitemap(sitemap_url)

        # Convert PDF content to Document objects
        documents = self.parser.convert_pdf_to_documents(content)

        # Clean documents
        documents = self.parser.clean_documents(documents["documents"])

        # Split documents into chunks
        chunks = self.parser.split_documents(documents["documents"])

        # TO DO: refactor embedding logic to embed from documents (add from_documents method)
        # Embed documents
        str_chunks = [doc.content for doc in chunks["documents"]]
        document_embeddings = self.embedding_client.embed_documents(str_chunks)
        document_embeddings = [e.embedding for e in document_embeddings]
        urls = [doc.meta["url"] for doc in chunks["documents"]]

        # Upsert documents into VectorDB
        for embedding, doc, url in zip(document_embeddings, str_chunks, urls):
            date = datetime.now()
            await queries.insert_rag(str(embedding), doc, url, date, date)

        return {"content": f"{sitemap_url}: PDF RAG data indexed successfully"}