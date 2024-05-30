import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, status, Body
from fastapi.responses import Response

# Load env variables
from config.base_config import indexing_config, indexing_app_config

# Load utility functions
from utils.db import get_db_connection, check_db_connection
from utils.embedding import get_embedding
from indexing.app.web_scraper import WebScraper

# Load models
from rag.app.models import ResponseBody

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create an instance of FastAPI
app = FastAPI(**indexing_app_config)

async def init_rag_vectordb():

    texts = [
        ("Comment déterminer mon droit aux prestations complémentaires? Vous pouvez déterminer votre droit aux prestations de façon simple et rapide, grâce au calculateur de prestations complémentaires en ligne : www.ahv-iv.ch/r/calculateurpc\n\n Le calcul est effectué de façon tout à fait anonyme. Vos données ne sont pas enregistrées. Le résultat qui en ressort constitue une estimation provisoire fondée sur une méthode de calcul simplifiée. Il s’agit d’une estimation sans engagement, qui ne tient pas lieu de demande de prestation et n’implique aucun droit. Le calcul n’est valable que pour les personnes qui vivent à domicile. Si vous résidez dans un home, veuillez vous adresser à sa direction, qui vous fournira les renseignements appropriés au sujet des prestations complémentaires.", "https://www.ahv-iv.ch/p/5.02.f"),
        ("Quand des prestations complémentaires sont-elles versées ? Lorsque la rente AVS ne suffit pas. Les rentes AVS sont en principe destinées à couvrir les besoins vitaux d'un assuré. Lorsque ces ressources ne sont pas suffisantes pour assurer la subsistance des bénéficiaires de rentes AVS, ceux-ci peuvent prétendre à des prestations complémentaires (PC).\n\nLe versement d'une telle prestation dépend du revenu et de la fortune de chaque assuré. Les PC ne sont pas des prestations d'assistance mais constituent un droit que la personne assurée peut faire valoir en toute légitimité lorsque les conditions légales sont réunies.", "https://www.ahv-iv.ch/fr/Assurances-sociales/Assurance-vieillesse-et-survivants-AVS/Prestations-complémentaires"),
        ("Quand le droit à une rente de vieillesse prend-il naissance ? Lorsque la personne assurée atteint l'âge de référence. Le droit à la rente de vieillesse prend naissance le premier jour du mois qui suit celui au cours duquel l'ayant droit atteint l'âge ordinaire de référence et s'éteint à la fin du mois de son décès. L'âge ordinaire de la retraite est fixé à 64 ans pour les femmes et à 65 ans pour les hommes. A partir du 1er janvier 2025, l'âge est fixé à 65 ans pour les hommes, tandis que pour les femmes, il est actuellement fixé à 64 ans et sera relevé de trois mois par an. À partir de 2028, l’âge de référence sera le même, à savoir 65 ans, pour les hommes et les femmes.", "https://www.ahv-iv.ch/fr/Assurances-sociales/Assurance-vieillesse-et-survivants-AVS/Rentes-de-vieillesse#qa-792"),
        ("Qu'est-ce qui change avec AVS 21? Le 25 septembre 2022, le peuple et les cantons ont accepté la réforme AVS 21 et assuré ainsi un financement suffisant de l’AVS jusqu’à l’horizon 2030. La modification entrera en vigueur le 1er janvier 2024. La réforme comprenait deux objets : la modification de la loi sur l’assurance-vieillesse et survivants (LAVS) et l’arrêté fédéral sur le financement additionnel de l’AVS par le biais d’un relèvement de la TVA. Les deux objets étaient liés. Ainsi, le financement de l’AVS et le niveau des rentes seront garantis pour les prochaines années. L’âge de référence des femmes sera relevé à 65 ans, comme pour les hommes, le départ à la retraite sera flexibilisé et la TVA augmentera légèrement. La stabilisation de l’AVS comprend quatre mesures : \n\n• harmonisation de l’âge de la retraite (à l’avenir «âge de référence») des femmes et des hommes à 65 ans\n• mesures de compensation pour les femmes de la génération transitoire\n• retraite flexible dans l’AVS\n• financement additionnel par le relèvement de la TVA", "https://www.ahv-iv.ch/p/31.f"),
        ("Que signifie l'âge de la retraite flexible ? La rente peut être anticipée ou ajournée. Anticipation de la rente : Femmes et hommes peuvent anticiper la perception de leur rente dès le premier jour du mois qui suit leur 63e anniversaire. Les femmes nées entre 1961 et 1969 pourront continuer à anticiper leur rente à 62 ans. Leur situation est régie par des dispositions transitoires spéciales. Pour plus d’informations à ce sujet, veuillez vous adresser à votre caisse de compensation. Durant la période d'anticipation, il n'existe pas de droit à une rente pour enfant. Ajournement de la rente : Les personnes qui ajournent leur retraite d'au moins un an et de cinq ans au maximum bénéficient d'une rente de vieilesse majorée d'une augmentation pendant toute la durée de leur retraite. Combinaison : Il est également possible de combiner l'anticipation et l'ajournement. Une partie de la rente de vieillesse peut être anticipée et une partie peut être ajournée une fois l'âge de référence atteint. Le montant de la réduction ou de la majoration de la rente est fixé selon le principe des calculs actuariels. Dans le cadre d'un couple, il est possible que l'un des conjoints anticipe son droit à la rente alors que l'autre l'ajourne.", "https://www.ahv-iv.ch/fr/Assurances-sociales/Assurance-vieillesse-et-survivants-AVS/Rentes-de-vieillesse#qa-1137"),
    ]

    conn = await get_db_connection()

    try:
        for text in texts:

            # Get the embedding vector
            embedding = get_embedding(text[0])[0]["embedding"]

            await conn.execute(
                "INSERT INTO embeddings (embedding, text, url, created_at, modified_at) VALUES ($1, $2, $3, $4, $5)",
                str(embedding), text[0], text[1], datetime.now(),  datetime.now()
            )

    except Exception as e:
        await conn.close()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if conn:
            await conn.close()

    return {"content": "RAG data indexed successfully"}

async def init_faq_vectordb():

    texts = [
        ('https://www.eak.admin.ch/eak/de/home/dokumentation/pensionierung/beitragspflicht.html', 'Wann endet meine AHV-Beitragspflicht?', 'Mit der Reform AHV 21 wird ein einheitliches Rentenalter von 65 Jahren für Mann und Frau eingeführt. Dieses bildet die Bezugsgrösse für die flexible Pensionierung und wird deshalb neu als Referenzalter bezeichnet. Die Beitragspflicht endet, wenn Sie das Referenzalter erreicht haben. Die Beitragspflicht bleibt auch im Falle einer frühzeitigen Pensionierung resp. eines Vorbezugs der AHV-Rente bis zum Erreichen des Referenzalters bestehen.', 'de'),
        ('https://www.eak.admin.ch/eak/de/home/dokumentation/pensionierung/beitragspflicht.html', 'Ich arbeite Teilzeit (weniger als 50 Prozent). Warum muss ich trotzdem AHV-Beiträge wie Nichterwerbstätige zahlen?', 'Die Beitragspflicht entfällt, wenn Ihre bereits bezahlten Beiträge den Mindestbeitrag (bei Verheirateten und in eingetragener Partnerschaft lebenden Personen den doppelten Mindestbeitrag) und die Hälfte der von Nichterwerbstätigen geschuldeten Beiträge erreichen. Für die Befreiung von der Beitragspflicht müssen beide Voraussetzungen erfüllt sein.', 'de'),
        ('https://www.eak.admin.ch/eak/de/home/dokumentation/pensionierung/beitragspflicht.html', 'Ich bin vorpensioniert, mein Partner bzw. meine Partnerin arbeitet jedoch weiter. Muss ich trotzdem Beiträge wie Nichterwerbstätige bezahlen?', 'Sie müssen nur dann keine eigenen Beiträge bezahlen, wenn Ihr Partner bzw. Ihre Partnerin im Sinne der AHV dauerhaft voll erwerbstätig sind und seine oder ihre Beiträge aus der Erwerbstätigkeit (inklusive Arbeitgeberbeiträge) den doppelten Mindestbeitrag erreichen. Ist Ihr Partner bzw. Ihre Partnerin nicht dauerhaft voll erwerbstätig, müssen sie nebst dem doppelten Mindestbeitrag auch die Hälfte der von nichterwerbstätigen Personen geschuldeten Beiträge erreichen (siehe vorheriger Punkt).', 'de'),
        ('https://www.eak.admin.ch/eak/de/home/dokumentation/pensionierung/beitragspflicht.html', 'Was bedeutet «im Sinne der AHV dauerhaft voll erwerbstätig»?', 'Als dauerhaft voll erwerbstätig gilt, wer während mindestens neun Monaten pro Jahr zu mindestens 50 Prozent der üblichen Arbeitszeit erwerbstätig ist.', 'de'),
        ('https://www.eak.admin.ch/eak/de/home/dokumentation/pensionierung/beitragspflicht.html', 'Wie viel AHV/IV/EO-Beiträge muss ich als nichterwerbstätige Person bezahlen?', 'Die Höhe der Beiträge hängt von Ihrer persönlichen finanziellen Situation ab. Als Grundlage für die Berechnung der Beiträge dienen das Vermögen und das Renteneinkommen (z. B. Renten und Pensionen aller Art, Ersatzeinkommen wie Kranken- und Unfalltaggelder, Alimente, regelmässige Zahlungen von Dritten usw.). Bei Verheirateten und in eingetragener Partnerschaft lebenden Personen bemessen sich die Beiträge – ungeachtet des Güterstands – nach der Hälfte des ehelichen bzw. partnerschaftlichen Vermögens und Renteneinkommens. Es ist nicht möglich, freiwillig höhere Beiträge zu bezahlen.', 'de'),
        ('https://www.eak.admin.ch/eak/de/home/dokumentation/pensionierung/beitragspflicht.html', 'Wie bezahle ich die Beiträge als Nichterwerbstätiger oder Nichterwerbstätige?', 'Sie bezahlen für das laufende Beitragsjahr Akontobeiträge, welche die Ausgleichskasse gestützt auf Ihre Selbstangaben provisorisch berechnet. Die Akontobeiträge werden jeweils gegen Ende jedes Quartals in Rechnung gestellt (für jeweils 3 Monate). Sie können die Rechnungen auch mit eBill begleichen. Die Anmeldung für eBill erfolgt in Ihrem Finanzportal. Kunden von PostFinance können die Rechnungen auch mit dem Lastschriftverfahren Swiss Direct Debit (CH-DD-Lastschrift) bezahlen. Über definitiv veranlagte Steuern wird die Ausgleichskasse von den kantonalen Steuerverwaltungen mit einer Steuermeldung informiert. Gestützt auf diese Steuermeldung werden die Beiträge für das entsprechende Beitragsjahr definitiv verfügt und mit den geleisteten Akontobeiträgen verrechnet.', 'de'),
        ('https://www.eak.admin.ch/eak/fr/home/dokumentation/pensionierung/beitragspflicht.html', 'Jusqu’à quand dois-je payer des cotisations AVS ?', 'La réforme AVS 21 instaure un même âge de la retraite pour les femmes et les hommes, soit 65 ans. Cet âge servira de valeur de référence pour un départ à la retraite flexible et sera donc désormais appelé âge de référence. L’obligation de cotiser prend fin lorsque vous atteignez l’âge de référence. Lors d’un départ à la retraite anticipée ou si le versement de la rente AVS est anticipé, l’obligation de cotiser est maintenue jusqu’à l’âge de référence.', 'fr'),
        ('https://www.eak.admin.ch/eak/fr/home/dokumentation/pensionierung/beitragspflicht.html', 'Je travaille à temps partiel (moins de 50 %). Pourquoi dois-je quand même payer des cotisations AVS comme les personnes sans activité lucrative ?', 'Vous n’êtes pas tenu(e) de cotiser si les cotisations versées avec votre activité lucrative atteignent la cotisation minimale (le double de la cotisation minimale pour les personnes mariées et les personnes vivant en partenariat enregistré) et représentent plus de la moitié des cotisations dues en tant que personne sans activité lucrative. Pour être exempté de l’obligation de cotiser, vous devez remplir ces deux conditions.', 'fr'),
        ('https://www.eak.admin.ch/eak/fr/home/dokumentation/pensionierung/beitragspflicht.html', 'Je suis préretraité(e), mais mon/ma conjoint(e) continue de travailler. Dois-je quand même payer des cotisations comme si je n’avais pas d’activité professionnelle ?', 'Si votre conjoint(e) exerce durablement une activité lucrative à plein temps au sens de l’AVS et si ses cotisations issues de l’activité lucrative (y compris la part de l’employeur) atteignent le double de la cotisation minimale, vous ne devez pas payer de cotisations AVS. Si votre conjoint(e) n’exerce pas durablement une activité lucrative à plein temps, en plus du double de la cotisation minimale, la moitié des cotisations dues en tant que personne sans activité lucrative doit être atteinte (voir point ci-dessus).', 'fr'),
        ('https://www.eak.admin.ch/eak/fr/home/dokumentation/pensionierung/beitragspflicht.html', 'Que signifie être « durablement actif à plein temps » au sens de l’AVS ?', 'Une personne est considérée comme exerçant durablement une activité lucrative à plein temps, si elle exerce son activité lucrative durant la moitié du temps usuellement consacré au travail pendant au moins neuf mois par an.', 'fr'),
        ('https://www.eak.admin.ch/eak/fr/home/dokumentation/pensionierung/beitragspflicht.html', 'Quel est le montant des cotisations AVS/AI/APG que je dois payer en tant que personne sans activité lucrative ?', 'Le montant des cotisations dépend de votre situation financière personnelle. La fortune et le revenu acquis sous forme de rente (p. ex. rentes et pensions de toutes sortes, les indemnités journalières de maladie et d’accident, les pensions alimentaires, les versements réguliers de tiers, etc.) servent de base au calcul des cotisations. Pour les personnes mariées ou vivant en partenariat enregistré, les cotisations sont calculées - indépendamment du régime matrimonial - sur la base de la moitié de la fortune et du revenu acquis sous forme de rente du couple. Il n’est pas possible de payer volontairement des cotisations supplémentaires.', 'fr'),
        ('https://www.eak.admin.ch/eak/fr/home/dokumentation/pensionierung/beitragspflicht.html', 'Comment payer les cotisations en tant que personne sans activité lucrative ?', 'Sur la base de vos propres indications, la caisse de compensation calcule et fixe provisoirement les acomptes de cotisations pour l’année en cours. Les acomptes de cotisations sont facturés (pour trois mois) à la fin de chaque trimestre. Vous pouvez également régler les factures par eBill. L’inscription à eBill se fait dans votre portail financier. Les clients de PostFinance peuvent également payer les factures par le système de recouvrement direct Swiss Direct Debit (prélèvement CH-DD). Les cotisations définitives seront fixées avec une décision une fois que la caisse de compensation aura reçu l’avis d’imposition de l’administration fiscale cantonale des impôts. Sur la base de cette communication fiscale, la différence entre les acomptes de cotisations et les cotisations définitives est calculée, le solde sera facturé ou le trop-payé remboursé.', 'fr'),
    ]

    conn = await get_db_connection()

    try:
        for text in texts:

            # Get the resulting embedding vector from the response
            embedding = get_embedding(text[1])[0]["embedding"]

            # insert FAQ data with embeddings into the 'faq_embeddings' table
            await conn.execute(
                "INSERT INTO faq_embeddings (url, question, answer, language, embedding) VALUES ($1, $2, $3, $4, $5)",
                text[0], text[1], text[2], text[3], str(embedding)
            )

    except Exception as e:
        await conn.close()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if conn:
            await conn.close()

    return {"content": "FAQ data indexed successfully"}

async def update_or_insert_data(
    url: str,
    question: str,
    answer: str,
    language: str,
    id: int = Body(None)
):
    """
    Update an existing data record or insert a new one based on the presence of the question.

    - **url**: The URL associated with the data.
    - **question**: The question.
    - **answer**: The answer.
    - **language**: The language of the data.

    Returns the updated or inserted data record.

    Note: The operation now checks for the presence of a question in the database to decide between insert and update.
    """
    conn = await get_db_connection()
    try:
        # Convert the search question to lowercase to perform a case-insensitive search
        search_query = f"%{question.lower()}%"
        # Check if a record with the same question exists
        existing_row = await conn.fetchrow("SELECT * FROM data WHERE LOWER(question) LIKE $1", search_query)

        if existing_row:
            # Update the existing record
            try:
                await conn.execute(
                    "UPDATE data SET url = $1, question = $2, answer = $3, language = $4 WHERE id = $5",
                    url, question, answer, language, existing_row['id']
                )
                logger.info(f"Update: {url}")
                return {"id": existing_row['id'], "url": url, "question": question, "answer": answer, "language": language}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Update exception: {str(e)}") from e
        else:
            # Insert a new record
            try:
                row = await conn.fetchrow(
                    "INSERT INTO data (url, question, answer, language) VALUES ($1, $2, $3, $4) RETURNING id",
                    url, question, answer, language
                )
                logger.info(f"Insert: {url}")
                return {"id": row['id'], "url": url, "question": question, "answer": answer, "language": language}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Insert exception: {str(e)}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        await conn.close()

async def init_expert():
    """
    Asynchronously retrieves and processes FAQ data from 'https://faq.bsv.admin.ch' to insert into the database.

    The endpoint 'https://faq.bsv.admin.ch/sitemap.xml' is utilized to discover all relevant FAQ URLs. For each URL,
    the method extracts the primary question (denoted by the 'h1' tag) and its corresponding answer (within an 'article' tag).
    Unnecessary boilerplate text will be removed for clarity and conciseness.

    Each extracted FAQ entry is then upserted (inserted or updated if already exists) into the database, with detailed
    logging to track the operation's progress and identify any errors.

    Returns a confirmation message upon successful completion of the process.

    TODO:
    - Consider implementing error handling at a more granular level to retry failed insertions or updates, enhancing the robustness of the data ingestion process.
    - Explore optimization opportunities in text extraction and processing to improve efficiency and reduce runtime, especially for large sitemaps.
    """
    logging.basicConfig(level=logging.INFO)

    sitemap_url = 'https://faq.bsv.admin.ch/sitemap.xml'
    scraper = WebScraper(sitemap_url)

    scraper.logger.info(f"Beginne Datenextraktion für: {sitemap_url}")
    urls = scraper.get_sitemap_urls()

    for url in urls:
        extracted_h1 = scraper.extract_text_from_tag(url, 'h1')
        extracted_article = scraper.extract_text_from_tag(url, 'article')
        language = scraper.detect_language(url)

        # Efficient text processing
        for term in ['Antwort\n', 'Rispondi\n', 'Réponse\n']:
            extracted_article = extracted_article.replace(term, '')

        if extracted_h1 and extracted_article:
            try:
                logger.info(f"extract: {url}")
                await update_or_insert_data(
                    url=url,
                    question=extracted_h1,
                    answer=extracted_article,
                    language=language,
                    id=None
                )
            except Exception as e:
                logger.error(f"Error: {e}")

    logger.info(f"Done! {len(urls)} wurden verarbeitet.")
    return {"message": f"Done! {len(urls)} wurden verarbeitet."}

@app.post("/indexing/index_rag_vectordb/", summary="Insert Embedding data for RAG", response_description="Insert Embedding data for RAG", status_code=200, response_model=ResponseBody)
async def index_rag_vectordb():
    return await init_rag_vectordb()

@app.post("/indexing/index_faq_vectordb/", summary="Insert Embedding data for FAQ autocomplete semantic similarity search", response_description="Insert Embedding data for FAQ semantic similarity search", status_code=200, response_model=ResponseBody)
async def index_faq_vectordb():
    return await init_faq_vectordb()

@app.get("/indexing/crawl_data/", summary="Crawling endpoint", response_description="Welcome Message")
async def crawl_data():
    """
    Dummy endpoint for data crawling.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)

@app.get("/indexing/scrap_data/", summary="Scraping Endpoint", response_description="Welcome Message")
async def scrap_data():
    """
    Dummy endpoint for data scraping.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)

@app.get("/indexing/index_data/", summary="Indexing Endpoint", response_description="Welcome Message")
async def index_data():
    """
    Dummy endpoint for data indexing.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)

@app.get("/indexing/parse_faq_data/", summary="FAQ Parsing Endpoint", response_description="Welcome Message")
async def parse_faq_data():
    """
    Dummy endpoint for FAQ data parsing.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)

@app.get("/indexing/parse_rag_data/", summary="Parsing Endpoint", response_description="Welcome Message")
async def parse_rag_data():
    """
    Dummy endpoint for data parsing (RAG).
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)

@app.get("/indexing/chunk_rag_data/", summary="Chunking Endpoint", response_description="Welcome Message")
async def chunk_rag_data():
    """
    Dummy endpoint for data chunking (RAG).
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)

@app.put("/indexing/index_faq_data/", summary="Insert Data from faq.bsv.admin.ch", response_description="Insert Data from faq.bsv.admin.ch")
async def index_faq_data():
    return await init_expert()

@app.put("/indexing/data/", summary="Update or Insert FAQ Data", response_description="Updated or Inserted Data")
async def index_data(url: str, question: str, answer: str, language: str, id: int = Body(None)):
    return await update_or_insert_data(url, question, answer, language, id)

@app.on_event("startup")
async def startup_event():
    await check_db_connection(retries=10, delay=10)

    if indexing_config["faq"]["auto_index"]:
        # With dev-mode, only index sample FAQ data
        if indexing_config["dev_mode"]:
            try:
                logger.info("Auto-indexing sample FAQ data")
                await init_faq_vectordb()
            except Exception as e:
                logger.error("Dev-mode: Failed to index sample FAQ data: %s", e)
        # If dev-mode is deactivated, scrap and index all bsv.admin.ch FAQ data
        else:
            try:
                logger.info("Auto-indexing bsv.admin.ch FAQ data")
                await init_expert()
            except Exception as e:
                logger.error("Failed to index bsv.admin.ch FAQ data: %s", e)

    if indexing_config["rag"]["auto_index"]:
        # With dev-mode, only index sample data
        if indexing_config["dev_mode"]:
            try:
                logger.info("Auto-indexing sample RAG data")
                await init_rag_vectordb()
            except Exception as e:
                logger.error("Failed to index sample RAG data: %s", e)
        # If dev-mode is deactivated, scrap and index all RAG data (NOTE: Will be implemented soon.)
        else:
            raise NotImplementedError("Feature is not implemented yet.")
