import logging
from typing import List, Union
from datetime import datetime

import asyncpg
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
import httpx

from rag.app.models import ResponseBody, RAGRequest, EmbeddingRequest

# Load env variables
from config.base_config import rag_config
from config.db_config import DB_PARAMS
from config.openai_config import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create an instance of FastAPI
app = FastAPI()

# Function to create a db connection
async def get_db_connection():
    """Establish a database connection."""
    conn = await asyncpg.connect(**DB_PARAMS)
    return conn

# Function to get embeddings for a text
def get_embedding(text: Union[List[str], str]):
    model = rag_config["embedding"]["model"]
    if model == "text-embedding-ada-002":
        response = openai.Embedding.create(
            input=text,
            engine=model,
        )
        return response['data']
    else:
        raise NotImplementedError("Model not supported")

@app.post("/rag/init_rag_vectordb/", summary="Insert Embedding data for RAG", response_description="Insert Embedding data for RAG", status_code=200, response_model=ResponseBody)
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

            # Make POST request to the RAG service to get the question embedding
            async with httpx.AsyncClient() as client:
                response = await client.post("http://rag:8010/rag/embed", json={"text": text[0]})

            # Ensure the request was successful
            response.raise_for_status()

            # Get the resulting embedding vector from the response
            embedding = response.json()["data"][0]["embedding"]

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

@app.post("/rag/init_faq_vectordb/", summary="Insert Embedding data for FAQ autocomplete semantic similarity search", response_description="Insert Embedding data for FAQ semantic similarity search", status_code=200, response_model=ResponseBody)
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

            # Make POST request to the RAG service to get the question embedding
            async with httpx.AsyncClient() as client:
                response = await client.post("http://rag:8010/rag/embed", json={"text": text[1]})

            # Ensure the request was successful
            response.raise_for_status()

            # Get the resulting embedding vector from the response
            embedding = response.json()["data"][0]["embedding"]

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

@app.post("/rag/docs", summary="Retrieve context docs endpoint", response_description="Return context docs from semantic search", status_code=200)
async def docs(request: RAGRequest):

    conn = await get_db_connection()

    try:

        # Make POST request to the RAG service to get the question embedding
        async with httpx.AsyncClient() as client:
            response = await client.post("http://rag:8010/rag/embed", json={"text": request.query})

        # Ensure the request was successful
        response.raise_for_status()

        # Get the resulting embedding vector from the response
        query_embedding = response.json()["data"][0]["embedding"]

        # Only supports retrieval of 1 document at the moment (set in /config/config.yaml). Will implement multi-doc retrieval later
        top_k = rag_config["retrieval"]["top_k"]
        similarity_metric = rag_config["retrieval"]["metric"]
        docs = await conn.fetch(f"""
            SELECT text, url,  1 - (embedding <=> '{query_embedding}') AS {similarity_metric}
            FROM embeddings
            ORDER BY {similarity_metric} desc
            LIMIT $1
        """, top_k)
        docs = [dict(row) for row in docs][0]

    except Exception as e:
        await conn.close()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if conn:
            await conn.close()

    return {"contextDocs": docs["text"], "sourceUrl": docs["url"], "cosineSimilarity": docs["cosine_similarity"]}

@app.post("/rag/embed", summary="Embedding endpoint", response_description="A dictionary with embeddings for the input text")
async def embed(text_input: EmbeddingRequest):
    try:
        embedding = get_embedding(text_input.text)
        return {"data": embedding}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/rerank", summary="Reranking endpoint", response_description="Welcome Message")
async def rerank():
    """
    Dummy endpoint for retrieved docs reranking.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)