CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table 'embeddings' for storing embeddings and associated text for RAG
CREATE TABLE embeddings (
  id SERIAL PRIMARY KEY,
  embedding vector(1536),   -- A vector of dimension 1536
  text text NOT NULL,                 -- Text associated with the vector
  url text NOT NULL,                 -- URL associated with the vector
  created_at timestamptz DEFAULT now(),  -- Timestamp when the record was created
  modified_at timestamptz DEFAULT now()  -- Timestamp when the record was last modified
);

-- Create a table 'faq_embeddings' for storing FAQ question embeddings
CREATE TABLE faq_embeddings (
    id SERIAL PRIMARY KEY,
    url text NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    language VARCHAR(2) DEFAULT 'de',
    embedding vector(1536)
);

-- Erstelle eine Tabelle namens 'data' für die Verwaltung der Informationen
CREATE TABLE data (
    id SERIAL PRIMARY KEY,
    url text NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    language VARCHAR(2) DEFAULT 'de'
);

-- Erstelle eine Trigger-Funktion, um modified_at zu aktualisieren
CREATE OR REPLACE FUNCTION update_modified_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.modified_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Erstelle den Trigger, der die Trigger-Funktion auslöst
CREATE TRIGGER update_data_modified_at
BEFORE UPDATE ON data
FOR EACH ROW
EXECUTE FUNCTION update_modified_at();

-- Einfügen von EAK FAQ Frage-Antwort-Paaren in die Tabelle 'data'
INSERT INTO data (url, question, answer, language) VALUES
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
('https://www.eak.admin.ch/eak/fr/home/dokumentation/pensionierung/beitragspflicht.html', 'Comment payer les cotisations en tant que personne sans activité lucrative ?', 'Sur la base de vos propres indications, la caisse de compensation calcule et fixe provisoirement les acomptes de cotisations pour l’année en cours. Les acomptes de cotisations sont facturés (pour trois mois) à la fin de chaque trimestre. Vous pouvez également régler les factures par eBill. L’inscription à eBill se fait dans votre portail financier. Les clients de PostFinance peuvent également payer les factures par le système de recouvrement direct Swiss Direct Debit (prélèvement CH-DD). Les cotisations définitives seront fixées avec une décision une fois que la caisse de compensation aura reçu l’avis d’imposition de l’administration fiscale cantonale des impôts. Sur la base de cette communication fiscale, la différence entre les acomptes de cotisations et les cotisations définitives est calculée, le solde sera facturé ou le trop-payé remboursé.', 'fr');