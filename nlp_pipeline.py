import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
import spacy
from spacy.language import Language
from spacy.util import filter_spans

# ============================================================
# 1. Read and parse the raw issue data
# ============================================================

file = open("allIssues.txt", "r")
raw = file.read()
years = raw.split("\n\n")

yearsDict = {}
filterPhrases = [
    "\tTable of Contents", "•\tIssue", "A Conversation with",
    "Book Reviews", "Books Reviews", "Books Received", "Books Review",
    "Editor's Note", "Editors' Note", "From the Editor's",
    "Recent Publications", "Interview with", "Author Index", "Forum",
    "Book Review", "Book Received", "Books in Brief",
    "Errata", "Erratum", "Reviews",
]

for i, item in enumerate(years):
    if (i % 2) == 0:
        strippedItem = item.split("(")[1].split(")")[0]
        yearsDict[strippedItem] = {}
        last_key = strippedItem
    else:
        issues = item.split("-\u2028Download PDF\u2028\u2028\n\t")[1:]
        yearsDict[last_key] = [
            item.split("\xa0")[0].split("\t")[1].split("\n")[0]
            for item in issues
            if len([phrase for phrase in filterPhrases if phrase in item]) == 0
        ]

# ============================================================
# 2. Set up spaCy pipeline with custom components
# ============================================================

@Language.component("merge_contractions")
def merge_contractions(doc):
    spans = []
    for token in doc:
        if token.pos_ == "AUX" and token.text.startswith("'") and token.i > 0:
            spans.append((token.i - 1, token.i + 1))
        elif token.text == "n't" and token.i > 0:
            spans.append((token.i - 1, token.i + 1))

    spans.sort(key=lambda s: (s[0], -s[1]))
    filtered = []
    for span in spans:
        if not filtered or span[0] >= filtered[-1][1]:
            filtered.append(span)

    with doc.retokenize() as retokenizer:
        for start, end in filtered:
            retokenizer.merge(doc[start:end])

    return doc

@Language.component("merge_hyphens")
def merge_hyphens(doc):
    spans = []
    i = 0
    while i < len(doc):
        if doc[i].text in ("-", "\u2013", "\u2014"):
            has_space_before = i > 0 and doc[i - 1].idx + len(doc[i - 1].text) < doc[i].idx
            has_space_after = i < len(doc) - 1 and doc[i].idx + len(doc[i].text) < doc[i + 1].idx

            if not has_space_before and not has_space_after and i > 0 and i < len(doc) - 1:
                start = i - 1
                end = i + 1

                while end + 1 < len(doc) and doc[end + 1].text in ("-", "\u2013", "\u2014"):
                    no_space_to_hyphen = doc[end].idx + len(doc[end].text) == doc[end + 1].idx
                    if no_space_to_hyphen and end + 2 < len(doc):
                        no_space_after_hyphen = doc[end + 1].idx + len(doc[end + 1].text) == doc[end + 2].idx
                        if no_space_after_hyphen:
                            end = end + 2
                        else:
                            break
                    else:
                        break

                spans.append(doc[start:end + 1])
                i = end + 1
                continue
        i += 1

    filtered = filter_spans(spans)

    with doc.retokenize() as retokenizer:
        for span in filtered:
            retokenizer.merge(span)

    return doc

@Language.component("merge_named_entities")
def merge_named_entities(doc):
    spans = []

    CUSTOM_TERMS = {
        ("United", "States"),
        ("United", "Nations"),
        ("United", "Kingdom"),
        ("European", "Union"),
        ("World", "Trade", "Organization"),
        ("World", "Bank"),
        ("World", "Order"),
        ("International", "Monetary", "Fund"),
        ("International", "Court"),
        ("Security", "Council"),
        ("General", "Assembly"),
        ("Human", "Rights"),
        ("Cold", "War"),
        ("Civil", "War"),
        ("Gulf", "War"),
        ("Middle", "East"),
        ("South", "Korea"),
        ("North", "Korea"),
        ("South", "Africa"),
        ("Saudi", "Arabia"),
        ("Hong", "Kong"),
        ("Sri", "Lanka"),
        ("Costa", "Rica"),
        ("Puerto", "Rico"),
        ("Soviet", "Union"),
        ("Third", "World"),
        ("El", "Salvador"),
        ("Western", "Europe"),
        ("Eastern", "Europe"),
        ("Latin", "America"),
        ("Berlin", "Wall"),
        ("Persian", "Gulf"),
        ("SALT", "II"),
        ("West", "German"),
        ("Reagan", "Era"),
        ("Weinberger", "Doctrine"),
        ("21st", "Century"),
        ("New", "York"),
        ("People", "'s", "Republic"),
        ("African", "Elephant"),
        ("Basel", "Convention"),
        ("Chemical", "Weapons"),
        ("Southeast", "Asia"),
        ("Northern", "Ireland"),
        ("North", "Sea"),
        ("Western", "Front"),
        ("Panama", "Canal"),
        ("West", "Germany"),
        ("East", "Germany"),
        ("Twentieth", "Anniversary"),
        ("South", "China", "Sea"),
        ("Sub", "-", "Saharan", "Africa"),
        ("Twenty", "-", "First", "Century"),
        ("Autonomous", "Weapon", "Systems"),
        ("International", "Law"),
        ("International", "Relations"),
        ("Natural", "Resources"),
        ("Hazardous", "Waste"),
        ("Organized", "Crime"),
        ("Marshall", "Plan"),
        ("Financial", "Crisis"),
        ("Single", "State"),
        ("Multiparty", "System"),
        ("Case", "Study"),
        ("Land", "Rights"),
        ("War", "Crimes"),
    }

    for term in CUSTOM_TERMS:
        term_len = len(term)
        for i in range(len(doc) - term_len + 1):
            window = tuple(doc[i + j].text for j in range(term_len))
            if window == term:
                spans.append(doc[i:i + term_len])

    filtered = filter_spans(spans)

    with doc.retokenize() as retokenizer:
        for span in filtered:
            retokenizer.merge(span)

    return doc

@Language.component("merge_modifier_nouns")
def merge_modifier_nouns(doc):
    """Merge any modifier with 'Relations'."""
    spans = []
    for i, token in enumerate(doc):
        if token.text == "Relations" and i > 0:
            spans.append(doc[i - 1:i + 1])
    filtered = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in filtered:
            retokenizer.merge(span)
    return doc

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_named_entities", after="ner")
nlp.add_pipe("merge_hyphens", after="merge_named_entities")
nlp.add_pipe("merge_contractions", after="merge_hyphens")
nlp.add_pipe("merge_modifier_nouns", after="merge_contractions")

# ============================================================
# 3. Build word_freq_by_year
# ============================================================

SKIP_POS = {"ADP", "DET", "PUNCT", "CCONJ", "PART", "SCONJ", "SYM", "SPACE", "NUM"}
STOP_WORDS = set(stopwords.words('english'))

KNOWN_ACRONYMS = {"nato", "un", "eu", "eec", "asean", "apec", "opec", "wto", "imf", "salt ii", "dmk-aiadmk", "dod", "r2p"}

TYPO_CORRECTIONS = {
    "afer": "after",
    "amazona": "amazonia",
    "challanges": "challenges",
    "cimes": "crimes",
    "commmissions": "commissions",
    "commmunity": "community",
    "decolonization(s": "decolonization",
    "evironmental": "environmental",
    "expanson": "expansion",
    "faustain": "faustian",
    "globaliziation": "globalization",
    "humnitarian": "humanitarian",
    "incongreuences": "incongruences",
    "internatioanl": "international",
    "mis)understanding": "misunderstanding",
    "multinatioanl": "multinational",
    "petrobus": "petrobras",
    "rappochement": "rapprochement",
    "refletions": "reflections",
    "salvadaor": "el salvador",
    "tranforming": "transforming",
    "transational": "transnational",
    "utlilization": "utilization",
    "westen": "western",
    "ethiopa-eritrea": "ethiopia-eritrea",
    "sef-defense": "self-defense",
}

CUSTOM_FILTER_WORDS = {
    "rudolf", "bahro", "johannes", "preisinger", "mr.", "51(1/2", "un70",
    "frei", "miguel", "madrid", "reagan", "kennan", "carter", "truman",
    "milosevic", "blair", "bush", "fujimori", "primakov", "alexander",
    "maliki", "awlaki", "obama", "trump", "greta", "tshilombo",
    "fletcher", "can't", "what's",
    "uti", "possidetis", "juris", "jus", "bello",
}

def normalize_token(text):
    """Convert dot-separated acronyms (U.S. -> US), preserve all-caps acronyms, lowercase everything else."""
    if re.match(r'^([A-Za-z]\.)+$', text):
        return text.replace('.', '').upper()
    if re.match(r'^[A-Z]{2,}$', text):
        return text
    lower = text.lower()
    if lower in KNOWN_ACRONYMS:
        return lower.upper()
    return lower

word_freq_by_year = {}

for year, titles in yearsDict.items():
    docs = [nlp(title) for title in titles]

    # First pass: count person name frequencies
    person_counts = Counter()
    for doc in docs:
        for token in doc:
            if token.ent_type_ == "PERSON":
                word = normalize_token(token.text)
                word = TYPO_CORRECTIONS.get(word, word)
                person_counts[word] += 1

    frequent_persons = {name for name, count in person_counts.items() if count > 3}

    # Second pass: build word list
    all_words = []
    for doc in docs:
        for token in doc:
            word = normalize_token(token.text)
            word = TYPO_CORRECTIONS.get(word, word)
            # Skip infrequent person names (but keep multi-word CUSTOM_TERMS tokens)
            if token.ent_type_ == "PERSON" and " " not in token.text and word not in frequent_persons:
                continue
            if token.pos_ in SKIP_POS:
                continue
            if word in CUSTOM_FILTER_WORDS:
                continue
            if word.lower() in STOP_WORDS:
                continue
            if len(word) < 3 and not word.isupper():
                continue
            all_words.append(word)
    word_freq_by_year[year] = dict(Counter(all_words))

# ============================================================
# 4. Export
# ============================================================

exec(open("export_data.py").read())
export_for_html(word_freq_by_year)
