"""Download the evaluation corpus: French Wikipedia medical articles.

Plain-text extracts are fetched through the official MediaWiki API and
stored as one markdown file per article in data/corpus/.

Usage:
    python -m src.corpus
"""

import re
import time
from pathlib import Path

import requests

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
API_URL = "https://fr.wikipedia.org/w/api.php"

ARTICLES = [
    "Diabète de type 2",
    "Hypertension artérielle",
    "Asthme",
    "Grippe",
    "Migraine",
    "Anémie",
    "Ostéoporose",
    "Hypothyroïdie",
    "Maladie d'Alzheimer",
    "Maladie de Parkinson",
    "Insuffisance cardiaque",
    "Bronchopneumopathie chronique obstructive",
    "Zona",
    "Mononucléose infectieuse",
    "Dermatite atopique",
    "Appendicite",
    "Sclérose en plaques",
    "Polyarthrite rhumatoïde",
    "Épilepsie",
    "Pneumonie",
]


def slugify(title: str) -> str:
    slug = title.lower()
    slug = re.sub(r"[àâä]", "a", slug)
    slug = re.sub(r"[éèêë]", "e", slug)
    slug = re.sub(r"[îï]", "i", slug)
    slug = re.sub(r"[ôö]", "o", slug)
    slug = re.sub(r"[ùûü]", "u", slug)
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug


def fetch_extract(title: str, max_retries: int = 4) -> str:
    for attempt in range(max_retries):
        response = requests.get(
            API_URL,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "format": "json",
                "titles": title,
                "redirects": 1,
            },
            headers={"User-Agent": "rag-medical-eval (educational project)"},
            timeout=30,
        )
        if response.status_code == 429:  # rate limited: back off and retry
            time.sleep(5 * (attempt + 1))
            continue
        break
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page:
        raise ValueError(f"No extract for article: {title}")
    return page["extract"]


def download_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for title in ARTICLES:
        path = corpus_dir / f"{slugify(title)}.md"
        if path.exists():
            written.append(path)
            continue
        text = fetch_extract(title)
        path.write_text(f"# {title}\n\n{text}", encoding="utf-8")
        written.append(path)
        print(f"  ✓ {title} ({len(text)} chars)")
        time.sleep(1)  # stay well under the API rate limit
    return written


if __name__ == "__main__":
    files = download_corpus()
    print(f"\n{len(files)} articles downloaded to {CORPUS_DIR}")
