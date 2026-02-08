# Forum Cloud

Interactive word cloud visualization of article titles from the [Fletcher Forum of World Affairs](https://www.fletcherforum.org/) (1976 -- present).

## What it does

1. **Parses** ~50 years of Fletcher Forum article titles from `allIssues.txt`
2. **Processes** text with spaCy NLP -- tokenization, POS filtering, named-entity merging, contraction/hyphen handling
3. **Generates** per-year word frequency data
4. **Visualizes** the results as an interactive D3.js word cloud (`wordCloudClaude.html`) with a year slider

## Project structure

```
allIssues.txt          # Raw article titles by year, scraped from the Fletcher Forum site
Untitled-1.ipynb       # Jupyter notebook -- main NLP pipeline
export_data.py         # Exports word frequencies to JS for the HTML visualization
word_cloud_data.js     # Generated frequency data (consumed by the HTML files)
wordCloudClaude.html   # Interactive D3 word cloud with year slider
fletcher-wordcloud.html# Alternate/in-progress visualization
main.py                # Entry point (placeholder)
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

Then open `Untitled-1.ipynb` in VS Code or Jupyter and run all cells.

## Viewing the word cloud

Open `wordCloudClaude.html` in a browser. Use the year slider to browse topics across decades.

## Dependencies

- **spaCy** -- NLP tokenization and entity recognition
- **NLTK** -- stopword lists
- **wordcloud / matplotlib** -- static word cloud generation (notebook)
- **D3.js + d3-cloud** -- interactive browser visualization
