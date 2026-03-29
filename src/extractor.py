"""Signal extraction from Berkshire letters using Claude."""
from __future__ import annotations

import json
from pathlib import Path

import anthropic

from .config import ANTHROPIC_API_KEY, MAX_TOKENS, MODEL, LETTERS_DIR, SIGNALS_DIR
from .prompts import SYSTEM_PROMPT, EXTRACTION_PROMPT
from .schema import LetterExtraction, flatten_for_timeseries


def load_letter(year: int) -> str:
    """Load letter text for a given year."""
    letter_path = LETTERS_DIR / f"{year}.txt"
    if not letter_path.exists():
        raise FileNotFoundError(f"No letter found for {year} at {letter_path}")
    return letter_path.read_text()


def extract_signals(letter_text: str) -> LetterExtraction:
    """Extract structured signals from letter text using Claude.

    Returns a validated LetterExtraction model.
    Raises ValidationError if LLM output doesn't match schema.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": EXTRACTION_PROMPT.format(letter_text=letter_text)}
        ],
    )

    # Parse response
    content = response.content[0].text

    # Handle potential markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])

    data = json.loads(content)

    # Validate against Pydantic model
    return LetterExtraction.model_validate(data)


def extract_and_save(year: int) -> LetterExtraction:
    """Extract signals from a letter and save to JSON."""
    letter_text = load_letter(year)
    analysis = extract_signals(letter_text)

    output_path = SIGNALS_DIR / f"{year}.json"
    output_path.write_text(analysis.model_dump_json(indent=2))

    return analysis


def load_analysis(year: int) -> LetterExtraction:
    """Load previously extracted analysis."""
    signal_path = SIGNALS_DIR / f"{year}.json"
    if not signal_path.exists():
        raise FileNotFoundError(f"No signals found for {year}")
    return LetterExtraction.model_validate_json(signal_path.read_text())


def get_available_letters() -> list[int]:
    """Get list of years with available letters."""
    return sorted([int(p.stem) for p in LETTERS_DIR.glob("*.txt")])


def get_available_signals() -> list[int]:
    """Get list of years with extracted signals."""
    return sorted([int(p.stem) for p in SIGNALS_DIR.glob("*.json")])


def load_all_flat() -> list[dict]:
    """Load all extractions as flat dicts for DataFrame creation."""
    return [
        flatten_for_timeseries(load_analysis(year))
        for year in get_available_signals()
    ]
