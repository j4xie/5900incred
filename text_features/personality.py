"""
Big Five (OCEAN) Personality Scoring Module

Features:
- LLM-based scoring with OpenAI API (adapted for title + job context)
- SHA256-based caching to avoid redundant API calls
- Offline deterministic fallback for zero-cost development
- Strict JSON output validation (0-1 scale)
- Rate limiting and exponential backoff

Adapted for LendingClub dataset where borrower descriptions are unavailable.
Uses loan title + employment title as weak personality signals.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

# OpenAI import (optional - graceful degradation if unavailable)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("OpenAI package not installed. Only offline mode available.")


# OCEAN dimension names
OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def get_text_hash(text: str) -> str:
    """
    Generate SHA256 hash of text for caching.

    Args:
        text: Input text

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def deterministic_fallback_scores(text: str) -> Dict[str, float]:
    """
    Generate deterministic fallback scores using hash-based pseudo-random values.

    Used for offline development and testing. Ensures reproducibility but
    has no psychological validity.

    Args:
        text: Input text

    Returns:
        Dictionary with OCEAN scores in [0, 1]
    """
    text_hash = get_text_hash(text)
    scores = {}

    # Use different hash segments for each dimension
    for i, dim in enumerate(OCEAN_DIMS):
        # Take 8 characters from different positions in hash
        hex_segment = text_hash[i*8:(i+1)*8]
        # Convert to int and normalize to [0, 1]
        value = int(hex_segment, 16) / (16**8 - 1)
        # Clip to reasonable range (avoid extreme 0 or 1)
        scores[dim] = max(0.2, min(0.8, value))

    return scores


def build_ocean_prompt(title: str, emp_title: Optional[str], max_chars: int = 800) -> str:
    """
    Build prompt for LLM to score Big Five personality traits.

    Adapted for limited text context (loan title + employment title).

    Args:
        title: Loan title/purpose
        emp_title: Employment title/company
        max_chars: Maximum character limit for combined text

    Returns:
        Formatted prompt string
    """
    # Combine available text
    text_parts = [f"Loan Purpose: {title}"]
    if emp_title and emp_title.strip():
        text_parts.append(f"Employment: {emp_title}")

    combined_text = " | ".join(text_parts)[:max_chars]

    prompt = f"""You are a psychologist specializing in personality assessment.

Based on the following LIMITED information about a loan applicant, estimate their Big Five (OCEAN) personality traits. This is a weak signal inference based only on loan purpose and occupation.

Input:
{combined_text}

Task:
Rate each Big Five dimension on a 0.0 to 1.0 scale:
- openness: Curiosity, creativity, openness to new experiences (0=traditional, 1=innovative)
- conscientiousness: Organization, responsibility, dependability (0=spontaneous, 1=disciplined)
- extraversion: Sociability, assertiveness, energy (0=introverted, 1=extroverted)
- agreeableness: Compassion, cooperation, trust (0=competitive, 1=collaborative)
- neuroticism: Emotional instability, anxiety, moodiness (0=calm, 1=anxious)

IMPORTANT:
1. Output ONLY valid JSON with these exact keys
2. Use decimal values between 0.0 and 1.0
3. Given limited context, avoid extreme values (stay within 0.3-0.7 range when uncertain)
4. No explanations, no additional text

Example output:
{{"openness": 0.55, "conscientiousness": 0.65, "extraversion": 0.45, "agreeableness": 0.60, "neuroticism": 0.40}}"""

    return prompt


def parse_ocean_response(response_text: str) -> Optional[Dict[str, float]]:
    """
    Parse and validate LLM response.

    Args:
        response_text: Raw response from LLM

    Returns:
        Dictionary with validated OCEAN scores, or None if invalid
    """
    try:
        # Try to extract JSON (handle cases where LLM adds extra text)
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return None

        json_str = response_text[start_idx:end_idx]
        scores = json.loads(json_str)

        # Validate all dimensions present
        if not all(dim in scores for dim in OCEAN_DIMS):
            return None

        # Validate and clip values to [0, 1]
        validated = {}
        for dim in OCEAN_DIMS:
            value = float(scores[dim])
            # If value is > 1, assume it's on 0-100 scale
            if value > 1:
                value = value / 100.0
            validated[dim] = max(0.0, min(1.0, value))

        return validated

    except (json.JSONDecodeError, ValueError, KeyError):
        return None


class OceanScorer:
    """
    Main class for scoring Big Five personality traits with caching.
    """

    def __init__(self,
                 cache_dir: str = "artifacts/persona_cache",
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 offline_mode: bool = False,
                 max_chars: int = 800):
        """
        Initialize OCEAN scorer.

        Args:
            cache_dir: Directory to store cached scores
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name (default: gpt-4o-mini for cost efficiency)
            offline_mode: If True, use deterministic fallback only
            max_chars: Maximum characters for text truncation
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.max_chars = max_chars
        self.offline_mode = offline_mode

        # Initialize OpenAI client
        if not offline_mode and OPENAI_AVAILABLE:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                warnings.warn("No API key provided. Switching to offline mode.")
                self.offline_mode = True
        else:
            self.offline_mode = True

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "api_calls": 0,
            "fallback_calls": 0,
            "errors": 0
        }

    def _get_cache_path(self, text_hash: str) -> Path:
        """Get cache file path for a text hash."""
        # Use first 2 chars of hash for subdirectory (avoid too many files in one dir)
        subdir = self.cache_dir / text_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{text_hash}.json"

    def _load_from_cache(self, text_hash: str) -> Optional[Dict[str, float]]:
        """Load scores from cache."""
        cache_path = self._get_cache_path(text_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data.get("scores")
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _save_to_cache(self, text_hash: str, text: str, scores: Dict[str, float], method: str):
        """Save scores to cache."""
        cache_path = self._get_cache_path(text_hash)
        data = {
            "text_hash": text_hash,
            "text_preview": text[:200],  # Save preview for debugging
            "scores": scores,
            "method": method,  # "api" or "fallback"
            "timestamp": time.time()
        }
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            warnings.warn(f"Failed to save cache: {e}")

    def _score_with_api(self, text: str, retries: int = 3) -> Optional[Dict[str, float]]:
        """
        Score text using OpenAI API with exponential backoff.

        Args:
            text: Input text (already combined title + emp_title)
            retries: Number of retry attempts

        Returns:
            OCEAN scores or None if failed
        """
        if self.offline_mode:
            return None

        # Extract title and emp_title from combined text
        parts = text.split(" | ")
        title = parts[0].replace("Loan Purpose: ", "")
        emp_title = parts[1].replace("Employment: ", "") if len(parts) > 1 else None

        prompt = build_ocean_prompt(title, emp_title, self.max_chars)

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful psychologist. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=150
                )

                response_text = response.choices[0].message.content
                scores = parse_ocean_response(response_text)

                if scores:
                    self.stats["api_calls"] += 1
                    return scores
                else:
                    warnings.warn(f"Failed to parse response: {response_text}")

            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    warnings.warn(f"API call failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    warnings.warn(f"API call failed after {retries} attempts: {e}")
                    self.stats["errors"] += 1

        return None

    def score(self, title: str, emp_title: Optional[str] = None,
              force_recompute: bool = False) -> Dict[str, float]:
        """
        Score a single text sample.

        Args:
            title: Loan title/purpose
            emp_title: Employment title (optional)
            force_recompute: If True, bypass cache and recompute

        Returns:
            Dictionary with OCEAN scores (always returns valid scores)
        """
        # Combine text for hashing
        text_parts = [title]
        if emp_title and emp_title.strip():
            text_parts.append(emp_title)
        combined_text = " | ".join(text_parts)

        text_hash = get_text_hash(combined_text)

        # Check cache first
        if not force_recompute:
            cached_scores = self._load_from_cache(text_hash)
            if cached_scores:
                self.stats["cache_hits"] += 1
                return cached_scores

        # Try API scoring
        scores = self._score_with_api(combined_text)

        if scores:
            self._save_to_cache(text_hash, combined_text, scores, method="api")
            return scores

        # Fallback to deterministic scores
        scores = deterministic_fallback_scores(combined_text)
        self.stats["fallback_calls"] += 1
        self._save_to_cache(text_hash, combined_text, scores, method="fallback")

        return scores

    def score_batch(self, titles: list, emp_titles: Optional[list] = None,
                   force_recompute: bool = False,
                   rate_limit_delay: float = 0.5) -> list:
        """
        Score multiple samples with rate limiting.

        Args:
            titles: List of loan titles
            emp_titles: List of employment titles (optional)
            force_recompute: If True, bypass cache
            rate_limit_delay: Seconds to wait between API calls

        Returns:
            List of OCEAN score dictionaries
        """
        if emp_titles is None:
            emp_titles = [None] * len(titles)

        assert len(titles) == len(emp_titles), "titles and emp_titles must have same length"

        results = []
        total = len(titles)

        print(f"[OceanScorer] Scoring {total} samples...")

        for i, (title, emp_title) in enumerate(zip(titles, emp_titles)):
            if i > 0 and i % 100 == 0:
                print(f"  Progress: {i}/{total} ({i/total*100:.1f}%)")

            scores = self.score(title, emp_title, force_recompute)
            results.append(scores)

            # Rate limiting (only if API call was made)
            if not self.offline_mode and self.stats["api_calls"] > 0:
                time.sleep(rate_limit_delay)

        print(f"[OceanScorer] Completed! Stats: {self.stats}")
        return results

    def get_stats(self) -> Dict[str, int]:
        """Return scoring statistics."""
        return self.stats.copy()
