"""
Simplified OCEAN Personality Scoring using existing categorical features.

Instead of relying on free-text descriptions (not available in dataset),
this module uses structured categorical features to infer personality traits.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional
import warnings

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("OpenAI package not installed. Only offline mode available.")


OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def get_text_hash(text: str) -> str:
    """Generate SHA256 hash for caching."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def deterministic_fallback_scores(profile_text: str) -> Dict[str, float]:
    """
    Generate deterministic OCEAN scores from borrower profile.
    Uses hash-based pseudo-random values for offline development.
    """
    text_hash = get_text_hash(profile_text)
    scores = {}

    for i, dim in enumerate(OCEAN_DIMS):
        hex_segment = text_hash[i*8:(i+1)*8]
        value = int(hex_segment, 16) / (16**8 - 1)
        scores[dim] = max(0.25, min(0.75, value))  # Keep in reasonable range

    return scores


def build_borrower_profile(row: Dict) -> str:
    """
    Build a structured borrower profile from categorical features.

    Args:
        row: Dictionary with borrower features (can be a pandas Series)

    Returns:
        Formatted profile string
    """
    profile_parts = []

    # Loan Purpose
    if 'purpose' in row and row['purpose']:
        profile_parts.append(f"Loan Purpose: {row['purpose']}")

    # Loan Term
    if 'term' in row and row['term']:
        profile_parts.append(f"Loan Term: {row['term']}")

    # Credit Grade
    if 'grade' in row and row['grade']:
        grade_text = f"Credit Grade: {row['grade']}"
        if 'sub_grade' in row and row['sub_grade']:
            grade_text += f" ({row['sub_grade']})"
        profile_parts.append(grade_text)

    # Employment Length
    if 'emp_length' in row and row['emp_length']:
        profile_parts.append(f"Employment Length: {row['emp_length']}")

    # Home Ownership
    if 'home_ownership' in row and row['home_ownership']:
        profile_parts.append(f"Home Ownership: {row['home_ownership']}")

    # Verification Status
    if 'verification_status' in row and row['verification_status']:
        profile_parts.append(f"Income Verification: {row['verification_status']}")

    # Application Type
    if 'application_type' in row and row['application_type']:
        profile_parts.append(f"Application Type: {row['application_type']}")

    return " | ".join(profile_parts)


def build_ocean_prompt(profile: str) -> str:
    """
    Build LLM prompt for OCEAN scoring based on borrower profile.

    Args:
        profile: Structured borrower profile string

    Returns:
        Formatted prompt
    """
    prompt = f"""You are a psychologist specializing in personality assessment for credit risk analysis.

Based on the following borrower profile (structured categorical features), estimate their Big Five (OCEAN) personality traits.

Borrower Profile:
{profile}

Task:
Rate each Big Five dimension on a 0.0 to 1.0 scale based on typical behavioral patterns associated with these choices:

- openness: Curiosity, willingness to try new things (0=traditional, 1=innovative)
  - Hints: Loan purpose variety, risk-taking in credit decisions

- conscientiousness: Organization, responsibility, planning (0=spontaneous, 1=disciplined)
  - Hints: Loan term choice, verification status, payment plan behavior

- extraversion: Sociability, assertiveness (0=introverted, 1=extroverted)
  - Hints: Joint vs individual application

- agreeableness: Cooperation, trust (0=competitive, 1=collaborative)
  - Hints: Application type, purpose alignment with social norms

- neuroticism: Emotional stability, anxiety (0=calm, 1=anxious)
  - Hints: Credit grade (stress indicator), home ownership stability

IMPORTANT:
1. Output ONLY valid JSON with exact keys: openness, conscientiousness, extraversion, agreeableness, neuroticism
2. Use decimal values between 0.0 and 1.0
3. Given limited context, stay within 0.3-0.7 range when uncertain
4. No explanations, no additional text

Example output:
{{"openness": 0.45, "conscientiousness": 0.65, "extraversion": 0.50, "agreeableness": 0.60, "neuroticism": 0.40}}"""

    return prompt


def parse_ocean_response(response_text: str) -> Optional[Dict[str, float]]:
    """Parse and validate LLM JSON response."""
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return None

        json_str = response_text[start_idx:end_idx]
        scores = json.loads(json_str)

        if not all(dim in scores for dim in OCEAN_DIMS):
            return None

        validated = {}
        for dim in OCEAN_DIMS:
            value = float(scores[dim])
            if value > 1:
                value = value / 100.0
            validated[dim] = max(0.0, min(1.0, value))

        return validated

    except (json.JSONDecodeError, ValueError, KeyError):
        return None


class SimplifiedOceanScorer:
    """
    OCEAN scorer using existing categorical features (no free text required).
    """

    def __init__(self,
                 cache_dir: str = "artifacts/persona_cache_simple",
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 offline_mode: bool = False):
        """
        Initialize simplified OCEAN scorer.

        Args:
            cache_dir: Cache directory
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini)
            offline_mode: If True, use deterministic fallback
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.offline_mode = offline_mode

        if not offline_mode and OPENAI_AVAILABLE:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                warnings.warn("No API key. Switching to offline mode.")
                self.offline_mode = True
        else:
            self.offline_mode = True

        self.stats = {
            "cache_hits": 0,
            "api_calls": 0,
            "fallback_calls": 0,
            "errors": 0
        }

    def _get_cache_path(self, text_hash: str) -> Path:
        """Get cache file path."""
        subdir = self.cache_dir / text_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{text_hash}.json"

    def _load_from_cache(self, text_hash: str) -> Optional[Dict[str, float]]:
        """Load from cache."""
        cache_path = self._get_cache_path(text_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f).get("scores")
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _save_to_cache(self, text_hash: str, profile: str, scores: Dict[str, float], method: str):
        """Save to cache."""
        cache_path = self._get_cache_path(text_hash)
        data = {
            "text_hash": text_hash,
            "profile_preview": profile[:300],
            "scores": scores,
            "method": method,
            "timestamp": time.time()
        }
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            warnings.warn(f"Cache save failed: {e}")

    def _score_with_api(self, profile: str, retries: int = 3) -> Optional[Dict[str, float]]:
        """Score using OpenAI API."""
        if self.offline_mode:
            return None

        prompt = build_ocean_prompt(profile)

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

            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    warnings.warn(f"API failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.stats["errors"] += 1

        return None

    def score_row(self, row: Dict, force_recompute: bool = False) -> Dict[str, float]:
        """
        Score a single borrower (row from DataFrame).

        Args:
            row: Dictionary or pandas Series with borrower features
            force_recompute: Bypass cache

        Returns:
            OCEAN scores dictionary
        """
        # Build profile from categorical features
        profile = build_borrower_profile(row)
        text_hash = get_text_hash(profile)

        # Check cache
        if not force_recompute:
            cached = self._load_from_cache(text_hash)
            if cached:
                self.stats["cache_hits"] += 1
                return cached

        # Try API
        scores = self._score_with_api(profile)
        if scores:
            self._save_to_cache(text_hash, profile, scores, method="api")
            return scores

        # Fallback
        scores = deterministic_fallback_scores(profile)
        self.stats["fallback_calls"] += 1
        self._save_to_cache(text_hash, profile, scores, method="fallback")

        return scores

    def score_batch(self, df, force_recompute: bool = False, rate_limit_delay: float = 0.5) -> list:
        """
        Score multiple borrowers (DataFrame).

        Args:
            df: pandas DataFrame with borrower features
            force_recompute: Bypass cache
            rate_limit_delay: Delay between API calls

        Returns:
            List of OCEAN score dictionaries
        """
        results = []
        total = len(df)

        print(f"[SimplifiedOceanScorer] Scoring {total} samples...")

        for idx, row in df.iterrows():
            if idx > 0 and idx % 100 == 0:
                print(f"  Progress: {idx}/{total} ({idx/total*100:.1f}%)")

            scores = self.score_row(row, force_recompute)
            results.append(scores)

            if not self.offline_mode and self.stats["api_calls"] > 0:
                time.sleep(rate_limit_delay)

        print(f"[SimplifiedOceanScorer] Done! Stats: {self.stats}")
        return results

    def get_stats(self) -> Dict[str, int]:
        """Return scoring statistics."""
        return self.stats.copy()
