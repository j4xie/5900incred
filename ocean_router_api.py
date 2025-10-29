"""
OCEAN Personality Feature Generation using HuggingFace Router API (Fireworks Backend)
This is the working approach using Router API instead of Inference API
"""

import requests
import json
import os
import time
from typing import Dict, Optional, List


class OceanRouterClient:
    """
    Client for generating OCEAN personality scores using HF Router API
    Uses Llama-3.1-8B-Instruct via Fireworks AI backend
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.model = "meta-llama/Llama-3.1-8B-Instruct:fireworks-ai"
        self.hf_token = hf_token or self._load_token()

    def _load_token(self) -> str:
        """Load HF token from .env file"""
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        if key == 'HF_TOKEN':
                            return value
        except:
            pass
        return os.getenv('HF_TOKEN', '')

    def get_ocean_scores(
        self,
        description: str,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> Optional[Dict[str, float]]:
        """
        Generate OCEAN scores from a loan description

        Args:
            description: Loan application description text
            max_retries: Number of retry attempts
            retry_delay: Seconds to wait between retries

        Returns:
            Dictionary with OCEAN scores or None if failed
        """

        # Build prompt for OCEAN scoring
        prompt = self._build_prompt(description)

        # API configuration
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "model": self.model,
            "stream": False,
            "max_tokens": 200,
            "temperature": 0.7
        }

        # Retry logic
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    # Extract text from response
                    if 'choices' in result and len(result['choices']) > 0:
                        text_output = result['choices'][0].get('message', {}).get('content', '')

                        # Parse OCEAN scores from JSON
                        ocean_scores = self._parse_ocean_json(text_output)

                        if ocean_scores and len(ocean_scores) == 5:
                            return ocean_scores

                    return None

                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        return None

                else:
                    return None

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return None

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return None

        return None

    def _build_prompt(self, description: str) -> str:
        """Build the prompt for OCEAN scoring"""
        return f"""Please analyze the following loan application description and provide OCEAN personality trait scores for the applicant.
Each trait score should be a number between 0 and 1.

Loan description:
{description}

Please return the result in the following JSON format only, no other content:
{{
  "openness": 0.5,
  "conscientiousness": 0.5,
  "extraversion": 0.5,
  "agreeableness": 0.5,
  "neuroticism": 0.5
}}

Explanation:
- openness: 0=conservative, 1=creative and open to new experiences
- conscientiousness: 0=careless, 1=very careful and responsible
- extraversion: 0=introverted, 1=extroverted and sociable
- agreeableness: 0=cold, 1=friendly and cooperative
- neuroticism: 0=emotionally stable, 1=anxious and sensitive
"""

    def _parse_ocean_json(self, text: str) -> Optional[Dict[str, float]]:
        """Extract and parse OCEAN JSON from response text"""
        try:
            # Find JSON in response
            json_start = text.find('{')
            if json_start == -1:
                return None

            json_end = text.rfind('}') + 1
            if json_end <= json_start:
                return None

            json_string = text[json_start:json_end]

            # Parse JSON
            data = json.loads(json_string)

            # Extract and validate OCEAN scores
            ocean_dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
            scores = {}

            for dim in ocean_dims:
                if dim not in data:
                    return None
                try:
                    score = float(data[dim])
                    # Clip to valid range
                    score = max(0.0, min(1.0, score))
                    scores[dim] = score
                except (ValueError, TypeError):
                    return None

            return scores if len(scores) == 5 else None

        except:
            return None


def test_ocean_scoring():
    """Test the OCEAN scoring functionality"""

    print("=" * 80)
    print("Testing OCEAN Router API Implementation")
    print("=" * 80)

    # Initialize client
    client = OceanRouterClient()

    # Test with a sample description
    test_description = """
    I am applying for a loan to start my own business. I have been working in the
    technology sector for 5 years and have saved significant capital. I am very
    organized and detail-oriented, always meeting deadlines. I enjoy collaborating
    with others and have led several successful projects. I am naturally curious
    and constantly learning new skills to stay ahead in the industry. I rarely
    get anxious and maintain a positive outlook even during challenges.
    """

    print(f"\nTest Description:")
    print(f"{test_description[:200]}...")

    print(f"\nGenerating OCEAN scores...")
    ocean_scores = client.get_ocean_scores(test_description)

    if ocean_scores:
        print(f"\n✅ SUCCESS! OCEAN Scores Generated:")
        for dim, score in ocean_scores.items():
            print(f"  {dim:20s}: {score:.3f}")
    else:
        print(f"\n❌ Failed to generate OCEAN scores")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_ocean_scoring()
