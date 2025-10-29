"""
Quick Demo: OCEAN Scoring with Existing Categorical Features

Run this to see the system in action (no notebooks needed).
Uses offline mode (no API calls).
"""
import sys
sys.path.append('.')

from text_features.personality_simple import SimplifiedOceanScorer, build_borrower_profile

def main():
    print("=" * 60)
    print("OCEAN Personality Scoring - Quick Demo")
    print("Using existing categorical features only")
    print("=" * 60)
    print()

    # Sample borrower data (manually created)
    sample_borrowers = [
        {
            "purpose": "debt_consolidation",
            "term": "60 months",
            "grade": "C",
            "sub_grade": "C4",
            "emp_length": "5 years",
            "home_ownership": "RENT",
            "verification_status": "Verified",
            "application_type": "Individual"
        },
        {
            "purpose": "small_business",
            "term": "36 months",
            "grade": "B",
            "sub_grade": "B2",
            "emp_length": "10+ years",
            "home_ownership": "OWN",
            "verification_status": "Verified",
            "application_type": "Individual"
        },
        {
            "purpose": "credit_card",
            "term": "36 months",
            "grade": "A",
            "sub_grade": "A1",
            "emp_length": "3 years",
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified",
            "application_type": "Joint App"
        }
    ]

    # Initialize scorer (offline mode)
    scorer = SimplifiedOceanScorer(offline_mode=True)
    print("[INFO] SimplifiedOceanScorer initialized (offline mode)")
    print()

    # Score each borrower
    for i, borrower in enumerate(sample_borrowers, 1):
        print(f"{'─' * 60}")
        print(f"Borrower {i}:")
        print(f"{'─' * 60}")

        # Build profile
        profile = build_borrower_profile(borrower)
        print(f"Profile: {profile}")
        print()

        # Score
        scores = scorer.score_row(borrower)

        print("OCEAN Scores:")
        for dim, score in scores.items():
            bar = '█' * int(score * 20)  # Visual bar
            print(f"  {dim:20s}: {score:.3f} {bar}")
        print()

    # Show stats
    print(f"{'=' * 60}")
    print("Scoring Statistics:")
    stats = scorer.get_stats()
    for key, value in stats.items():
        print(f"  {key:20s}: {value}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
