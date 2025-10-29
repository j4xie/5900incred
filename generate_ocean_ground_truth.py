"""
Generate OCEAN ground truth for 500 test samples using Router API
"""

import pandas as pd
import json
import os
from ocean_router_api import OceanRouterClient


def main():
    print("=" * 80)
    print("OCEAN Ground Truth Generation - 500 Test Samples")
    print("=" * 80)

    # Load test samples
    test_samples_file = 'test_samples_500.csv'
    print(f"\nLoading test samples from {test_samples_file}...")

    try:
        df_samples = pd.read_csv(test_samples_file)
    except FileNotFoundError:
        print(f"❌ Error: {test_samples_file} not found")
        return

    print(f"✓ Loaded {len(df_samples)} samples")

    # Initialize OCEAN client
    client = OceanRouterClient()

    # Generate OCEAN scores
    ocean_scores = []
    success_count = 0
    failure_count = 0

    print(f"\nGenerating OCEAN scores...")
    print("=" * 80)

    for idx, (_, row) in enumerate(df_samples.iterrows(), 1):
        description = row.get('desc', '')

        if len(description) < 10:
            print(f"\r[{idx:3d}/500] Skipping: description too short", end='', flush=True)
            ocean_scores.append(None)
            failure_count += 1
            continue

        # Generate OCEAN scores
        scores = client.get_ocean_scores(description, max_retries=3)

        if scores:
            ocean_scores.append(scores)
            success_count += 1
            status = "✓"
        else:
            ocean_scores.append(None)
            failure_count += 1
            status = "✗"

        # Display progress every 10 samples
        if idx % 10 == 0 or idx == 500:
            percentage = idx / 500 * 100
            print(f"\r[{idx:3d}/500] ({percentage:5.1f}%) Success: {success_count}, Failed: {failure_count}", end='', flush=True)

        # Add delay to avoid rate limiting
        import time
        time.sleep(0.5)

    print(f"\n\n✓ OCEAN generation complete!")
    print(f"  Success: {success_count}/500")
    print(f"  Failed: {failure_count}/500")
    print(f"  Success rate: {success_count/500*100:.1f}%")

    # Create results dataframe
    print(f"\nCreating results dataframe...")

    results_list = []
    for idx, (_, row) in enumerate(df_samples.iterrows()):
        result = {
            'id': idx,
            'openness': None,
            'conscientiousness': None,
            'extraversion': None,
            'agreeableness': None,
            'neuroticism': None
        }

        if ocean_scores[idx] is not None:
            result.update(ocean_scores[idx])

        results_list.append(result)

    df_results = pd.DataFrame(results_list)

    # Save results
    output_file = 'ocean_ground_truth_500.csv'
    df_results.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")

    # Save summary statistics
    print(f"\nOCEAN Score Statistics:")
    for dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
        valid_scores = df_results[dim].dropna()
        if len(valid_scores) > 0:
            print(f"  {dim:20s}: mean={valid_scores.mean():.3f}, std={valid_scores.std():.3f}, " +
                  f"min={valid_scores.min():.3f}, max={valid_scores.max():.3f}")

    # Save summary report
    summary = {
        'total_samples': len(df_samples),
        'success_count': success_count,
        'failure_count': failure_count,
        'success_rate': f"{success_count/500*100:.1f}%",
        'output_file': output_file,
        'statistics': {}
    }

    for dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
        valid_scores = df_results[dim].dropna()
        if len(valid_scores) > 0:
            summary['statistics'][dim] = {
                'mean': float(valid_scores.mean()),
                'std': float(valid_scores.std()),
                'min': float(valid_scores.min()),
                'max': float(valid_scores.max()),
                'count': int(len(valid_scores))
            }

    with open('ocean_ground_truth_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved: ocean_ground_truth_summary.json")

    print("\n" + "=" * 80)
    print("✅ OCEAN Ground Truth Generation Complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Extract BGE-Large embeddings from 500 samples")
    print(f"2. Train Ridge regression models to map embeddings → OCEAN scores")
    print(f"3. Apply Ridge models to full 34,529 samples for OCEAN feature generation")


if __name__ == "__main__":
    main()
