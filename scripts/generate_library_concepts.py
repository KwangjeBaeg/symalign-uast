"""
Generate Library Concepts using OpenAI API

This script generates semantic concepts for all frequent libraries using GPT-4o-mini.

Strategy:
1. Load list of frequent libraries (95 total)
2. For each library, generate concept using OpenAI API
3. Use batch API for cost efficiency
4. Save results to libraries_concepts.json

Model Choice: gpt-4o-mini
- Reasoning: Good balance of quality and cost
- Cost: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- Expected total: ~$2-3 for all 95 libraries
- Benchmark: Better than GPT-3.5, cheaper than GPT-4
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
ROOT_DIR = Path(__file__).parent.parent
LIBRARY_USAGE_PATH = ROOT_DIR / "Data profiler" / "library_concept_usage_count.json"
LIBRARY_CONCEPTS_PATH = ROOT_DIR / "extras" / "LLM interaction pipleline" / "libraries_concepts.json"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "library_concepts_full.json"

# Ensure output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_frequent_libraries(min_count: int = 50) -> List[str]:
    """Load list of frequent libraries."""
    with open(LIBRARY_USAGE_PATH) as f:
        data = json.load(f)
        library_usage = data["library_usage_count"]

    # Filter by minimum count
    frequent_libs = [
        lib for lib, count in library_usage.items()
        if count >= min_count
    ]

    print(f"Found {len(frequent_libs)} libraries with >={min_count} uses")
    return sorted(frequent_libs)


def load_existing_concepts() -> Dict[str, Dict]:
    """Load existing concepts to avoid re-generating."""
    if not LIBRARY_CONCEPTS_PATH.exists():
        return {}

    with open(LIBRARY_CONCEPTS_PATH) as f:
        concepts_list = json.load(f)

    # Convert list to dict
    concepts_dict = {}
    for item in concepts_list:
        lib_name = item.get("Library Name")
        if lib_name:
            concepts_dict[lib_name] = item

    print(f"Loaded {len(concepts_dict)} existing concepts")
    return concepts_dict


def create_concept_prompt(library_name: str) -> str:
    """
    Create prompt for GPT to generate library concept.

    Optimized for:
    - Concise output (reduce token cost)
    - Structured JSON response
    - High-quality semantic information
    """
    return f"""You are an expert software engineer analyzing Python libraries.

For the library "{library_name}", provide the following information in JSON format:

{{
  "Library Name": "{library_name}",
  "Functionality": "<concise 2-3 word description, e.g., 'Matrix Operations', 'Web Framework'>",
  "Domain": "<primary domain, choose from: Scientific Computing, Web Development, Data Processing, Machine Learning, Systems Programming, Testing, Networking, Security, GUI, Database, DevOps, Cloud Computing, Quantum Computing, or 'Unknown'>",
  "Dependency Type": "<'Standard' if part of Python stdlib, 'Third Party' otherwise>",
  "Platform": "<'Cross-platform' or 'Platform-specific'>",
  "Framework": "<name of framework if it's part of one, e.g., 'Django', 'Flask', 'PyTorch', else 'None'>",
  "Description": "<one sentence describing what developers use this library for>"
}}

Rules:
1. If you're unsure about any field, use "Unknown" or "None"
2. Be concise but accurate
3. Focus on the PRIMARY use case
4. Output ONLY the JSON, no explanations

JSON:"""


def generate_concept_batch(libraries: List[str],
                           existing_concepts: Dict[str, Dict],
                           batch_size: int = 10,
                           delay: float = 1.0) -> List[Dict]:
    """
    Generate concepts for libraries using OpenAI API.

    Args:
        libraries: List of library names
        existing_concepts: Already generated concepts (to skip)
        batch_size: Number of concurrent requests
        delay: Delay between batches (to avoid rate limits)

    Returns:
        List of concept dictionaries
    """
    all_concepts = []

    # Filter out libraries that already have concepts
    to_generate = [lib for lib in libraries if lib not in existing_concepts]

    print(f"\nGenerating concepts for {len(to_generate)} libraries...")
    print(f"Skipping {len(libraries) - len(to_generate)} libraries (already have concepts)")

    if not to_generate:
        print("All libraries already have concepts!")
        return list(existing_concepts.values())

    # Process in batches
    for i in tqdm(range(0, len(to_generate), batch_size), desc="Batch processing"):
        batch = to_generate[i:i+batch_size]

        for library in batch:
            try:
                # Create prompt
                prompt = create_concept_prompt(library)

                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Cost-effective model
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that provides structured information about Python libraries in JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,  # Lower for more deterministic output
                    max_tokens=300,   # Sufficient for our JSON
                    response_format={"type": "json_object"}  # Ensure JSON output
                )

                # Parse response
                content = response.choices[0].message.content
                concept = json.loads(content)

                all_concepts.append(concept)

                # Small delay to avoid rate limits
                time.sleep(0.5)

            except Exception as e:
                print(f"\nError generating concept for {library}: {e}")
                # Add placeholder
                all_concepts.append({
                    "Library Name": library,
                    "Functionality": "Unknown",
                    "Domain": "Unknown",
                    "Dependency Type": "Unknown",
                    "Platform": "Unknown",
                    "Framework": "None",
                    "Description": "Error generating concept"
                })

        # Delay between batches
        if i + batch_size < len(to_generate):
            time.sleep(delay)

    # Merge with existing concepts
    all_concepts_dict = {c["Library Name"]: c for c in all_concepts}
    existing_concepts.update(all_concepts_dict)

    return list(existing_concepts.values())


def save_concepts(concepts: List[Dict], output_path: Path):
    """Save concepts to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(concepts, f, indent=2)

    print(f"\nSaved {len(concepts)} concepts to {output_path}")


def print_statistics(concepts: List[Dict]):
    """Print statistics about generated concepts."""
    from collections import Counter

    # Count by domain
    domains = [c.get("Domain", "Unknown") for c in concepts]
    domain_counts = Counter(domains)

    # Count by dependency type
    dep_types = [c.get("Dependency Type", "Unknown") for c in concepts]
    dep_counts = Counter(dep_types)

    print("\n" + "="*60)
    print("CONCEPT GENERATION STATISTICS")
    print("="*60)
    print(f"\nTotal libraries: {len(concepts)}")

    print("\nTop domains:")
    for domain, count in domain_counts.most_common(10):
        print(f"  {domain:30s}: {count:3d}")

    print("\nDependency types:")
    for dep_type, count in dep_counts.most_common():
        print(f"  {dep_type:30s}: {count:3d}")

    # Show some examples
    print("\nExample concepts:")
    for i, concept in enumerate(concepts[:5]):
        print(f"\n{i+1}. {concept['Library Name']}")
        print(f"   Domain: {concept.get('Domain', 'Unknown')}")
        print(f"   Functionality: {concept.get('Functionality', 'Unknown')}")
        print(f"   Description: {concept.get('Description', 'N/A')}")


def estimate_cost(num_libraries: int) -> float:
    """
    Estimate API cost for generating concepts.

    gpt-4o-mini pricing (as of Feb 2025):
    - Input: $0.150 per 1M tokens
    - Output: $0.600 per 1M tokens

    Estimated tokens per library:
    - Input (prompt): ~200 tokens
    - Output (JSON): ~150 tokens
    """
    input_tokens = num_libraries * 200
    output_tokens = num_libraries * 150

    input_cost = (input_tokens / 1_000_000) * 0.150
    output_cost = (output_tokens / 1_000_000) * 0.600
    total_cost = input_cost + output_cost

    print("\n" + "="*60)
    print("COST ESTIMATION")
    print("="*60)
    print(f"Number of libraries: {num_libraries}")
    print(f"Estimated input tokens: {input_tokens:,}")
    print(f"Estimated output tokens: {output_tokens:,}")
    print(f"Estimated input cost: ${input_cost:.4f}")
    print(f"Estimated output cost: ${output_cost:.4f}")
    print(f"TOTAL ESTIMATED COST: ${total_cost:.2f}")
    print("="*60)

    return total_cost


def main():
    """Main execution."""
    print("="*60)
    print("LIBRARY CONCEPT GENERATION")
    print("="*60)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file")
        return

    print("✓ OpenAI API key loaded")

    # Load frequent libraries
    libraries = load_frequent_libraries(min_count=50)

    # Load existing concepts
    existing_concepts = load_existing_concepts()

    # Estimate cost
    to_generate = len([lib for lib in libraries if lib not in existing_concepts])
    estimated_cost = estimate_cost(to_generate)

    if to_generate > 0:
        # Ask for confirmation
        print(f"\nReady to generate concepts for {to_generate} libraries")
        response = input("Proceed? (y/n): ").strip().lower()

        if response != 'y':
            print("Aborted by user")
            return

        # Generate concepts
        all_concepts = generate_concept_batch(
            libraries=libraries,
            existing_concepts=existing_concepts,
            batch_size=10,  # Process 10 at a time
            delay=1.0       # 1 second between batches
        )
    else:
        print("\n✓ All libraries already have concepts!")
        all_concepts = list(existing_concepts.values())

    # Save results
    save_concepts(all_concepts, OUTPUT_PATH)

    # Also save to original location for compatibility
    save_concepts(all_concepts, LIBRARY_CONCEPTS_PATH)

    # Print statistics
    print_statistics(all_concepts)

    print("\n" + "="*60)
    print("CONCEPT GENERATION COMPLETED!")
    print("="*60)
    print(f"\nOutput saved to:")
    print(f"  1. {OUTPUT_PATH}")
    print(f"  2. {LIBRARY_CONCEPTS_PATH}")
    print(f"\nYou can now run the baseline experiments:")
    print(f"  cd experiments")
    print(f"  python3 baseline/baseline_experiment.py")


if __name__ == "__main__":
    main()
