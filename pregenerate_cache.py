"""
Pre-generates cached JSON for all resumes to avoid LLM rate limits.
Run this once after data.py to populate data/parsed_resumes/ with JSON cache.
"""

import os
import sys
from utils import parse_and_index_resumes

print("Pre-generating resume cache to avoid LLM rate limits...")
print("This may take a minute the first time...\n")

count, errors = parse_and_index_resumes()

print(f"\n✅ Pre-generation complete!")
print(f"   Indexed {count} resumes into ChromaDB")

if errors:
    print(f"\n⚠️  {len(errors)} warning(s):")
    for err in errors:
        print(f"   - {err}")
else:
    print("   No errors!")

print("\nReady to run: python app.py")
print("Or try the demo: python app.py --test")
