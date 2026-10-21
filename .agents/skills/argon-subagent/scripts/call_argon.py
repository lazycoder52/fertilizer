import argparse
import os
import sys
from pathlib import Path

# Load environment variables (check workspace .env first, then ~/.gemini/.env)
env_paths = [
    Path(__file__).resolve().parents[4] / ".env",  # Workspace root
    Path.cwd() / ".env",
    Path.home() / ".gemini" / ".env",
]

for env_path in env_paths:
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key or "PASTE_YOUR" in api_key:
    sys.stderr.write("Error: Valid GEMINI_API_KEY not found in .env\n")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.stderr.write("Error: google-genai SDK not installed. Run: pip install google-genai\n")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Call Gemini 4 Argon with telemetry tracking.")
parser.add_argument("--thinking-level", choices=["low", "medium", "high"], default="medium")
parser.add_argument("--prompt", required=True, help="Full context payload for Argon")
args = parser.parse_args()

client = genai.Client(api_key=api_key)

config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level=args.thinking_level)
)

response = client.models.generate_content(
    model="gemini-4-argon", contents=args.prompt, config=config
)

# Print model output
if response.text:
    print(response.text)

# Token Telemetry & Cost Accounting
# Pricing: Uncached Input $2.00/1M | Cached Input $0.10/1M | Output $10.00/1M
usage = response.usage_metadata
uncached_input = getattr(usage, "prompt_token_count", 0) or 0
cached_input = getattr(usage, "cached_content_token_count", 0) or 0
if cached_input > 0 and uncached_input >= cached_input:
    uncached_input -= cached_input

output_tokens = getattr(usage, "candidates_token_count", 0) or 0

cost = (
    (uncached_input * (2.00 / 1_000_000))
    + (cached_input * (0.10 / 1_000_000))
    + (output_tokens * (10.00 / 1_000_000))
)

print("\n" + "=" * 60)
print(
    f"Argon Telemetry: [In: {uncached_input:,} | Cached: {cached_input:,} | Out+Think: {output_tokens:,}] "
    f"Est. Deducted: ${cost:.5f} USD"
)
print("=" * 60)
