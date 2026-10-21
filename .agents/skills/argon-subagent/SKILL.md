---
name: argon-subagent
description: Delegates hard reasoning, complex architectural refactoring, and deep debugging tasks to gemini-4-argon via API using developer credits.
rules:
  - Trigger ONLY when the user explicitly includes phrases like "use Argon", "ask Argon", or "run Argon". Never invoke automatically.
  - Do NOT run unattended execution loops. Run once per explicit user request.
  - Before invoking, use standard Antigravity tools (Gemini Pro/Flash) to inspect local files, logs, and terminal state. Compile all relevant context into a single structured prompt payload.
  - Structure the prompt with static references and system guidelines first (>4,096 tokens) so the call benefits from implicit context caching discounts.
  - Pass the compiled prompt into `.agents/skills/argon-subagent/scripts/call_argon.py`.
  - Accept `--thinking-level` flags (`low`, `medium`, `high`). Default to `medium` if unspecified.
---

# Argon Subagent Skill

This skill delegates deep reasoning, complex architectural refactoring, or critical debugging tasks to `gemini-4-argon`.

## Usage Instructions
When the user explicitly invokes Argon (e.g., *"use Argon to solve this"*, *"ask Argon"*):

1. **Context Compilation**:
   Collect relevant code files, terminal logs, or error traces using standard inspection tools (`view_file`, `grep_search`).

2. **Invoke Runner**:
   Execute the runner script via `run_command`:
   ```powershell
   python .agents/skills/argon-subagent/scripts/call_argon.py --thinking-level medium --prompt "<COMPILED_CONTEXT_AND_TASK>"
   ```

3. **Incorporate Results**:
   Read the output and telemetry provided by Argon to guide final code implementation or answer the user's request.
