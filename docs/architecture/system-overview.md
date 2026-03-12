# System Overview

`llm-expose` connects four primary layers:

1. Client adapters (`telegram`, `discord`) receive and send messages.
2. Orchestrator coordinates history, provider calls, and runtime behavior.
3. Provider layer (LiteLLM) resolves model completions.
4. MCP runtime manages tool-aware integrations.

This separation keeps channel integrations independent from provider and tooling concerns.
