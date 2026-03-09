## Plan: LiteLLM MCP Tool Support

Add first-class MCP tool support by passing tool definitions directly to LiteLLM (`acompletion`) in both non-streaming and streaming paths, while keeping orchestration simple (no local tool execution loop). This aligns with LiteLLM MCP gateway behavior where tool discovery/execution/result reinjection can be handled server-side. The `BaseTool` abstraction should be deprecated for now, not expanded, because current scope is MCP-only.

**Steps**
1. Phase 1 - Provider contract update
2. Update `BaseProvider.complete` and `BaseProvider.stream` signatures to accept optional tool arguments and optional tool-choice metadata while preserving backward compatibility for existing callers. This step blocks provider implementation and tests.
3. Define narrow typed aliases for request messages/tools payloads in provider layer (or use precise `dict[str, Any]` with docstrings) so MCP tool payload shape is explicit and discoverable. This can be done in parallel with step 2 if isolated.
4. Phase 2 - LiteLLM provider MCP wiring
5. In `LiteLLMProvider.complete`, forward optional `tools` and optional `tool_choice` into both branches: local OpenAI-compatible path and remote `litellm.acompletion` path. Keep existing return behavior (`str`) for now so orchestrator/client behavior does not regress.
6. In `LiteLLMProvider.stream`, forward optional `tools` and optional `tool_choice` similarly for both branches, while preserving current chunk filtering semantics (`delta.content` only). This depends on step 5 only for consistency, but can be implemented in parallel after step 2.
7. Ensure `_common_kwargs` remains provider/auth focused and does not silently swallow tool args; tool args should be passed explicitly at call sites for readability and testability.
8. Phase 3 - MCP config to request transformation
9. Add a helper (provider-side or small dedicated module) that maps configured MCP servers from `MCPConfig` to LiteLLM/OpenAI-compatible MCP `tools` entries, including `type: "mcp"`, server identifier/url, and optional `allowed_tools` passthrough if available in config. This blocks orchestrator wiring.
10. Wire orchestrator to load MCP config once (or via injectable loader for testability), filter `enabled` servers, build `tools` payload, and pass it to provider on each call. Keep message history format unchanged. This depends on step 9.
11. Apply MCP settings mapping in orchestrator request options where applicable (for example confirmation mode -> `require_approval` strategy when constructing tool entries). If one-to-many mapping is ambiguous, centralize it in one helper and document chosen defaults.
12. Phase 4 - BaseTool decision and cleanup
13. Mark `llm_expose/tools/base.py` and `llm_expose/tools/__init__.py` as legacy/local-tools path: update docstrings/TODOs to clarify MCP-first architecture and that local `BaseTool.execute()` loop is intentionally out of scope for this change.
14. Do not remove `BaseTool` in this iteration to avoid breaking imports; instead deprecate in docs/comments and leave interface stable until a dedicated removal migration is approved.
15. Phase 5 - Tests and verification hardening
16. Update `tests/test_providers.py` to assert `tools`/`tool_choice` are forwarded to `litellm.acompletion` and local `AsyncOpenAI` calls in both complete and stream paths.
17. Add orchestrator tests in `tests/test_clients.py` verifying MCP tools payload is built from enabled MCP servers and passed to `provider.complete` without changing history growth behavior.
18. Add config-level tests in `tests/test_config.py` only if new MCP fields are introduced (e.g., allowed-tools list); otherwise keep existing coverage unchanged.
19. Run full test suite and inspect for signature fallout in mocks/fakes; adjust all call sites to satisfy updated provider interface.

**Relevant files**
- `c:\Users\eduar\Documents\GitHub\llm-expose\llm_expose\providers\base.py` - extend abstract provider method signatures for optional tools/tool choice.
- `c:\Users\eduar\Documents\GitHub\llm-expose\llm_expose\providers\litellm_provider.py` - pass MCP tool payload into `acompletion`/OpenAI client in complete and stream.
- `c:\Users\eduar\Documents\GitHub\llm-expose\llm_expose\core\orchestrator.py` - build MCP tool payload from config and pass to provider per request.
- `c:\Users\eduar\Documents\GitHub\llm-expose\llm_expose\config\loader.py` - reuse `load_mcp_config()` as source of MCP server definitions.
- `c:\Users\eduar\Documents\GitHub\llm-expose\llm_expose\config\models.py` - only modify if MCP schema needs additional fields for tool restrictions.
- `c:\Users\eduar\Documents\GitHub\llm-expose\llm_expose\tools\base.py` - deprecation/positioning update for BaseTool under MCP-first strategy.
- `c:\Users\eduar\Documents\GitHub\llm-expose\llm_expose\tools\__init__.py` - package doc clarity update.
- `c:\Users\eduar\Documents\GitHub\llm-expose\tests\test_providers.py` - forwarding assertions for tools/tool_choice in both execution paths.
- `c:\Users\eduar\Documents\GitHub\llm-expose\tests\test_clients.py` - orchestrator MCP tools pass-through tests.
- `c:\Users\eduar\Documents\GitHub\llm-expose\tests\test_config.py` - optional schema tests if MCP model changes.

**Verification**
1. Run `pytest -q` and confirm all existing tests pass with updated provider signatures.
2. Add/execute targeted provider tests that inspect mocked `litellm.acompletion` kwargs for `tools` and `tool_choice` in `complete` and `stream`.
3. Add/execute targeted orchestrator tests confirming enabled MCP servers from config are transformed into tools payload and sent to `provider.complete`.
4. Manual smoke check: configure one MCP server in local config, send a prompt requiring a tool, and confirm response arrives without orchestrator-side tool loop errors.
5. Manual streaming smoke check: same MCP setup with streaming enabled, verify token stream still emits text chunks and no regressions in chunk filtering.

**Decisions**
- Chosen approach: MCP-only tool execution via LiteLLM gateway; no local Python tool execution loop in orchestrator.
- `BaseTool` is not required for this scope; keep temporarily for backward compatibility but treat as deprecated/legacy.
- Scope includes complete+stream parity for tool argument forwarding.
- Scope excludes implementing concrete local tools (calculator/web search) and excludes orchestrator-managed tool-call recursion.

**Further Considerations**
1. MCP payload shape differs between Chat Completions and Responses APIs; recommendation: keep current chat-completions flow and normalize to the tool schema accepted by `litellm.acompletion` in this codebase.
2. If local provider path does not support MCP tool payload uniformly, recommendation: feature-flag MCP pass-through by provider type with explicit warning/log rather than silent failure.
3. In a later cleanup, either fully remove `BaseTool` or implement a true hybrid strategy; recommendation: defer until usage telemetry or explicit product direction requires local execution.
