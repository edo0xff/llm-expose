# Quick Start

## 1. Add a model

```bash
llm-expose add model --name gpt4o-mini --provider openai --model-id gpt-4o-mini
```

## 2. Add a channel

```bash
llm-expose add channel
```

## 3. Pair a chat/user ID

```bash
llm-expose add pair 123456789 --channel my-telegram
```

## 4. Run the exposure service

```bash
llm-expose start --channel my-telegram
```

Use `llm-expose <command> --help` to inspect all options.
