---
name: kimi-cli-help
description: Answer Kimi Code CLI usage, configuration, and troubleshooting questions. Use when user asks about Kimi Code CLI installation, setup, configuration, slash commands, keyboard shortcuts, MCP integration, providers, environment variables, how something works internally, or any questions about Kimi Code CLI itself.
---

# Kimi Code CLI Help

Answer Kimi CLI questions using official docs and source code.

## Strategy

1. **Prefer official docs** for most questions
2. **Read local source** when in kimi-cli project or user imports `kimi_cli`
3. **Clone source** for complex internals - ask confirmation first

## Documentation

Base URL: `https://moonshotai.github.io/kimi-cli/`

Fetch index: `https://moonshotai.github.io/kimi-cli/llms.txt`

| Topic | Page (prepend `/en/` or `/zh/`) |
|-------|--------------------------------|
| Installation | `/guides/getting-started.md` |
| Config files | `/configuration/config-files.md` |
| Providers, models | `/configuration/providers.md` |
| Environment variables | `/configuration/env-vars.md` |
| Slash commands | `/reference/slash-commands.md` |
| CLI flags | `/reference/kimi-command.md` |
| Keyboard shortcuts | `/reference/keyboard.md` |
| MCP | `/customization/mcp.md` |
| Agents | `/customization/agents.md` |
| Skills | `/customization/skills.md` |
| FAQ | `/faq.md` |

## Source Code

Repository: `https://github.com/MoonshotAI/kimi-cli`

Read source when: in kimi-cli project (check `pyproject.toml`), user imports `kimi_cli`, or internals question not covered by docs.
