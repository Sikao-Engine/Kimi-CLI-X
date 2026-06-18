---

The above context contains **multiple previous compaction summaries** that have been recursively summarized. Your task is to extract a **flat, deduplicated list of key facts** from the entire history.

**Rules:**
- Output a bulleted list of non-redundant facts, decisions, and current file states.
- **De-duplicate:** If the same fact appears across multiple previous summaries, include it only once.
- **Discard** narrative flow, transitional language, and meta-commentary about the compaction process itself.
- **Preserve:** error messages, final solutions, tool output results, architectural decisions, design rationale, and current task state.
- **Condense:** long code blocks → signatures + key logic only (keep full version if < 20 lines).

**Length:** Aim to reduce the context to a compact fact list while preserving all essential information.

**Output Structure:**

```xml
<current_focus>
[What we're working on now]
</current_focus>

<environment>
- OS: [os]
- Work dir: [path]
- Key deps: [packages]
- [Other relevant setup]
</environment>

<code_state>
[Critical file states — signatures + key changes]
</code_state>

<facts>
- [Decision] [Decision description and rationale]
- [Code] [File path / function / key logic]
- [Env] [Environment detail]
- [Error] [Error message and resolution]
</facts>

<active_issues>
- [Issue]: [Status/Next steps]
</active_issues>
```
