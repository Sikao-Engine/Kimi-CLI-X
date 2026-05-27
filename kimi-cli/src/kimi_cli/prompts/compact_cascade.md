---

The above context contains **multiple previous compaction summaries** that have been recursively summarized. Your task is to extract a **flat, deduplicated list of key facts** from the entire history.

**Rules:**
- Output **only** a bulleted list of non-redundant facts, decisions, and current file states.
- **De-duplicate:** Do not repeat facts that already appear in earlier summaries.
- **Discard** narrative flow, transitional language, and meta-commentary about the compaction process itself.
- **Preserve:** error messages, final solutions, architectural decisions, and current task state.
- **Condense:** long code blocks → signatures + key logic only (keep full version if < 20 lines).

**Output Structure:**

```xml
<current_focus>
[What we're working on now]
</current_focus>

<facts>
- [Fact/decision/file state]
- [Fact/decision/file state]
</facts>

<active_issues>
- [Issue]: [Status/Next steps]
</active_issues>
```
