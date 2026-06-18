---

Compact the above agent conversation context.

**What to keep (ordered by priority):**
1. **Current Task State** — what is being worked on right now, plus any user-supplied custom instructions, preferences, or constraints for future turns.
2. **Errors & Solutions** — preserve the full error message and the final working solution. For multi-turn debugging, summarize intermediate steps as a brief narrative (1-2 lines).
3. **Code State** — final working versions only (drop intermediate attempts).
4. **Design Decisions** — architectural choices and rationale.
5. **Environment** — OS, work directory, Python version, key dependencies, and other relevant setup.
6. **TODO Items** — unfinished tasks and known issues.

**What to remove or condense:**
- **Drop:** redundant explanations, failed intermediate attempts (retain lessons learned), verbose comments, conversational filler.
- **Merge:** similar discussions into single summary points.
- **Condense code:** 
  - Keep full version if ≤ 20 lines.
  - For longer code, keep signature + **key logic** only.
  
  **Key logic** means:
  - The core algorithm or business logic (not boilerplate/imports)
  - Critical control flow (conditionals, loops, error handling)
  - Non-obvious transformations or side effects
  - Exclude: imports, logging, type annotations, docstrings, setup/teardown boilerplate

**Length:** Aim to reduce the context to approximately 20-30% of the original length while preserving all essential information. Err on the side of brevity for aggressive mode and completeness for retentive mode.

**User Instructions:** Preserve any explicit user preferences, constraints, or custom compaction instructions for future turns.

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

<completed_tasks>
- [Task]: [Brief outcome]
</completed_tasks>

<active_issues>
- [Issue]: [Status/Next steps]
</active_issues>

<todo>
- [ ] [Unfinished task]
</todo>

<code_state>
<file name="path/to/file.py">
<summary>What this file does</summary>
<key_elements>
- FunctionA: does X
- ClassB: handles Y
</key_elements>
<latest_version>
[Critical code snippets]
</latest_version>
</file>
</code_state>

<decisions>
- [Decision]: [Rationale]
</decisions>

<important_context>
- [Crucial information not covered above]
</important_context>
```
