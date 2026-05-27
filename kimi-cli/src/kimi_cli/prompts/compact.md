---

Compact the above agent conversation context according to the following priorities and rules.

**Priorities (ordered):**
1. **Current Task State** — what is being worked on right now
2. **Errors & Solutions** — all errors encountered and how they were resolved
3. **Code Evolution** — final working versions only (drop intermediate attempts)
4. **System Context** — project structure, dependencies, environment setup
5. **Design Decisions** — architectural choices and rationale
6. **TODO Items** — unfinished tasks and known issues

**Rules:**
- **Keep:** error messages, stack traces, working solutions, current task
- **Merge:** similar discussions into single summary points
- **Remove:** redundant explanations, failed attempts (retain lessons learned), verbose comments
- **Condense:** long code blocks → signatures + key logic only

**Special Handling:**
- **Code:** keep full version if < 20 lines; otherwise keep signature + key logic
- **Errors:** keep full error message + final solution
- **Discussions:** extract decisions and action items only

**Output Structure:**

```xml
<current_focus>
[What we're working on now]
</current_focus>

<environment>
- [Key setup/config points]
</environment>

<completed_tasks>
- [Task]: [Brief outcome]
</completed_tasks>

<active_issues>
- [Issue]: [Status/Next steps]
</active_issues>

<code_state>
<file>
[filename]

**Summary:** [What this file does]

**Key elements:**
- [Important functions/classes]

**Latest version:**
[Critical code snippets]
</file>
</code_state>

<important_context>
- [Crucial information not covered above]
</important_context>
```
