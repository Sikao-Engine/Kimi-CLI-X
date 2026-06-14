---
name: skill-creator
description: Guide for creating effective skills. Use when users want to create or update a skill that extends agent's capabilities.
---

# Skill Creator

Guidance for creating modular skill packages that extend Kimi's capabilities.

## Core Principles

### Concise is Key

Context window is limited. Skills share space with system prompt, history, and user requests. Only add information Kimi doesn't already know. Prefer concise examples over verbose explanations.

### Degrees of Freedom

Match specificity to task fragility:
- **High freedom (text)**: Multiple valid approaches, context-dependent decisions
- **Medium freedom (pseudocode/scripts with params)**: Preferred pattern with acceptable variation
- **Low freedom (specific scripts)**: Fragile operations requiring consistency

### Anatomy

```
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter: name, description, optional type
│   └── Markdown instructions
└── Bundled Resources (optional)
    ├── scripts/      - Executable code
    ├── references/   - Documentation loaded on demand
    └── assets/       - Templates, images, fonts
```

**SKILL.md frontmatter**: `name` and `description` are required. `type` is optional (`standard` or `flow`). Description must include what the skill does AND when to use it.

**SKILL.md body**: Instructions loaded only after skill triggers. Keep under 500 lines.

**Resources**:
- `scripts/`: Deterministic, reusable code (token efficient)
- `references/`: Large docs/schemas loaded when needed (keep >10k word files here)
- `assets/`: Output resources (templates, images)

**Do NOT include**: README.md, CHANGELOG.md, or other auxiliary docs.

### Progressive Disclosure

Three-level loading system:
1. **Metadata** (name + description) - Always in context
2. **SKILL.md body** - When skill triggers
3. **Bundled resources** - As needed

Keep SKILL.md lean; move detailed info to references. Link reference files from SKILL.md with clear guidance on when to read them. Avoid deeply nested references.

**Patterns**:
- High-level guide with references: Core workflow in SKILL.md, details in linked files
- Domain-specific org: `references/finance.md`, `references/sales.md`, etc.
- Conditional details: Basic content in SKILL.md, advanced features linked

## Locations

Skills are discovered from the following locations, in priority order (most specific wins):

1. `--skills-dir` (overrides default discovery)
2. Project: `.kimi/skills`, `.claude/skills`, `.codex/skills`, `.agents/skills`
3. User: `~/.config/agents/skills`, `~/.agents/skills`, `~/.kimi/skills`, `~/.claude/skills`, `~/.codex/skills`
4. Built-in: bundled with kimi-cli

Within each layer, brand-specific directories (`.kimi`, `.claude`, `.codex`) take priority over generic ones (`.agents`, `.config/agents`). Use `--skills-dir` to test skills without placing them in the standard paths.

## Supported Forms

### Subdirectory form (canonical)

```
<skills-root>/<skill-name>/SKILL.md
```

Use this for skills with bundled resources.

### Flat form

```
<skills-root>/<skill-name>.md
```

Use this for single-file skills with no extra resources. The file stem is used as the skill name when `name` is omitted from frontmatter.

## Creation Process

1. **Understand**: Gather concrete usage examples
2. **Plan**: Identify reusable resources (scripts, references, assets)
3. **Initialize**: Create directory with SKILL.md and resource folders
4. **Edit**: Implement resources and write SKILL.md
5. **Validate**: Check frontmatter, naming, structure, and discovery
6. **Iterate**: Improve based on real usage

### Naming

- Lowercase letters, digits, hyphens only
- Under 64 characters
- Verb-led phrases: `gh-address-comments`, `linear-address-issue`
- Folder name matches skill name

### Writing SKILL.md

**Frontmatter**:
```yaml
---
name: skill-name
description: What it does. Use when: (1) condition A, (2) condition B...
---
```

For a flow skill, add `type: flow` and include a `mermaid` or `d2` fenced code block:
```yaml
---
name: approval-flow
description: Route requests through an approval flow. Use when: user asks for gated workflows.
type: flow
---
```

**Body**: Use imperative form. Include:
- Multi-step workflows with decision points
- Output formats and quality standards
- Links to reference files for detailed info

### Testing

Test all scripts before use. For many similar scripts, test a representative sample. Verify the skill is discoverable by running:

```bash
kimi --skills-dir <path-to-parent-skills-root>
```

### Validation

Before considering a skill complete, verify:
- [ ] Frontmatter starts with `---` and ends with `---`
- [ ] `name` matches the directory or file stem
- [ ] `description` is present and describes what + when
- [ ] `type` is either `standard` or `flow` (omit for standard)
- [ ] Flow skills contain a valid `mermaid` or `d2` code block
- [ ] No README.md, CHANGELOG.md, or other auxiliary docs
- [ ] Resource files are referenced clearly from SKILL.md
