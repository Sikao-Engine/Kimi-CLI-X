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
│   ├── YAML frontmatter: name, description
│   └── Markdown instructions
└── Bundled Resources (optional)
    ├── scripts/      - Executable code
    ├── references/   - Documentation loaded on demand
    └── assets/       - Templates, images, fonts
```

**SKILL.md frontmatter**: `name` and `description` are the only fields Kimi reads to determine when to use the skill. Description must include what the skill does AND when to use it.

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

- User: `~/.config/agents/skills/`, `~/.kimi/skills/`, `~/.claude/skills/`
- Project: `.agents/skills/`
- `--skills-dir` overrides discovery

## Creation Process

1. **Understand**: Gather concrete usage examples
2. **Plan**: Identify reusable resources (scripts, references, assets)
3. **Initialize**: Create directory with SKILL.md and resource folders
4. **Edit**: Implement resources and write SKILL.md
5. **Package**: Create `<skill-name>.skill` zip archive
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

**Body**: Use imperative form. Include:
- Multi-step workflows with decision points
- Output formats and quality standards
- Links to reference files for detailed info

### Testing

Test all scripts before packaging. For many similar scripts, test a representative sample.

### Packaging

```bash
cd <skills-root>
zip -r my-skill.skill my-skill
```

Validate before packaging: frontmatter format, required fields, naming conventions, file organization.
