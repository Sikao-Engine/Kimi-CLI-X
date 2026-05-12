from typing import Optional, Callable
from pathlib import Path
import os
from enum import Enum
from kaos.path import KaosPath
import kimix.base as base
from kimi_cli.soul.agent import BuiltinSystemPromptArgs

# Concise system prompt to reduce LLM overthinking and hallucination
_SYSTEM_PROMP = (
    '{AGENT_ROLE}:\n{NUMBERED}\n{AGENTS_MD}\n{SKILLS}\n{EXTRA}'
)

class SystemPromptType(Enum):
    Worker = 0
    TodoMaker = 1
    Thinker = 2
    SwarmCoordinator = 3
    Recaller = 4
    SkillSearcher = 5,
    TrivialSubAgent = 6
    

def get_system_prompt(
        yolo: bool | None = None,
        work_dir: Optional[KaosPath] = None,
        extra_system_prompt: str | None = None,
        agent_role: SystemPromptType = SystemPromptType.Worker,
) -> Callable[[BuiltinSystemPromptArgs], str]:
    agent_md = (Path(str(work_dir)) if work_dir is not None else Path(
        os.curdir)) / 'AGENTS.md'
    yolo = yolo if yolo is not None else base._default_yolo

    def system_prompt_func(args: BuiltinSystemPromptArgs) -> str:
        items: list[str] = []
        agent_md_doc = ''
        skill_doc = ''
        use_agent_md = False
        use_skills = False
        if agent_role != SystemPromptType.Thinker:
            items.append('No pseudocode, flowcharts, reasoning, planning, filler, restating, or self-correction. Act directly.')
        def worker_logic(role: str):
            nonlocal role_doc, use_agent_md, use_skills
            use_agent_md = True
            use_skills = True
            role_doc = f'You are a {role}'
            items.append('Interactive: `Run` short timeout, then `TaskOutput`/`Input`.')
            items.append('Python: `python -c <code>`.')
            items.append('Multi-step: use `SetTodoList`. Finish all before ending.')
            if args.KIMI_OS != 'Windows':
                items.append(f'Shell: {args.KIMI_SHELL}. Use `Run`.')
            else:
                items.append('No Shell, use `Run`.')
            if yolo:
                items.append('Yolo: no asking. Stay in workdir.')
            items.append('`SkillSearch` to find skills.')
            items.append('Drop context aggressively. `Remember` important/long-running info.')
            items.append('`Forget` stale or duplicate info.')
            items.append('`Recall` before any work.')
        match agent_role:
            case SystemPromptType.Worker:
                worker_logic('terse coder')
            case SystemPromptType.TodoMaker:
                use_agent_md = True
                use_skills = True
                role_doc = 'You are a planner'
                items.append('Plan only. Do not implement.')
                items.append('Record steps with `Note`.')
                items.append('No multiple steps at once.')
                items.append('Use `Agent` to enable sub-agent, for research, analyze, find, retrieval.')
            case SystemPromptType.SwarmCoordinator:
                use_agent_md = True
                use_skills = True
                role_doc = 'You are a swarm coordinator'
                items.append('Build DAG with `AddNode` and `AddEdge`.')
                items.append('AddNode: clear, actionable sub-task prompt')
                items.append('AddEdge: upstream → downstream')
                items.append('Keep acyclic. Minimize edges, maximize parallelism.')
                items.append('Report nodes and edges.')
                items.append('`SkillSearch` for skills.')
            case SystemPromptType.Thinker:
                worker_logic('thinker')
                items.append('Think in <thinking>...</thinking>. End with <quit/>. Concise. No text outside tags.')
                items.append('Self-verify: catch errors and bad assumptions.')
            case SystemPromptType.Recaller:
                role_doc = 'You are a memory recaller'
                items.append('Reject write/edit tasks')
                items.append('Use `Recall` and `Reflect`.')
                items.append('Multi-step: use `SetTodoList`.')
                items.append('Search, analyze, report concisely. Read-only.')
                items.append('`SkillSearch` for skills.')
            case SystemPromptType.SkillSearcher:
                use_skills = True
                role_doc = 'You are a skill searcher'
                items.append('Reject write/edit tasks')
                items.append('`SkillSearch` to find skills.')
                items.append('Multi-step: use `SetTodoList`.')
                items.append('Search, analyze, report concisely. Read-only.')
            case SystemPromptType.TrivialSubAgent:
                use_skills = True
                role_doc = 'You are a read-only sub-agent'
                items.append('Reject write/edit tasks')
                items.append('`SkillSearch` to find skills.')
                items.append('Multi-step: use `SetTodoList`.')
                items.append('Search, analyze, report concisely. Read-only.')


        if use_agent_md and agent_md.is_file():
            agent_md_content = agent_md.read_text(encoding='utf-8', errors='replace')
            agent_md_doc = f'AGENTS.md:\n```\n{agent_md_content}\n```\n'
        if use_skills and args.KIMI_SKILLS:
            skill_doc = f'Skills:\n{args.KIMI_SKILLS}\n'
        numbered_block = ''
        if items:
            numbered_block = ''.join(
                f'- {item}\n' for item in items
            )

        return _SYSTEM_PROMP.format(
            AGENT_ROLE=role_doc.strip(),
            NUMBERED=numbered_block,
            AGENTS_MD=agent_md_doc,
            SKILLS=skill_doc,
            EXTRA=extra_system_prompt if extra_system_prompt is not None else ''
        ).strip()
    return system_prompt_func
