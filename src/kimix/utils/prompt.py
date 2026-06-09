import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from kimi_agent_sdk import Session

import kimix.base as base
from kimix.base import Color, MessageType, Style, print_agent_json
from kimix.tools.common import _export_to_temp_file
from .session import _create_default_session, _create_session_async, close_session_async, _print_usage
from .system_prompt import SystemPromptType


async def prompt_async(
    prompt_str: str,
    session: Session | None = None,
    # settings
    output_function: Callable[[str, MessageType], Any] | None = None,
    info_print: bool = True,
    cancel_callable: Callable[[], bool] | None = None,
    close_session_after_prompt: bool = False,
    merge_wire_messages: bool | None = None
) -> None:
    from kimix.utils.prompt_str import escape_file_paths
    if session is None:
        session = _create_default_session()
        close_session_after_prompt = False
    prompt_str = prompt_str.strip()
    prompt_str = escape_file_paths(prompt_str)
    if len(prompt_str) > 65536:  # too long, save to file
        name, new_id = _export_to_temp_file(content=prompt_str)
        prompt_str = f'read and execute: `{name}`'
    try:
        if info_print:
            base._stream.colorful_print_word(f'Start...\n', fg=base.Color.BRIGHT_CYAN, require_new_line=True)

        max_retries = 5
        prompt_success = False
        for attempt in range(max_retries):
            if session._cancel_event is not None and session._cancel_event.is_set():
                break
            try:
                import time
                start_time = time.time()
                base._stream._last_char_was_newline = True
                if merge_wire_messages is None and output_function is not None:
                    merge_wire_messages = True
                async for message in session.prompt(prompt_str, merge_wire_messages=merge_wire_messages if merge_wire_messages is not None else False):
                    if cancel_callable is not None and cancel_callable():
                        session.cancel()
                        break
                    print_agent_json(message, session, output_function)
                base._stream.print_word('\n', require_new_line=True)
                if info_print:
                    end_time = time.time()
                    _print_usage(session, end_time - start_time)
                prompt_success = True
                break
            except KeyboardInterrupt as e:
                if session:
                    session.cancel()
                break
            except Exception as e:
                base._stream.colorful_print_word(str(e), fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True)
                if session:
                    session.cancel()
                if "429" in str(e) or "400" in str(e) or "500" in str(e) or "502" in str(e) or "503" in str(e):
                    wait_time = min(2 ** attempt, 60)
                    base._stream.colorful_print_word(f"Rate limited. Waiting {wait_time}s...", fg=Color.BRIGHT_YELLOW, styles=[Style.BOLD], require_new_line=True)
                    await asyncio.sleep(wait_time)
                elif attempt == max_retries - 1:
                    raise
                else:
                    await asyncio.sleep(1)

        if not prompt_success:
            base._stream.colorful_print_word('prompt failed.', fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True)

    finally:
        if close_session_after_prompt and session:
            await close_session_async(session)
        base._stream.print_word('', True)


def prompt(
    prompt_str: str,
    session: Session | None = None,
    # settings
    output_function: Callable[[str, MessageType], Any] | None = None,
    info_print: bool = True,
    cancel_callable: Callable[[], bool] | None = None,
    close_session_after_prompt: bool = False,
    merge_wire_messages: bool | None = None
) -> None:
    asyncio.run(
        prompt_async(
            prompt_str,
            session,
            # settings
            output_function,
            info_print,
            cancel_callable,
            close_session_after_prompt,
            merge_wire_messages=merge_wire_messages
        ))


def prompt_path(path: Path, split_word: Optional[str] = None, session: Session | None = None, after_prompt_coro: Any = None) -> None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            s = f.read()
    except Exception:
        base._stream.colorful_print_word(f'File {str(path)} not found.', fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True)
    coro = None
    if after_prompt_coro is not None:
        coro = after_prompt_coro()
    if split_word:
        words = s.strip().split(split_word)
        for i in words:
            prompt(i, session=session)
            if coro is not None:
                try:
                    next(coro)
                except StopIteration as e:
                    coro = None
    else:
        prompt(s, session=session)
        if coro is not None:
            try:
                next(coro)
            except StopIteration as e:
                coro = None


async def prompt_plan_async(requirement: str, plan_file: str | Path = "plan.md") -> None:
    from kimix.tools.note import _enable_plan

    plan_file = Path(plan_file)
    if plan_file.is_file():
        plan_file.unlink()

    _enable_plan.value = True
    planner_session: Session | None = None

    try:
        planner_session = await _create_session_async(agent_type=SystemPromptType.TodoMaker, agent_file='agent_planner.json')
        planner_session.get_custom_data()["plan_writing_path"] = plan_file

        reminder = (
            "read the following requirement carefully and generate a comprehensive plan. "
            f"Save the complete plan to a file using the WritePlan tool.\n\n"
            f"Requirement:\n{requirement.strip()}"
        )

        max_plan_attempts = 3
        plan_generated = False
        for attempt in range(max_plan_attempts):
            if planner_session._cancel_event is not None and planner_session._cancel_event.is_set():
                break
            try:
                base._stream.colorful_print_word(
                    f"Generating plan (attempt {attempt + 1}/{max_plan_attempts})...\n",
                    fg=Color.BRIGHT_CYAN,
                    require_new_line=True,
                )
                async for message in planner_session.prompt(reminder):
                    print_agent_json(message, planner_session, None)
                base._stream.print_word("\n", require_new_line=True)

                if plan_file.exists() and plan_file.stat().st_size > 0:
                    plan_generated = True
                    break

                if attempt < max_plan_attempts - 1:
                    reminder = (
                        "The plan file was not generated. "
                        "Please generate the plan and save it using the WritePlan tool.\n\n"
                        f"Requirement:\n{requirement.strip()}"
                    )
            except KeyboardInterrupt:
                if planner_session:
                    planner_session.cancel()
                break
            except Exception as exc:
                base._stream.colorful_print_word(
                    str(exc), fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True
                )
                if planner_session:
                    planner_session.cancel()
                if attempt == max_plan_attempts - 1:
                    raise
                await asyncio.sleep(1)
        if not plan_generated:
            base._stream.colorful_print_word(
                "Plan generation failed: plan file not found.",
                fg=Color.BRIGHT_RED,
                styles=[Style.BOLD],
                require_new_line=True,
            )
            return

        def _open_plan_file(filepath: Path) -> None:
            """Open the plan file with the system default application."""
            try:
                if sys.platform == "win32":
                    os.startfile(str(filepath))
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(filepath)])
                else:
                    subprocess.run(["xdg-open", str(filepath)])
            except Exception:
                pass

        base._stream.colorful_print_word(
            f"Plan generated: {plan_file.absolute()}\n",
            fg=Color.BRIGHT_GREEN,
            styles=[Style.BOLD],
            require_new_line=True,
        )
        _open_plan_file(plan_file)

        # Review loop: let the user approve or request revisions
        execute_plan = True
        while True:
            user_input = await asyncio.to_thread(
                input, "Do you want to implement the plan? (y/n): "
            )
            if user_input.strip().lower() == "y":
                break

            # User wants to revise the plan — get feedback and loop back to planner
            feedback = await asyncio.to_thread(
                input, "Please describe the changes you want (/quit to give up): "
            )
            feedback = feedback.strip()
            if not feedback:
                continue
            if feedback.lower() == '/quit':
                execute_plan = False
                break
            revision_reminder = (
                "The user reviewed the plan and wants the following changes:\n\n"
                f"{feedback.strip()}\n\n"
                "Please update the plan file accordingly using the WritePlan or EditPlan tools."
            )
            try:
                base._stream.colorful_print_word(
                    "Revising plan...\n",
                    fg=Color.BRIGHT_CYAN,
                    require_new_line=True,
                )
                async for message in planner_session.prompt(revision_reminder):
                    print_agent_json(message, planner_session, None)
                base._stream.print_word("\n", require_new_line=True)

                # Re-open the updated plan file for review
                if plan_file.exists():
                    _open_plan_file(plan_file)
            except KeyboardInterrupt:
                if planner_session:
                    planner_session.cancel()
                return
            except Exception as exc:
                base._stream.colorful_print_word(
                    f"Revision failed: {exc}",
                    fg=Color.BRIGHT_RED,
                    styles=[Style.BOLD],
                    require_new_line=True,
                )
                # Continue the loop so the user can try again

        # User approved — close planner session, then proceed to implementation
        if planner_session:
            await close_session_async(planner_session)
            planner_session = None
        _enable_plan.value = False
        if not execute_plan:
            return

        if not plan_file.exists():
            base._stream.colorful_print_word(
                f"Plan file {plan_file} no longer exists. Aborting.",
                fg=Color.BRIGHT_RED,
                styles=[Style.BOLD],
                require_new_line=True,
            )
            return

        plan_content = plan_file.read_text(encoding="utf-8", errors="replace")
        regular_session = _create_default_session()
        await prompt_async(
            f"Please implement the following plan:\n\n{plan_content}",
            session=regular_session,
        )

        review_reminder = (
            "Please review the last plan that was just implemented and ensure all tasks are finished. "
            "Verify every item in the plan has been completed successfully, and address any remaining issues.\n\n"
            f"Plan:\n{plan_content}"
        )
        await prompt_async(
            review_reminder,
            session=regular_session,
        )
    except Exception as exc:
        base._stream.colorful_print_word(
            f"prompt_plan failed: {exc}",
            fg=Color.BRIGHT_RED,
            styles=[Style.BOLD],
            require_new_line=True,
        )
    finally:
        _enable_plan.value = False
        if planner_session:
            await close_session_async(planner_session)


def prompt_plan(requirement: str, plan_file: str | Path = "plan.md") -> None:
    asyncio.run(prompt_plan_async(requirement, plan_file))
