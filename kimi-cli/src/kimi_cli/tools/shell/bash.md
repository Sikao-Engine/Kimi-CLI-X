Execute a shell command.

**Output**
- stdout and stderr are combined and returned.
- On failure, the exit code is included.
- If `run_in_background=true`, a task ID is returned. Use `TaskOutput` (set `block=true` to wait) and `TaskStop` to manage.

**Safety**
- Each call runs in a fresh shell; state is not preserved.
- Do not run interactive or infinite commands.
- Avoid `..` outside the working directory. Do not modify files outside it or run superuser commands unless instructed.

**Efficiency**
- Chain commands with `&&`, `;`, or `||`. Use pipes and quoted paths.
- Use `run_in_background=true` for long-running tasks.

**Available commands**
- Shell: cd, pwd, export, unset, env
- Files: ls, find, mkdir, rm, cp, mv, touch, chmod, chown
- View/edit: cat, grep, head, tail, diff, patch
- Text: awk, sed, sort, uniq, wc
- System: ps, kill, top, df, free, uname, whoami, id, date
- Network: curl, wget, ping, telnet, ssh
- Archive: tar, zip, unzip
- Other: Run `which <command>` to check availability.
