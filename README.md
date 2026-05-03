# Kimi-CLI-X

## 快速安装
```bash
# 安装
pip install kimix
# 运行
python -m kimix.cli
# 或
kimix
python -m kimix
```

注意！这个仓库不止支持 KIMI LLM，支持各类 API Key !!! 各类默认 config 模板在 `docs/`, 配置后通过 `kimix --config xx.json` 即可使用 !

![teasor](teasor.png)
## 为什么选择 Kimi-CLI-X？

Kimi-CLI-X 在原版 Kimi-CLI 基础上，围绕**提示词效率**、**工具可靠性**与**可扩展性**进行了深度优化，并补充了多款面向实际开发场景的工具。

### 优化

1. **精简系统提示词** — 压缩初始提示词与工具说明的篇幅，保证信息完整的同时让上下文更干净，覆盖几乎全部内置工具的同时，将初始化 token 降到 2000 左右。
2. **强化权限与校验** — 妥善处理 Shell、Glob 等工具的校验和权限问题，减少因失败导致的反复修正。
3. **优化子进程输出** — 主动将大量输出重定向到临时文件，过滤冗余日志，便于后端检索。
4. **简化并发架构** — 理顺子进程、子代理与后台多任务的设计，使多任务调度更直观可控。
5. **可编程提示词** — 支持在上层自定义、注入系统提示词，灵活适配不同场景。
6. **显式对话管理** — 提供更清晰的多任务编排与对话状态追踪，降低复杂交互的隐晦性。
7. **写入即校验** — 对格式严格的配置文件自动触发格式检查和警告，防止因模型幻觉产生错误。
8. **兼容多种 API** — 支持直接导入自定义配置，兼容 OpenAI、Anthropic 等多种 API 格式。
9. **快速兼容多家API Key** — 已覆盖测试验证的全部后端（kimi、anthropic、openai_legacy、openai_responses、google_genai、vertexai 等），详见 `kimi-cli\tests\core\test_create_llm.py`。

### 新增

| 能力 | 说明 |
|------|------|
| **Input 工具** | 与运行中的进程进行 TUI 交互。 |
| **Docx / PDF 转换** | 内置文档格式转换，无需外部依赖。 |
| **Python 脚本执行** | 允许 Agent 直接执行 Python 脚本。 |
| **错误记录** | 记录工具调用错误，供模型回溯与改进。 |
| **脚本系统** | 将提示词与 Python 逻辑结合，编排复杂任务。 |
| **增强网页抓取 (FetchURL)** | 基于无头浏览器输出 Markdown（而非纯文本），支持 `output_path` 直接落盘与超长内容自动截断；零外部服务依赖，更稳更轻。 |

### 脚本化工作流（核心优势）

与需要人工逐条输入命令的 CLI 交互不同，**Kimi-CLI-X 允许你直接编写 Python 脚本来编排整个工作流**。你可以将提示词、循环、条件判断和工具调用组合在一起，实现全自动、可复现的任务编排：

```python
# Import core API utilities: session management, prompting, plan_mode, etc.
from kimix import set_plan_mode, prompt
from kimix.summarize import summarize
from pathlib import Path

# enable plan mode and start a new context (no resume)
set_plan_mode(value=True, resume=False)

for i in Path('docs').glob('*.md'):
    prompt(f'''
according to the new git commits, update document `{i}`
''')
    summarize() # store memory, fresh context
```

这种方式的优势在于：

- **批量自动化**：结合 Python 的原生语法（如 `for` 循环、文件遍历），一次性向多个目标文件发起任务，无需人工等待和重复输入。
- **编排复杂流程**：在脚本中自由组合模式切换、工具调用与逻辑判断，构建多阶段、多分支的复杂工作流。
- **可复现与可维护**：工作流以脚本形式保存，可纳入版本控制，随时复用、修改和分享，而不是依赖临时的对话历史。

---

### 实验性功能
| **LSP 工具** | 语言服务器协议，辅助代码分析。 |
| **错误列表返回** | 返回工具调用的错误列表，用于生成总结和反省。 |

---

## 文档索引

### 教程系列

| 文档 | 简介 |
|------|------|
| [`docs/tutorials/1_quick_start.md`](docs/tutorials/1_quick_start.md) | **快速入门指南**。涵盖 Git Submodule 拉取、`uv` 环境安装、CLI 启动参数与交互命令的完整说明。 |
| [`docs/tutorials/2_long_task.md`](docs/tutorials/2_long_task.md) | **Long Task**。KimiX 对于长任务的策略。 |
| [`docs/tutorials/3_builtin_tools.md`](docs/tutorials/3_builtin_tools.md) | **内置工具完全指南**。系统介绍 Agent 的全部内置工具（文件 I/O、搜索、代码执行、进程管理、文档转换、计划模式、子代理等），并给出提示词引导策略与最佳实践。 |
| [`docs/tutorials/4_skills.md`](docs/tutorials/4_skills.md) | **自定义 Skill 编写教程**。讲解 Skill 的设计原则、目录结构、`SKILL.md` 编写规范、附属资源组织方式、测试打包流程及安装使用方法。 |
| [`docs/tutorials/5_server.md`](docs/tutorials/5_server.md) | **JSON-RPC 服务端教程**。介绍基于 TCP 的 JSON-RPC 2.0 协议格式、错误码、服务端接口、WebSocket 桥接及命令行启动参数。 |

### 配置参考

| 文件 | 简介 |
|------|------|
| [`docs/config.json`](docs/config.json) | 模型配置示例文件，包含 `model`、`url`、`api_key`、`capabilities` 等字段，可供编写自定义配置时参考。 |

