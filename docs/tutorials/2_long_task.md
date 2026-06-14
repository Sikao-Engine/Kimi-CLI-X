# 长任务处理

对于复杂或耗时的任务，Kimix 提供两种主要处理方式：

1. **`/plan` 命令**：让专门的 Planner Agent 生成可审阅的任务计划，确认后再由 Worker Agent 串行实现并复核
2. **`/swarm` 命令**：通过 Agent Swarm 将任务拆分为 DAG，并行调度多个子 Agent

---

## /plan 命令

`/plan` 命令采用**计划生成 → 用户审阅 → 执行实现 → 复核**的流程处理长任务。适合需要先明确步骤、再经用户确认后执行的场景。

### 基本用法

进入交互终端后执行：

```
/plan
```

进入多行输入模式：
- 以 `/end` 结束输入提交任务描述
- 以 `/cancel` 取消当前操作

**示例**：

```text
/plan
>>>> Start input requirement for plan, end with /end, cancel with /cancel
为这个项目添加完整的错误处理机制：
1. 为所有函数添加参数校验
2. 统一异常类型和错误码
3. 添加日志记录
/end
```

### 执行流程 (`prompt_plan_async`)

参考 `src/kimix/utils/prompt.py` 和 `src/kimix/cli_impl/commands.py`：

#### 阶段 1：计划生成

1. **创建 Planner 会话**：使用 `agent_planner.json` 配置和 `TodoMaker` 系统提示词创建专用会话。
2. **指定计划文件**：若未提供文件路径，则在当前目录的 `.kimix_cache/` 下自动生成 `plan_<uuid>.md`；若指定的文件已存在，会先删除。
3. **生成计划**：Planner 读取任务需求，将任务拆解为步骤列表，并通过 `WritePlan` 工具写入计划文件。生成过程最多尝试 **3 次**，确保计划被正确写入。
4. **打开审阅**：计划生成后会用系统默认程序打开该文件，供用户查看。

#### 阶段 2：审阅与修订

计划生成后进入审阅循环：

- 提示：`Do you want to implement the plan? (y/n)`
  - 输入 `y`：进入执行阶段。
  - 输入 `n` 或其他：进入修订流程。
- 修订提示：`Please describe the changes you want (/quit to give up):`
  - 输入 `/quit`：放弃执行。
  - 输入具体修改意见：Planner 会根据反馈使用 `WritePlan` 或 `EditPlan` 工具更新计划文件，并再次打开供审阅，循环往复直到用户确认或放弃。

#### 阶段 3：执行与复核

1. **关闭 Planner 会话**：用户确认后，Planner 会话会被关闭。
2. **创建 Worker 会话**：使用默认的 Worker Agent 创建常规会话。
3. **发送实现提示**：
   - 若计划文件小于 **100 KB**，会直接将计划内容嵌入提示词；
   - 若计划文件大于 **100 KB**，则提示 Agent 读取计划文件并执行。
4. **追加复核提示**：实现完成后，再发送一次 review 提示，要求 Agent 检查计划是否全部完成。

### 指定计划输出文件

可通过 `/plan:<file>` 指定计划文件的输出路径。注意：该文件是**计划输出文件**，任务需求仍需通过多行输入提供：

```
/plan:docs/plan.md
>>>> Start input requirement for plan, end with /end, cancel with /cancel
为这个项目添加完整的错误处理机制：
1. 为所有函数添加参数校验
2. 统一异常类型和错误码
3. 添加日志记录
/end
```

若指定的计划文件已存在，会被覆盖。

---

## 长提示词与错误处理

参考 `src/kimix/utils/prompt.py` 中的 `prompt_async`：

### 自动截断超长提示词

当输入的提示词超过 **65536 字符**时，系统会自动将其导出到临时文件，并将提示词替换为 `read and execute: <temp_file>`，避免超出模型上下文限制。

### 自动重试机制

`prompt_async` 内置了最多 **5 次**的重试逻辑，遇到以下 HTTP 状态码时会自动指数退避等待：

| 状态码 | 行为 |
|--------|------|
| `429` | 请求过快，按 `min(2^attempt, 60)` 秒等待后重试 |
| `400`, `500`, `502`, `503` | 服务端错误，同样触发指数退避 |
| 其他异常 | 等待 1 秒后重试，最后一次直接抛出 |

这意味着在 `/plan` 的长时间执行过程中，偶发的网络波动或 API 限流通常会被自动恢复，无需手动干预。

### 待办提醒

每次主提示执行结束后，系统会检查当前会话中是否存在未完成的 `todos`。若存在，会自动追加一条 `<system-reminder>` 提醒 Agent 先完成所有 pending / in_progress 的待办事项再结束。该机制与 `SetTodoList` 工具配合，确保长任务不会遗漏中间步骤。

---

## SetTodoList 工具

`SetTodoList` 是用于跟踪任务进度的工具，在执行过程中自动调用。

### 功能概述

参考 `kimi-cli/src/kimi_cli/tools/todo/__init__.py`：

- **读取模式**（`todos` 为 `null`）：返回当前待办列表
- **写入模式**（提供 `todos` 列表）：更新并持久化待办事项

### 数据结构

```python
class Todo:
    title: str           # 待办事项标题
    status: str          # 状态："pending" | "in_progress" | "done"

class Params:
    todos: list[Todo] | None  # 为 null 时读取，提供时写入
```

### 显示效果

CLI 中以可视化卡片形式展示：

```
┌─────────────────────────────────┐
│ [done] 添加参数校验              │
│ [in_progress] 统一异常类型       │
│ [pending] 添加日志记录           │
└─────────────────────────────────┘
```

### 持久化

- **Root Agent**：保存在会话状态的 `state.todos` 中
- **Sub Agent**：保存在子代理目录的 `state.json` 中

---

## Agent Swarm 多智能体协作

Agent Swarm 通过协调者（Coordinator）将复杂任务拆分为有向无环图（DAG），并调度多个子 Agent 并行执行，最后合并结果。本文档介绍如何在 CLI 中使用 `/swarm` 命令调用该能力。

---

### 交互命令

进入 Kimix 交互终端后，执行：

```
/swarm
```

#### 输入任务描述

执行后进入多行输入模式：

- 以 `/end` 结束输入并提交任务
- 以 `/cancel` 取消当前操作
- 空输入会被忽略并提示跳过

**示例**：

```text
/swarm
>>>> Start input multiple-lines for swarm task, end with /end, cancel with /cancel
请为这个项目生成单元测试，覆盖 src/utils.py 和 src/core.py 的核心函数
/end
```

#### 执行流程

`_cmd_swarm` 内部按以下步骤执行（参考 `src/kimix/cli_impl/commands.py`）：

1. **收集提示词**：将多行文本拼接为完整任务描述 `task_prompt`。
2. **创建 Swarm 会话**：调用 `create_swarm_session(task_prompt)`，由协调者 Agent 根据 `agent_swarm.json` 规划 DAG 节点。
3. **执行 DAG**：使用 `Executor().execute(dag)` 按依赖关系调度各节点运行。
4. **输出结果**：执行完成后打印各节点返回结果；若任一阶段失败，打印对应错误信息。

---

### 状态与输出

| 场景 | 输出 |
|------|------|
| 创建 DAG 成功 | `Swarm session created, DAG has N node(s).` |
| DAG 执行完成 | `Swarm execution completed. Results: ...` |
| 空输入 | `Empty task prompt, skipping swarm command.` |
| 创建失败 | `Failed to create swarm session: ...` |
| 执行失败 | `Swarm execution failed: ...` |
| 取消操作 | `Swarm command cancelled.` |

---

### 典型场景

#### 场景 A：批量代码审查

```text
/swarm
请审查 src/ 目录下所有 Python 文件的代码风格，找出潜在的性能瓶颈和安全隐患，并给出修改建议。
/end
```

协调者会为每个文件或模块创建独立节点，并行执行审查任务。

#### 场景 B：多模块重构

```text
/swarm
将项目中的日志模块从 print 替换为标准 logging，涉及 utils.py、core.py 和 cli.py。
/end
```

协调者会按依赖顺序编排节点，先处理底层模块，再处理上层调用方。

---

### 注意事项

1. **耗时较长**：Swarm 任务涉及多次 LLM 调用与 DAG 调度，请耐心等待。
2. **结果合并**：子 Agent 的输出若涉及文件修改，最终会通过 VFS（虚拟文件系统）合并到统一目录；冲突时由协调者或额外会话仲裁。
3. **错误隔离**：单个节点失败不会阻塞其他独立节点，但最终结果中会包含错误信息。

---

## 方案对比

| 特性 | `/plan` | `/swarm` |
|------|---------|----------|
| **执行方式** | 串行执行 | 并行执行（DAG） |
| **任务拆分** | 线性步骤列表 | 有向无环图 |
| **断点续传** | ❌ 不支持（当前实现会在一次流程中完成生成、审阅、执行与复核） | ❌ 不支持 |
| **进度跟踪** | Steps 可视化 | DAG 节点状态 |
| **适用场景** | 步骤明确、顺序依赖的任务 | 可并行、模块独立的任务 |
| **调用开销** | 较低（单会话） | 较高（多 Agent） |
| **典型用例** | 功能实现、代码重构 | 批量审查、多模块分析 |

### 选择建议

- **使用 `/plan`**：任务需要先明确步骤、经用户审阅确认后再执行，适合一次性完成的长任务
- **使用 `/swarm`**：任务可分解为多个相对独立的子任务，希望并行加速执行，或需要多视角协作
