# 第五阶段：有界多 Agent 协作

## 范围

第五阶段把第四阶段的单控制器拆成三个职责分离的组件：`PlannerAgent`、`ExecutorAgent` 和 `ReviewerAgent`。它们由 `MultiAgentCoordinator` 在同一服务进程内按固定顺序协调，不是三个独立进程，也不是三个可以自由讨论、自由选工具的语言模型。

当前多 Agent 只处理两类已有固定模板的模拟任务：

1. 匿名演示课表、开课班和聚合评价联合推荐；
2. 2～4 本模拟二手书各选一本的比价与预算筛选。

RAG 校园问答和单工具查询仍使用原有链路。系统不连接真实学校系统，不处理真实学号，不执行选课、购书、联系卖家或交易，也没有主动提醒、后台定时任务和通知能力。

## 三个角色

### Planner Agent

Planner 只分析当前问题、提取约束，并从两个确定性模板中生成计划。它不调用 MCP 工具，也不根据语言模型输出执行任意步骤。

计划交给 Executor 前会完整校验：任务类型、计划编号、步骤数量与顺序、工具名称、参数类型、匿名演示编号、预算范围及最多 6 次工具调用都必须符合白名单。需要澄清或包含不支持约束时不会启动 Executor。用户问题中的“多 Agent”“Planner、Executor 和 Reviewer 协作”等编排措辞只用于请求展示协作，不会变成课程名或工具参数。

### Executor Agent

Executor 只接收已经校验的计划，并在执行前再次校验。它在计划的私有副本上顺序调用官方 MCP Client 的白名单工具，不能选择新的任务类型、增加工具或改变工具顺序。

执行过程继续使用第四阶段的确定性筛选逻辑，并在内存中保留每次 MCP `ToolEnvelope` 的深拷贝，供 Reviewer 使用。原始证据不会通过网页事件下发；网页仍只收到脱敏参数、状态、数据模式和结果数量。依赖工具失败时停止后续筛选并把后续步骤标记为 `skipped`，不生成部分推荐。

### Reviewer Agent

Reviewer 不调用工具，也不信任 Executor 给出的结论。它读取 Planner 的原始计划、Executor 的调用轨迹和仅存于内存的原始工具证据，独立检查：

- 计划身份、约束和工具顺序是否被改变；
- 调用轨迹与原始证据的工具名、状态、数量和 `data_mode=demo` 是否一致；
- 是否出现禁止进入工作流的敏感字段；
- 课程任务的匿名编号、校区和工具参数是否一致，并重新执行时间、周次、冲突、评分与评价筛选；
- 二手书任务是否每书各选一本、记录编号不重复，并独立枚举候选组合，重算最低总价和预算结论；
- 最终状态、推荐项和正文中的关键事实是否与独立复算一致。

如果工具本身失败，Reviewer 可以确认这是一个“如实失败且没有推荐项”的安全结果；这只代表失败处理通过审计，任务状态仍为 `failed`，不会伪装为成功。

## 失败关闭与结果发布

协调顺序固定为：

```text
Planner -> Executor -> Reviewer -> 最终结果
```

Executor 的候选正文在 Reviewer 完成前不会以 `delta` 或 `task_result` 发送。只有 Reviewer 通过后，协调器才发布经过复核的任务状态和正文。Reviewer 拒绝、异常或缺少执行报告时，协调器返回 `review_failed`，结果数量强制为 0，并用统一停止说明替代候选正文：未经确认的推荐不会进入网页或会话历史。

计划在 Planner 与 Executor 交接前还会计算指纹；交接前若内容发生变化，执行会停止。该机制和 Reviewer 的独立复算共同防止“轨迹看似成功、结论却已被篡改”的情况。

## 流式事件与网页展示

可执行任务的 NDJSON 顺序为：

```text
start
  -> agent_start(planner) -> agent_result(planner)
  -> plan
  -> agent_start(executor)
       -> step_start -> tool_start -> tool_result -> step_result ...
  -> agent_result(executor)
  -> agent_start(reviewer) -> agent_result(reviewer)
  -> task_result -> delta -> done
```

纯本地筛选或汇总步骤没有工具事件。`agent_start` 与 `agent_result` 只包含运行编号、角色标识、展示名称、职责、状态和简短说明；Reviewer 结果额外包含 `approved`。网页以固定的 Planner／Executor／Reviewer 三角色卡展示进行中、完成、失败或拒绝状态。流被中断时，未完成角色显示为已中断，且不保存未发布结果。

需要澄清时只有 Planner 产生角色记录，不启动 Executor 或 Reviewer，也不调用工具。普通 RAG 和单工具请求不伪装成三 Agent 协作。

## 健康检查

`GET /api/health` 的阶段为 `[1,2,3,4,5]`，并公开以下能力描述：

```json
{
  "planning": {
    "configured": true,
    "mode": "bounded-multi-agent",
    "tasks": ["course_recommendation", "book_comparison"],
    "max_tool_calls": 6
  },
  "multi_agent": {
    "configured": true,
    "roles": ["planner", "executor", "reviewer"],
    "reviewer_fail_closed": true
  }
}
```

这表示三角色有界工作流已启用，不表示支持任意任务编排、真实校园操作或主动服务。

## 验收重点

- Planner 不调用工具，Executor 不改计划，Reviewer 不调用工具；
- 非白名单工具、改变后的步骤、参数或证据会被拒绝；
- Reviewer 能发现被篡改的推荐项、总价、工具轨迹或原始证据；
- 审核完成前没有 `task_result` 和答案 `delta`；
- Reviewer 拒绝时不泄露 Executor 的候选正文，且不保存为成功回答；
- 原第四阶段课程推荐、多书比价、失败停止和模拟数据声明保持兼容；
- 主动提醒、真实系统、下单与交易仍明确不支持。
