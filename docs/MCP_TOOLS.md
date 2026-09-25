# 第三至五阶段 MCP 工具说明

## 实现范围

项目使用官方 `mcp>=2,<3` Python SDK。`campus_agent.mcp.server.create_mcp_server()` 创建真实 `MCPServer`，Agent 通过 SDK 的进程内 `Client(server)` 调用；外部 MCP Host 可通过 `python -m campus_agent mcp-serve` 使用 stdio 连接。同一套工具函数服务于两种连接方式。

第三阶段保留一个意图对应一个工具的路由；第四阶段增加有界的课程推荐和多书比价模板。第五阶段把复杂任务拆成 Planner、Executor、Reviewer 三个固定角色：Planner 生成并校验白名单计划，Executor 按计划调用工具，Reviewer 不调用工具，而是根据内存中的完整返回证据独立复算并决定是否发布答案。当前共五个工具，不支持任意自由组合。课程推荐不是只靠评价工具判断是否开课，而是读取完整匿名课表、开课班和聚合评价三类证据。

## 统一返回格式

五个工具都返回以下 JSON 对象：

```json
{
  "ok": true,
  "tool": "secondhand_book_search",
  "data_mode": "demo",
  "items": [],
  "message": "面向用户和模型的简短说明",
  "warnings": ["模拟数据提示或可靠性提示"]
}
```

- `data_mode=demo`：来自 `data/tools` 的模拟数据，回答必须明确说明“模拟数据”。
- `data_mode=local_knowledge_base`：来自第二阶段本地知识库，并保留来源元数据。
- `items` 给应用读取；MCP SDK 同时生成文本内容和 `structuredContent`。
- `ok=false` 表示参数或数据层错误。一次参数有效但没有匹配项的查询仍可返回 `ok=true`、空 `items` 和明确的未命中说明；Agent 不得据此扩写或猜测真实库存、价格。

## 工具清单

### `campus_knowledge_search`

输入：`query: str`，可选 `limit: int=4`。调用现有混合 RAG 检索，不绕过知识库置信度门控。

### `secondhand_book_search`

输入：`book_name: str`，可选 `price_limit: float`。按书名、版本、作者和分类搜索，随后执行最高价格过滤并按价格排序。结果只有匿名卖家别名和公共交接位置，不含联系方式。

当前仓储包含 100 条本地模拟二手书记录，并支持常见教材简称，例如“高数、线代、概统、大物、大英、计网、计组、模电、数电、自控、马原、毛概、近代史、思修”。简称只影响本地检索归一化，不会把数据转换为真实库存；返回仍必须带 `data_mode=demo` 和模拟数据警示。

### `student_schedule_query`

输入：`student_id: str="DEMO001"`，可选 `weekday: str`。只接受 `DEMO001`、`DEMO002` 等匿名演示编号；绝不接受或保存真实学号。

### `course_review_search`

输入：`course_name: str`。按课程名或标签查询，返回模拟聚合评分、评价数量、工作量、点名与考核特点，不包含具体学生评价或个人信息。当前有 8 门评价；第四阶段使用 `course_id` 与开课班连接，使用结构化 `attendance_requirement` 判断不点名，只有 `none` 符合该硬约束，`optional`、`required` 或未知值不能视作不点名。

### `course_offering_search`

输入：`course_name: str="通识"`，可选 `campus: str`。按课程名或分类检索模拟开课班并可按校区过滤，最多返回 30 项。接受“五山／五山校区”“大学城／大学城校区”“广州国际／广州国际校区”等校区值。

当前有 12 个模拟开课班。每项包含 `offering_id`、`course_id`、课程名、分类、`semester`、校区、地点和 `meetings`；每个上课时间包含星期、`HH:MM` 起止时间和闭区间 `week_start/week_end`。`meetings=[]` 表示尚未公布时间，工具附带警告，不能据此推断无冲突。整个开课数据集在过滤前进行结构和唯一编号校验，损坏记录不会被静默当成未命中。

该工具只查询，不执行选课、占位或退课。第四阶段必须结合相同学期的完整课表和可匹配评价，校验全部上课时间及周次后才能推荐。详见 `PHASE4.md`。

## Agent 二手书路由与防猜测

MCP 工具本身接收结构化参数；自然语言问法由 Agent 的 `ToolRouter` 处理。目前覆盖“有卖、有没有卖、有人卖、出售、转卖、转让、在售、有货、出一本、卖一本、收一本、求购、想买、哪里能买到、能否买到”等常见交易表达，并清理“吗、呢、呀”等句末语气词。无法得到有效书名时不会以“吗”“教材”等无效值发起调用。

会话只保存最近一次成功工具的名称，并最多为接下来的 3 个追问机会提供领域提示；不会保存或复用工具参数。例如查询二手书成功后，“那高数呢？”可继续使用 `secondhand_book_search`，但书名必须从当前问题重新提取。当前消息出现明确的课表、课程评价或校园知识意图时，当前意图优先。

“有卖二手书吗？”或上下文中的“多少钱？”缺少明确书名，Agent 会直接请求补充信息，不调用 MCP，也不让语言模型猜测库存或价格。即使工具返回未命中，也只能转述该次本地模拟查询的结果，不能推断真实校园市场中不存在该书。

## 本地调试

```powershell
python -m campus_agent tools-list
python -m campus_agent tool-call secondhand_book_search --arguments '{"book_name":"高等数学","price_limit":30}'
python -m campus_agent tool-call student_schedule_query --arguments '{"student_id":"DEMO001","weekday":"周二"}'
python -m campus_agent tool-call course_review_search --arguments '{"course_name":"电影艺术赏析"}'
python -m campus_agent tool-call course_offering_search --arguments '{"course_name":"通识","campus":"五山校区"}'
```

外部 Host 的通用配置形态如下，Python 路径应替换为实际安装依赖的同一个解释器：

```json
{
  "mcpServers": {
    "scut-campus-tools": {
      "command": "C:\\Path\\To\\python.exe",
      "args": ["-m", "campus_agent", "mcp-serve"],
      "cwd": "D:\\鲤工助手"
    }
  }
}
```

## 替换为真实数据时的接口

当前 JSON 是只读演示仓储。后续可保持五个工具签名和统一 envelope 不变，只替换 `CampusToolRepository` 的数据访问实现，例如数据库或经授权的校内 API。真实课程推荐还需要稳定的课程／班次编号、同一学期、完整上课周次与时间、评价口径和更新时效；缺失字段不能简单填零或默认“无冲突”。接入真实系统前至少需要确认：身份认证、最小权限、访问审计、字段脱敏、数据保留期限、失败重试和学校授权。不要把 Cookie、Token、密码或真实学生数据写入 JSON 文件。
