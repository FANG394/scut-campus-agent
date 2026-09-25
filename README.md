# 华工校园助手（阶段一至五）

2026-09-25 验收：项目 Python 3.14 下 156 项回归全部通过（0 失败、0 错误、0 跳过），网页阶段四／五事件测试共 11 组通过，JavaScript 语法检查通过；实际浏览器已验证三个角色按顺序运行、Reviewer 通过前不发布任务正文。当前进度和使用边界见 [PROGRESS.md](docs/PROGRESS.md)。

这是一个面向华南理工大学师生的本地校园智能体 Demo。当前覆盖执行方案中的前五个阶段：

- 阶段一：网页与终端对话入口、Prompt、基础问答、本机 ChatOllama 接入。
- 阶段二：校园文档解析、本地向量索引、RAG 检索、来源追踪、证据不足时拒绝猜测。
- 阶段三：官方 MCP Python SDK v2、校园工具、单工具意图路由、Agent 主动调用及网页工具轨迹。
- 阶段四：有边界的任务拆分与顺序多工具执行，支持课程时间／评价联合推荐，以及 2～4 本二手书各一本的比价与预算筛选。
- 阶段五：Planner／Executor／Reviewer 三角色协作；Reviewer 根据内存中的原始工具证据独立复算，审核通过后才发布答案。

二手书、课表、课程评价和开课班当前均为明确标注的模拟数据，不连接真实学校系统。第五阶段把固定、可校验的任务模板拆成同一服务进程内的三个职责组件，最多调用 6 次白名单工具；它不是三个自由对话模型，不执行模型自由生成的任意计划。主动提醒、真实校务接入、选课、下单和交易尚未实现。

二手书演示库现包含 100 条模拟记录。Agent 可识别“有卖、有人卖、在售、出售、转卖、求购、想买、哪里能买到”等常见交易问法，也支持“高数、线代、概统、大物、计网、计组、模电、数电、马原、毛概”等常见教材简称。以上能力只用于查询本地演示 JSON，不代表真实库存、真实价格或实际成交信息。

## 快速运行

环境要求：Python 3.11 或更新版本、已启动的 Ollama，以及本机已下载
`qwen3:4b` 和 `qwen3-embedding:4b`。

下面的依赖安装、索引重建和服务启动命令必须使用同一个 Python 解释器。
如果电脑上安装了多个 Python，请始终用同一个解释器的完整路径替换示例中的
`python`；否则可能把 `requirements.txt` 安装到一个环境，却用另一个环境启动服务。

在克隆后的项目根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m campus_agent check
python -m campus_agent ingest --force
python -m campus_agent serve
```

浏览器打开 `http://127.0.0.1:8000`。按 `Ctrl+C` 停止服务。

也可以直接在终端使用：

```powershell
python -m campus_agent ask "公共自习室允许占座吗？"
python -m campus_agent search "体测毕业要求"
python -m campus_agent tools-list
python -m campus_agent tool-call secondhand_book_search --arguments '{"book_name":"高等数学","price_limit":30}'
python -m campus_agent chat
```

二手书连续问答示例：

```text
有卖概率论二手书吗？
有卖概率论吗？
有卖概率论与数理统计吗？
```

三问都应调用二手书工具，并命中演示记录 `BOOK-DEMO-005`。系统只保存最近一次成功工具的名称，不保存上一轮书名、价格等工具参数；最近工具上下文最多覆盖接下来的 3 个追问机会，因此“那高数呢？”可以沿用二手书查询领域，但仍会从当前问题重新提取“高数”。遇到“有卖二手书吗？”、“多少钱？”等缺少明确书名的问题时，系统会先要求补充书名，不会调用语言模型猜测库存或价格。

如需把工具服务接到外部 MCP Host，可使用 stdio 命令：

```powershell
python -m campus_agent mcp-serve
```

该命令的标准输出只用于 MCP 协议，不要在同一管道打印调试文字。完整工具参数与 Host 配置示例见 `docs/MCP_TOOLS.md`。

第五阶段多 Agent 对话示例：

```text
推荐一门 DEMO001 周二下午评分高且不点名的通识课
推荐 DEMO002 周二下午评分至少4.5分的通识课
推荐 DEMO001 避开周二下午、评分高的通识课
比较《高等数学》和《线性代数》，总预算50元，每本最多30元
```

课程任务依次读取完整匿名课表、候选开课班和聚合评价，再校验时间、周次、学期、校区和评分。未指定演示编号时会明确声明使用 `DEMO001`，不会声称掌握用户真实课表。“周二下午没课”解释为寻找该时段内可选且不冲突的课程，不代表整段下午空闲；明确写“避开周二下午”则排除该时段。缺少时间、没有可匹配评价、存在冲突的班次不作为成功推荐；“不点名”要求结构化字段明确为 `none`，“偶尔点名”也会排除。“评分高”默认最低 4.5/5。任务约束详见 `docs/PHASE4.md`，三角色职责与审核边界见 `docs/PHASE5.md`。

当前演示数据为 100 条二手书、2 个匿名学生共 12 条课表安排、8 门聚合课程评价和 12 个开课班；课程评价与开课班按 `course_id` 连接，不把“评价好”直接当作“开课且无冲突”。

## 运行测试

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```

自动化测试使用确定性的模型替身，可以离线复现；正常启动服务时则会调用上述本机
Ollama 模型。

## 模型配置

默认使用本机 Ollama：`ChatOllama(model="qwen3:4b")` 负责回答，
`OllamaEmbeddings(model="qwen3-embedding:4b")` 负责文档与问题向量化，不需要 API Key。

通常无需创建 `.env`；启动 Ollama 后可直接运行。需要修改本机地址、模型或批大小时，
再将 `.env.example` 复制为 `.env`。每次更换嵌入模型后必须重新运行
`python -m campus_agent ingest --force`。

默认配置：

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
```

OpenAI Responses API 示例：

```dotenv
LLM_PROVIDER=openai
LLM_API_STYLE=responses
LLM_API_KEY=你的密钥
LLM_MODEL=你的模型名
LLM_BASE_URL=https://api.openai.com/v1
```

项目也支持 OpenAI-compatible Chat Completions。外部地址必须使用 HTTPS；只有 `localhost` 或回环 IP 可以使用 HTTP。接入远程模型后，命中的知识片段会随问题发送给该模型供应商，因此上线前需要确认数据分类与供应商合规要求。

网页聊天直接消费 `ChatOllama.stream()` 的原生输出，`POST /api/chat` 以 NDJSON
发送 `start`、若干 `delta` 和 `done` 事件。单工具请求会在回答前增加
`tool_start` 与 `tool_result`，其中只含脱敏参数、状态和结果数量，不下发完整工具数据。RAG 回答在流结束后执行证据校验；
如果模型回答引用无效或缺少实质支持，服务会发送 `replace`，让网页用本地证据摘录
整体替换已显示的临时文本。RAG 流中的 `delta` 是尚未完成证据校验的临时草稿，不应当作已核验结论；只有校验后的最终回答写入会话，未结束或被中断的草稿不写入历史。网页不展示内部引用编号、来源对象或告警。多 Agent 任务在回答前发送 Planner、Executor、Reviewer 的 `agent_start/agent_result`，并穿插 `plan`、步骤和脱敏工具事件；Reviewer 完成复核前不会发送 `task_result` 或答案 `delta`，工具原始数据也不会下发到网页。

## 知识库

本地开发目录使用过 4 份用户提供的原始文档进行验证，并生成过 534 个、`qwen3-embedding:4b` 2560 维的检索片段。公开仓库不会提交这些原始文档、文本缓存或生成索引，以免公开文档作者元数据、内部资料或尚未确认再分发许可的材料；克隆后需要自行把有权使用的资料放入 `data/knowledge/source` 并重建索引。因此，刚克隆的公开版本在完成知识注入前不会回答依赖这些资料的校园事实问题。

检索阶段同时使用向量相似度、BM25 和关键词覆盖率：向量检索负责同义表达，BM25/关键词负责召回包含明确事项名称的条文，融合后再执行去重与数量限制。服务内部保留本地文件、权威级别、核验状态和日期；网页聊天只展示回答正文，不显示引用、来源卡或可靠性告警。未注明正式发布单位的资料应在元数据中标记为待核验，不能视为校方正式规定。

新增知识不是“训练模型”，而是将资料放入 `data/knowledge/source` 后用本地嵌入模型重建向量索引：

```powershell
python -m campus_agent ingest --force
```

支持 Markdown、TXT、HTML、DOCX；文本型 PDF 需要可选依赖：

```powershell
python -m pip install -r requirements-pdf.txt
```

扫描 PDF 暂不自动 OCR。本项目为当前学生手册保存了与原 PDF 同步生成的 `.pdf.txt` 文本缓存，使未安装 `pypdf` 的运行环境仍可检索；新增 PDF 若没有缓存，仍需安装可选依赖。`体测.docx` 中的两张图片评分表另有逐格人工转录的 `.docx.ocr.txt` 检索缓存，仍沿用原文档的待核验状态。完整元数据格式、审核建议和更新流程见 `docs/KNOWLEDGE_INGESTION.md`。

## 目录结构

```text
campus_agent/              对话、模型适配、会话、安全与 HTTP 服务
campus_agent/rag/          文档加载、切分、索引和检索
campus_agent/mcp/          MCP Server、Client、统一结果结构和工具数据访问
campus_agent/tool_router.py 单工具意图识别与参数提取
campus_agent/planning.py    固定模板规划、顺序执行、课程冲突校验与多书比价
campus_agent/multi_agent.py 三角色职责、计划交接校验、证据复算与失败关闭
data/knowledge/source/     本地知识源目录（公开仓库不提交实际语料）
data/knowledge/index/      自动生成的本地索引
data/tools/                二手书、匿名课表、课程评价和开课班模拟 JSON
web/                       零外部依赖网页界面
tests/                     自动化测试
docs/                      架构、知识注入和验收说明
```

## 安全与可靠性边界

- 校园事实优先走知识库；没有足够证据就明确拒答。
- 校园事实回答在服务内部必须逐段引用对应证据；引用无效或与证据缺少实质支持时，网页通过 `replace` 事件自动切换为本地证据摘要。网页展示层会移除内部引用标记。
- 检测到密码、验证码、身份证号、学号/账号或明确标注的手机号后，该消息不会保存、检索或发送给外部模型。
- 课表工具仅接受 `DEMO001`、`DEMO002` 等匿名演示编号；疑似真实长学号即使没有“学号”标签也会在调用前拦截。
- 所有 MCP 工具返回统一的 `ok/tool/data_mode/items/message/warnings` 结构；Agent 只允许调用五个白名单工具。
- 二手书查询只读取 100 条本地模拟记录；结果始终带有模拟数据提示，不能据此判断真实商品是否在售或实际成交价格。
- 最近一次成功工具的名称可为接下来的最多 3 个追问机会提供领域提示；工具参数不会跨轮复用，缺少明确书名时直接澄清，不让语言模型猜测库存或价格。
- 多 Agent 任务只执行课程推荐和多书比价模板。Planner 只能生成白名单计划，Executor 只能执行该计划，Reviewer 根据内存原始证据独立复算；依赖工具失败时停止后续筛选，审核拒绝时不发布候选答案。
- 主动提醒、后台任务和真实系统操作仍明确拒绝执行。
- `.env` 和生成索引不纳入版本控制；模型请求关闭服务端存储参数（支持该参数的 Responses API）。
- 会话只存在内存，服务重启后自动清空；本版本不接入需登录的校内系统。

## 尚需项目方确认

1. 品牌名称：执行指令写的是“华工助手”，目录名是“鲤工助手”。当前界面按执行指令使用“华工校园助手”。
2. 知识治理：建议校园事实默认只发布学校官方公开资料；学院资料需标明适用范围，个人上传内容先进入待审核区。
3. 私有资料与 OCR：是否允许导入需登录的内部文件、是否需要扫描件 OCR，需要在上线前明确。

详细设计见 `docs/ARCHITECTURE.md`，MCP 工具说明见 `docs/MCP_TOOLS.md`，任务约束见 `docs/PHASE4.md`，多 Agent 设计见 `docs/PHASE5.md`，验收清单见 `docs/ACCEPTANCE.md`。
