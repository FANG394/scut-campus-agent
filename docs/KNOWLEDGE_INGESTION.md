# 知识注入与治理说明

## “知识注入”在本项目中的含义

阶段二不训练或微调模型。知识注入采用 RAG：保存原始资料，解析为文本，切成片段，再由本机 `qwen3-embedding:4b` 生成向量并建立本地索引；用户提问时同时执行向量检索与 BM25/关键词检索，将融合排序后命中的少量片段交给 `qwen3:4b`。这样更新资料无需重新训练，服务内部也能追踪每条答案的来源。

## 最短操作流程

1. 将文件放入 `data/knowledge/source`，可以使用子目录分类。
2. 为资料填写来源、核验和适用范围元数据。
3. 执行 `python -m campus_agent ingest --force`。
4. 执行 `python -m campus_agent search "测试问题"` 检查命中。
5. 启动网页，用真实问法测试流式正文；需要核对来源时，使用终端 `search` 或 `ask` 查看内部来源信息。

服务启动时也会比较指纹并自动重建变化过的索引；显式执行 `ingest --force` 更适合发布前验收。
嵌入模型名称和向量格式也属于索引指纹；更换嵌入模型后旧索引不会被复用。

## 支持格式

| 格式 | 默认支持 | 说明 |
| --- | --- | --- |
| `.md` / `.markdown` | 是 | 推荐格式，可在开头直接写元数据 |
| `.txt` | 是 | 自动尝试 UTF-8 和 GB18030 |
| `.html` / `.htm` | 是 | 忽略脚本、样式、SVG 等不可见内容 |
| `.docx` | 是 | 使用标准库提取正文段落；可附加同名 `.docx.ocr.txt` 图片文字转录缓存 |
| `.pdf` | 可选 | 安装 `requirements-pdf.txt` 后读取文本型 PDF；也可提供同名 `.pdf.txt` 文本缓存 |

扫描 PDF 没有文本层时不会自动 OCR，而会在入库报告中给出警告。若提供 `文件名.pdf.txt`，加载器会优先使用该文本缓存，并把原 PDF 与缓存共同纳入语料指纹；缓存本身不会被重复当成第二份文档。

DOCX 中的图片不会被普通正文解析器读取。若人工复核或本地 OCR 后提供 `文件名.docx.ocr.txt`，加载器会把它追加到该 DOCX 的正文，并把缓存纳入语料指纹；缓存仍属于同一份文档，权威级别和核验状态继承原 DOCX，不会因转录而提升。

## 推荐元数据

Markdown 可以使用简化 front matter：

```markdown
---
title: 本科生补办学生证流程
source_url: https://example.scut.edu.cn/official-page
source_owner: 华南理工大学某主管部门
published_at: 2026-08-01
updated_at: 2026-08-20
verified_at: 2026-08-28
authority: official
verification_status: verified-summary
volatility: high
audience: 全日制本科生
campus_scope: 五山校区
auth_required: false
tags: 学生证, 补办, 本科生
---

# 本科生补办学生证流程

这里放经过核对的正文或摘要。
```

非 Markdown 文件可以增加同名旁路 JSON，例如 `notice.pdf.meta.json`：

```json
{
  "title": "通知标题",
  "source_url": "https://example.scut.edu.cn/notice",
  "source_owner": "主管部门",
  "authority": "official",
  "verification_status": "verified",
  "verified_at": "2026-08-28",
  "volatility": "high",
  "auth_required": false,
  "tags": ["通知", "本科生"]
}
```

若旁路 JSON 和 Markdown front matter 同时存在，front matter 优先。

## 字段建议

- `authority`：建议使用 `official`、`college`、`community`、`unknown`。
- `verification_status`：建议使用 `verified`、`verified-summary`、`source-provided`、`pending-review`、`unverified`。其中 `source-provided` 只证明本地原文件已收到并可解析，不等同于官网核验。
- `volatility`：建议使用 `low`、`medium`、`high`、`very_high`。开放时间、费用、资格和联系方式通常至少为 `high`。
- `campus_scope`：资料只适用于某校区时必须填写。
- `audience`：标明本科生、研究生、教职工等适用对象。
- `auth_required`：原始页面是否需要登录。不要把密码、Cookie、Token 或个人账号写入任何字段。

## 推荐治理流程

```text
收集候选资料 -> 核对原始来源 -> 填写元数据 -> 复核摘要
      -> 入库与检索测试 -> 发布 -> 按时效等级定期复核/下架
```

- 官方公开资料：可以作为事实回答的默认来源。
- 学院资料：需标记学院、校区和人群，避免推广到全校。
- 学生经验或论坛内容：建议单独标为非官方，不用于规章、费用、资格等确定性回答。
- 需登录资料：在明确授权、访问控制和第三方模型数据政策前，不应直接导入当前 Demo。
- 过期资料：更新正文和 `verified_at`，或移出源目录后重建索引。

## 公开仓库与本地资料

公开仓库不附带本地原始资料、派生文本缓存或生成索引。克隆后请只导入有权使用和处理的资料，并通过同名 `.meta.json` 记录来源属性、适用范围和核验状态；本地资料清单模板见 `KNOWLEDGE_SOURCES.md`。

开发验收曾使用 4 份本地资料生成 534 个、2560 维的检索片段，但这些数字只是当时的本地历史结果，不代表公开仓库内置内容。`source-provided` 只表示文件由用户提供并已成功解析，不表示已与学校官网逐条核验；`pending-review` 状态与可靠性告警保留在服务内部结果中。网页按当前产品要求只显示正文，不展示来源或告警，资料治理时仍必须按待核验内容处理并以学校最新正式通知为准。

## 需要项目方确认的问题

1. 是否坚持“只有校级官方公开资料可直接发布”，还是允许学院资料与用户上传资料？
2. 谁负责审核，`high` 和 `very_high` 资料多久复核一次？
3. 是否需要 OCR；若需要，允许本地 OCR 还是云端 OCR？
4. 启用远程 LLM 后，哪些知识片段允许发送给外部供应商？

在这些规则确定前，推荐维持当前的保守策略：官方公开资料优先、来源可追溯、证据不足即拒答。
