"use strict";

// No browser, network, or npm dependencies required. Run: node tests/test_web_phase5.js
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");

class Element {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.className = "";
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this.attributes = {};
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.scrollHeight = 30;
    this.ownText = "";
  }

  get textContent() {
    return this.ownText + this.children.map((child) => child.textContent).join("");
  }

  set textContent(value) {
    this.ownText = String(value);
    this.children = [];
  }

  set innerHTML(_value) {
    throw new Error("Dynamic HTML must not be inserted into the UI");
  }

  append(...nodes) {
    nodes.forEach((node) => {
      node.parent = this;
      this.children.push(node);
    });
  }
  appendChild(node) { this.append(node); return node; }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((node) => node !== this);
  }
  replaceChildren(...nodes) { this.children = []; this.ownText = ""; this.append(...nodes); }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  querySelector(selector) {
    return descendants(this).find((node) => selector.startsWith(".")
      && node.className.split(" ").includes(selector.slice(1))) || null;
  }
  scrollTo() {}
  focus() {}
  requestSubmit() { this.listeners.submit({ preventDefault() {} }); }
}

function descendants(node) {
  return node.children.flatMap((child) => [child, ...descendants(child)]);
}

const source = fs.readFileSync(path.resolve(__dirname, "../web/app.js"), "utf8");
const injection = "<img src=x onerror=alert(1)>";
const line = (event) => JSON.stringify(event) + "\n";

const agent = (type, id, status, extra = {}) => line({
  type,
  payload: {
    run_id: "run-1",
    agent_id: id,
    display_name: id === "planner" ? "Planner Agent" : id === "executor" ? "Executor Agent" : "Reviewer Agent",
    role: id === "planner" ? "规划" : id === "executor" ? "执行" : "复核",
    status,
    message: `${id}-${status}`,
    ...extra,
  },
});

function regularPhase5Chunks() {
  return [
    line({ type: "start", session_id: "SESSION-PHASE5" }),
    agent("agent_start", "planner", "running"),
    agent("agent_result", "planner", "completed"),
    line({ type: "plan", plan_id: "p5", title: "推荐通识课", constraints: {}, steps: [{ step_id: "s1", title: "查询与筛选" }] }),
    agent("agent_start", "executor", "running"),
    line({ type: "step_start", plan_id: "p5", step_id: "s1", title: "查询与筛选" }),
    line({ type: "tool_start", call_id: "c1", display_name: "开课查询", arguments: { student_id: "DEMO001" } }),
    line({ type: "tool_result", call_id: "c1", status: "completed", data_mode: "demo", item_count: 2, message: "返回模拟开课" }),
    line({ type: "step_result", plan_id: "p5", step_id: "s1", status: "completed", message: "筛选完成" }),
    agent("agent_result", "executor", "completed"),
    agent("agent_start", "reviewer", "running"),
    agent("agent_result", "reviewer", "completed", { approved: true, message: "证据和约束检查通过" }),
    line({ type: "task_result", plan_id: "p5", status: "completed", item_count: 1, message: "找到 1 门课程" }),
    line({ type: "delta", text: "推荐电影艺术赏析。" }),
    line({ type: "done" }),
  ];
}

async function settle() {
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
}

async function fixture(chunks, observe = () => {}) {
  const ids = [
    "chatForm", "messageInput", "sendButton", "messageList", "welcomePanel", "clearButton",
    "rebuildButton", "healthButton", "modelStatus", "knowledgeStatus", "toolingStatus",
    "agentStatus", "toastRegion",
  ];
  const nodes = Object.fromEntries(ids.map((id) => [id, new Element()]));
  for (const id of ["modelStatus", "knowledgeStatus", "toolingStatus", "agentStatus"]) {
    const value = new Element("strong");
    value.className = "status-value";
    nodes[id].append(value);
  }
  const storage = new Map();
  const find = (className) => descendants(nodes.messageList)
    .filter((node) => node.className.split(" ").includes(className));
  const suggestion = new Element("button");
  suggestion.dataset.question = "用多 Agent 为 DEMO001 推荐通识课";
  const document = {
    createElement: (tag) => new Element(tag),
    getElementById: (id) => nodes[id],
    querySelector: () => new Element(),
    querySelectorAll: () => [suggestion],
  };
  const jsonResponse = (value) => ({
    ok: true,
    headers: { get: () => "application/json" },
    json: async () => value,
  });
  const fetch = async (url) => {
    if (url === "/api/health") return jsonResponse({
      provider: { name: "mock", configured: true },
      knowledge: { documents: 4, chunks: 534 },
      tooling: { configured: true, tool_count: 5 },
      multi_agent: { roles: [
        { agent_id: "planner", display_name: "Planner Agent" },
        { agent_id: "executor", display_name: "Executor Agent" },
        { agent_id: "reviewer", display_name: "Reviewer Agent" },
      ] },
    });
    if (url === "/api/session/clear") return jsonResponse({});
    if (url === "/api/knowledge/rebuild") return jsonResponse({ status: "rebuilt" });
    assert.equal(url, "/api/chat");
    let index = 0;
    return {
      ok: true,
      body: { getReader: () => ({ read: async () => {
        observe(index, find);
        if (index === chunks.length) return { done: true };
        return { done: false, value: new TextEncoder().encode(chunks[index++]) };
      } }) },
    };
  };
  const context = {
    document, fetch, TextDecoder, Uint8Array, AbortController, Intl, Map, JSON, Object, Array,
    Number, String, Boolean, Error, TypeError,
    sessionStorage: {
      getItem: (key) => storage.get(key),
      setItem: (key, value) => storage.set(key, value),
      removeItem: (key) => storage.delete(key),
    },
    window: { setTimeout: () => 1, clearTimeout() {}, requestAnimationFrame: (callback) => callback() },
  };
  vm.runInNewContext(source, context);
  await settle();
  return {
    nodes, storage, find,
    send: async (message) => {
      nodes.messageInput.value = message;
      nodes.chatForm.requestSubmit();
      await settle();
    },
  };
}

async function main() {
  const snapshots = {};
  const regular = await fixture(regularPhase5Chunks(), (index, find) => {
    if (index === 2) snapshots.plannerRunning = find("agent-stage").map((row) => row.dataset.state);
    if (index === 5) snapshots.executorRunning = find("agent-stage").map((row) => row.dataset.state);
    if (index === 11) {
      snapshots.reviewerRunning = find("agent-stage").map((row) => row.dataset.state);
      snapshots.beforeReviewResult = {
      answerHidden: find("message-bubble").at(-1)?.hidden,
      progress: find("agent-card-progress")[0]?.textContent,
      };
    }
    if (index === 13) snapshots.afterTaskBeforeAnswer = find("message-bubble").at(-1)?.hidden;
  });
  await regular.send("用多 Agent 为 DEMO001 推荐通识课");
  assert.deepEqual(snapshots.plannerRunning, ["running", "pending", "pending"]);
  assert.deepEqual(snapshots.executorRunning, ["completed", "running", "pending"]);
  assert.deepEqual(snapshots.reviewerRunning, ["completed", "completed", "running"]);
  assert.deepEqual(snapshots.beforeReviewResult, { answerHidden: true, progress: "协作中 · 2/3" });
  assert.equal(snapshots.afterTaskBeforeAnswer, true);
  assert.equal(regular.find("multi-agent-card")[0].dataset.state, "completed");
  assert.equal(regular.find("agent-card-progress")[0].textContent, "复核通过 · 可以展示结果");
  assert.equal(regular.find("agent-stage").length, 3);
  assert.deepEqual(regular.find("agent-stage").map((row) => row.dataset.agentId), ["planner", "executor", "reviewer"]);
  assert.equal(regular.find("answer-text").at(-1).textContent, "推荐电影艺术赏析。");
  assert.match(regular.nodes.agentStatus.textContent, /3 个角色 · 协作就绪/);
  console.log("PASS: streamed Planner → Executor → Reviewer timing, fixed role order, health status, and post-review result");

  const gatedSnapshots = {};
  const gated = await fixture([
    line({ type: "start", session_id: "SESSION-PHASE5" }),
    agent("agent_start", "planner", "running"), agent("agent_result", "planner", "completed"),
    agent("agent_start", "executor", "running"), agent("agent_result", "executor", "completed"),
    agent("agent_start", "reviewer", "running"),
    line({ type: "task_result", plan_id: "gate", status: "completed", item_count: 1, message: "不应提前出现" }),
    line({ type: "delta", text: injection }),
    agent("agent_result", "reviewer", "completed", { approved: true }),
    line({ type: "done" }),
  ], (index, find) => {
    if (index === 8) gatedSnapshots.beforeApproval = {
      answerHidden: find("message-bubble").at(-1)?.hidden,
      plans: find("execution-plan").length,
    };
    if (index === 9) gatedSnapshots.afterApproval = {
      answerHidden: find("message-bubble").at(-1)?.hidden,
      plans: find("execution-plan").length,
    };
  });
  await gated.send("检查提前结果拦截");
  assert.deepEqual(gatedSnapshots.beforeApproval, { answerHidden: true, plans: 0 });
  assert.deepEqual(gatedSnapshots.afterApproval, { answerHidden: false, plans: 1 });
  assert.equal(gated.find("answer-text").at(-1).textContent, injection);
  assert(!descendants(gated.nodes.messageList).some((node) => node.tagName === "img"));
  console.log("PASS: task result and answer stay hidden until Reviewer approval; dynamic values remain text-only");

  const rejected = await fixture([
    line({ type: "start", session_id: "SESSION-PHASE5" }),
    agent("agent_start", "planner", "running"), agent("agent_result", "planner", "completed"),
    agent("agent_start", "executor", "running"), agent("agent_result", "executor", "completed"),
    agent("agent_start", "reviewer", "running"),
    line({ type: "delta", text: "不合格的提前答案" }),
    agent("agent_result", "reviewer", "completed", { approved: false, message: injection }),
    line({ type: "task_result", plan_id: "reject", status: "completed", item_count: 1, message: "不得展示" }),
    line({ type: "delta", text: "不得展示" }),
    line({ type: "done" }),
  ]);
  await rejected.send("构造复核拒绝");
  assert.equal(rejected.find("multi-agent-card")[0].dataset.state, "rejected");
  assert.equal(rejected.find("agent-card-progress")[0].textContent, "复核未通过");
  assert.equal(rejected.find("agent-stage")[2].dataset.state, "rejected");
  assert.equal(rejected.find("agent-stage-detail")[2].textContent, injection);
  assert.equal(rejected.find("message-bubble").at(-1).hidden, true);
  assert.equal(rejected.find("execution-plan").length, 0);
  assert.equal(rejected.find("error-bubble").length, 0);
  assert(!descendants(rejected.nodes.messageList).some((node) => node.tagName === "img"));
  console.log("PASS: Reviewer rejection is prominent and suppresses both buffered and later user-facing results");

  const interrupted = await fixture([
    line({ type: "start", session_id: "SESSION-PHASE5" }),
    agent("agent_start", "planner", "running"), agent("agent_result", "planner", "completed"),
    agent("agent_start", "executor", "running"),
  ]);
  await interrupted.send("构造中断");
  assert.equal(interrupted.find("multi-agent-card")[0].dataset.state, "failed");
  assert.deepEqual(interrupted.find("agent-stage").map((row) => row.dataset.state), ["completed", "interrupted", "interrupted"]);
  assert.equal(interrupted.find("error-bubble").length, 1);
  console.log("PASS: interrupted multi-Agent trace is retained and unfinished roles are marked uncertain");

  const legacy = await fixture([
    line({ type: "start", session_id: "SESSION-LEGACY" }),
    line({ type: "tool_start", call_id: "legacy", display_name: "二手书查询", arguments: { book_name: "高数" } }),
    line({ type: "tool_result", call_id: "legacy", status: "completed", data_mode: "demo", item_count: 2, message: "找到 2 本" }),
    line({ type: "delta", text: "以下为模拟数据。" }),
    line({ type: "done" }),
  ]);
  await legacy.send("有卖高数吗");
  assert.equal(legacy.find("multi-agent-card").length, 0);
  assert.equal(legacy.find("tool-call-card")[0].dataset.state, "completed");
  assert.equal(legacy.find("answer-text").at(-1).textContent, "以下为模拟数据。");
  console.log("PASS: legacy single-tool stream remains compatible without requiring a Reviewer");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
