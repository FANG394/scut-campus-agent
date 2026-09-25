"use strict";

// No browser, network, or npm dependencies required. Run: node tests/test_web_phase4.js
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

  appendChild(node) {
    this.append(node);
    return node;
  }

  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((node) => node !== this);
  }

  replaceChildren(...nodes) {
    this.children = [];
    this.ownText = "";
    this.append(...nodes);
  }

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
const plan = {
  type: "plan", plan_id: "p1", title: injection,
  constraints: { book_names: ["高数", "线代"], total_budget: 50 },
  steps: [{ step_id: "s1", title: "检索高数" }, { step_id: "s2", title: "检索线代" }],
};

function taskChunks(status) {
  const failed = status === "failed";
  const task = {
    plan_id: "p1", status, item_count: status === "completed" ? 3 : 0,
    message: status === "completed" ? injection : status === "no_result" ? "没有满足总预算的组合。" : "查询失败，无法生成方案。",
  };
  return [
    line({ type: "start", session_id: "SESSION-DEMO" }), line(plan),
    line({ type: "step_start", plan_id: "p1", step_id: "s1", title: "检索高数" }),
    line({ type: "tool_start", call_id: "c1", display_name: "二手书查询", arguments: { book_name: injection } }),
    line({ type: "tool_result", call_id: "c1", status: failed ? "failed" : "completed", data_mode: "demo", item_count: 2, message: injection }),
    line({ type: "step_result", plan_id: "p1", step_id: "s1", status: failed ? "failed" : "completed", message: "高数查询返回" }),
    line({ type: "step_start", plan_id: "p1", step_id: "s2", title: "检索线代" }),
    line({ type: "tool_start", call_id: "c2", display_name: "二手书查询", arguments: { book_name: "线代" } }),
    line({ type: "tool_result", call_id: "c2", status: "completed", data_mode: "demo", item_count: 3, message: "线代查询返回" }),
    line({ type: "step_result", plan_id: "p1", step_id: "s2", status: "completed", message: "线代查询返回" }),
    // Exercise both direct and payload-wrapped event contracts.
    line(status === "no_result" ? { type: "task_result", payload: task } : { type: "task_result", ...task }),
    line({ type: "delta", text: "临时回答" }), line({ type: "replace", text: injection }),
    JSON.stringify({ type: "done" }),
  ];
}

async function settle() {
  await new Promise((resolve) => setImmediate(resolve));
}

async function fixture(chunks, observe = () => {}) {
  const ids = ["chatForm", "messageInput", "sendButton", "messageList", "welcomePanel", "clearButton", "rebuildButton", "healthButton", "modelStatus", "knowledgeStatus", "toolingStatus", "toastRegion"];
  const nodes = Object.fromEntries(ids.map((id) => [id, new Element()]));
  for (const id of ["modelStatus", "knowledgeStatus", "toolingStatus"]) {
    const value = new Element("strong");
    value.className = "status-value";
    nodes[id].append(value);
  }
  const storage = new Map();
  const calls = [];
  const suggestion = new Element("button");
  suggestion.dataset.question = "比较高数和线代二手书，总预算50元";
  const find = (className) => descendants(nodes.messageList)
    .filter((node) => node.className.split(" ").includes(className));
  const document = {
    createElement: (tag) => new Element(tag), getElementById: (id) => nodes[id],
    querySelector: () => new Element(), querySelectorAll: () => [suggestion],
  };
  const response = (value) => ({ ok: true, headers: { get: () => "application/json" }, json: async () => value });
  const fetch = async (url, options = {}) => {
    calls.push({ url, body: options.body ? JSON.parse(options.body) : null });
    if (url === "/api/health") return response({
      provider: { name: "mock", configured: true }, knowledge: { documents: 4, chunks: 534 },
      tooling: { configured: true, tool_count: 5 },
    });
    if (url === "/api/session/clear") return response({});
    if (url === "/api/knowledge/rebuild") return response({ status: "rebuilt" });
    assert.equal(url, "/api/chat");
    let index = 0;
    return { ok: true, body: { getReader: () => ({ read: async () => {
      observe(index, find);
      if (index === chunks.length) return { done: true };
      return { done: false, value: new TextEncoder().encode(chunks[index++]) };
    } }) } };
  };
  const context = {
    document, fetch, TextDecoder, Uint8Array, AbortController, Intl, Map, JSON, Object, Array,
    Number, String, Boolean, Error, TypeError,
    sessionStorage: {
      getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value),
      removeItem: (key) => storage.delete(key),
    },
    window: { setTimeout: () => 1, clearTimeout() {}, requestAnimationFrame: (callback) => callback() },
  };
  vm.runInNewContext(source, context);
  await settle();
  return { nodes, storage, calls, find, suggestion, send: async (message) => {
    nodes.messageInput.value = message;
    nodes.chatForm.requestSubmit();
    await settle();
  } };
}

async function main() {
  const snapshots = {};
  const ui = await fixture(taskChunks("completed"), (index, find) => {
    if (index === 2) snapshots.plan = { count: find("execution-plan").length, state: find("plan-step")[0]?.dataset.state, answerHidden: find("message-bubble").at(-1)?.hidden };
    if (index === 3) snapshots.running = find("plan-step")[0]?.dataset.state;
    if (index === 8) snapshots.multitool = { count: find("tool-call-card").length, state: find("tool-call-card")[1]?.dataset.state };
    if (index === 10) snapshots.beforeTaskResult = find("plan-progress")[0]?.textContent;
    if (index === 11) snapshots.task = find("execution-plan")[0]?.dataset.state;
  });
  await ui.send("比较高数和线代二手书，总预算50元");
  assert.deepEqual(snapshots.plan, { count: 1, state: "pending", answerHidden: true });
  assert.equal(snapshots.running, "running");
  assert.deepEqual(snapshots.multitool, { count: 2, state: "running" });
  assert.equal(snapshots.beforeTaskResult, "步骤完成 · 等待汇总");
  assert.equal(snapshots.task, "completed");
  assert.equal(ui.find("plan-progress")[0].textContent, "任务完成 · 3 项结果");
  assert.equal(ui.find("plan-task-summary")[0].textContent, injection);
  assert.equal(ui.find("answer-text").at(-1).textContent, injection);
  assert.equal(ui.find("message-bubble").at(-1).hidden, false);
  assert(!descendants(ui.nodes.messageList).some((node) => node.tagName === "img"));
  assert.equal(ui.storage.get("scut-agent-session-id"), "SESSION-DEMO");
  assert.match(ui.nodes.toolingStatus.textContent, /5 个工具/);
  console.log("PASS: immediate streamed plan, step states, multi-tool, task completion, delta/replace, text-only XSS, 5-tool health");

  const empty = await fixture(taskChunks("no_result"));
  await empty.send("总预算太低的买书任务");
  assert(empty.find("plan-step").every((step) => step.dataset.state === "completed"));
  assert.equal(empty.find("execution-plan")[0].dataset.state, "no_result");
  assert.equal(empty.find("plan-progress")[0].textContent, "无匹配结果");
  assert.equal(empty.find("plan-task-summary")[0].textContent, "没有满足总预算的组合。");
  console.log("PASS: completed steps are not confused with successful task results; no_result clearly displayed");

  const failed = await fixture(taskChunks("failed"));
  await failed.send("工具失败的任务");
  assert.equal(failed.find("execution-plan")[0].dataset.state, "failed");
  assert.equal(failed.find("plan-progress")[0].textContent, "任务失败");
  assert.equal(failed.find("plan-task-summary")[0].textContent, "查询失败，无法生成方案。");
  console.log("PASS: failed task status and explanation");

  const localizedChunks = taskChunks("completed");
  localizedChunks[1] = line({ ...plan, constraints: {
    student_id: "DEMO001", demo_student_defaulted: false, course_query: "通识", min_rating: 4.5,
    no_attendance: true, low_workload: false, limit: 3, time_mode: "within", weekday: "周二", period: "下午",
    time_range: ["14:00", "16:00"], time_interpretation: "仅筛选指定时段。",
    book_names: ["高数", "线代"], total_budget: 50, price_limit: null, selection: "各选一本",
    resolved_campus: "大学城校区",
  } });
  const localized = await fixture(localizedChunks);
  await localized.send("完整条件展示");
  const constraintText = localized.find("plan-constraints")[0].textContent;
  for (const expected of ["演示编号默认：否", "课程检索：通识", "最低评分：4.5/5", "点名要求：明确不点名", "工作量：未限制",
    "备选数：3", "时间筛选：指定时段", "时段：下午", "时间范围：14:00–16:00", "时间含义：仅筛选指定时段。",
    "书名列表：高数、线代", "总预算：50 元", "单本预算：未限制", "选择规则：各选一本", "确认校区：大学城校区"]) {
    assert(constraintText.includes(expected), `Missing localized constraint: ${expected}`);
  }
  assert(!/demo_student_defaulted|course_query|no_attendance|low_workload|time_mode|time_interpretation|book_names|price_limit|true|false|within/.test(constraintText));
  localizedChunks[1] = line({ ...plan, constraints: { demo_student_defaulted: true, no_attendance: false, low_workload: true, time_mode: "avoid", price_limit: 30 } });
  const avoid = await fixture(localizedChunks);
  await avoid.send("排除时段展示");
  assert.match(avoid.find("plan-constraints")[0].textContent, /演示编号默认：是.*点名要求：未限制.*工作量：低工作量.*时间筛选：排除时段.*单本预算：30 元/);
  console.log("PASS: complete Chinese constraint labels, boolean requirements, within/avoid, time ranges and budgets");

  const broken = await fixture([
    line({ type: "start", session_id: "SESSION-DEMO" }), line(plan),
    line({ type: "step_start", plan_id: "p1", step_id: "s1" }),
    line({ type: "tool_start", call_id: "broken", display_name: "测试工具" }),
  ]);
  await broken.send("中断任务");
  assert.equal(broken.find("execution-plan").length, 1);
  assert.equal(broken.find("execution-plan")[0].dataset.state, "failed");
  assert(broken.find("plan-step").every((step) => step.dataset.state === "interrupted"));
  assert.equal(broken.find("tool-call-card")[0].dataset.state, "failed");
  assert.equal(broken.find("error-bubble").length, 1);
  console.log("PASS: interrupted execution records are retained and marked uncertain");

  ui.nodes.clearButton.listeners.click(); await settle();
  assert(ui.calls.some((call) => call.url === "/api/session/clear"));
  assert.equal(ui.nodes.messageList.children.length, 0);
  assert.equal(ui.nodes.welcomePanel.hidden, false);
  assert.equal(ui.storage.size, 0);
  ui.nodes.rebuildButton.listeners.click(); await settle();
  assert(ui.calls.some((call) => call.url === "/api/knowledge/rebuild"));
  assert.equal(ui.nodes.rebuildButton.disabled, false);
  console.log("PASS: clear session and rebuild knowledge remain operational");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
