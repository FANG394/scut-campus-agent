(function () {
  "use strict";

  const SESSION_KEY = "scut-agent-session-id";
  let requestTimeoutMs = 60_000;

  const elements = {
    chatMain: document.querySelector(".chat-main"),
    chatForm: document.getElementById("chatForm"),
    messageInput: document.getElementById("messageInput"),
    sendButton: document.getElementById("sendButton"),
    messageList: document.getElementById("messageList"),
    welcomePanel: document.getElementById("welcomePanel"),
    clearButton: document.getElementById("clearButton"),
    rebuildButton: document.getElementById("rebuildButton"),
    healthButton: document.getElementById("healthButton"),
    modelStatus: document.getElementById("modelStatus"),
    knowledgeStatus: document.getElementById("knowledgeStatus"),
    toolingStatus: document.getElementById("toolingStatus"),
    agentStatus: document.getElementById("agentStatus"),
    toastRegion: document.getElementById("toastRegion"),
  };

  let sessionId = sessionStorage.getItem(SESSION_KEY) || "";
  let isSending = false;

  function createElement(tagName, className, text) {
    const node = document.createElement(tagName);
    if (className) node.className = className;
    if (typeof text === "string") node.textContent = text;
    return node;
  }

  function formatTime(date = new Date()) {
    return new Intl.DateTimeFormat("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(date);
  }

  function compactText(value, fallback = "") {
    return typeof value === "string" && value.trim() ? value.trim() : fallback;
  }

  function showToast(message, kind = "default") {
    const toast = createElement("div", "toast", message);
    toast.dataset.kind = kind;
    elements.toastRegion.appendChild(toast);
    window.setTimeout(() => toast.remove(), 3600);
  }

  async function apiRequest(path, options = {}) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), requestTimeoutMs);

    try {
      const response = await fetch(path, {
        ...options,
        headers: {
          Accept: "application/json",
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...(options.headers || {}),
        },
        signal: controller.signal,
      });

      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json")
        ? await response.json()
        : { detail: await response.text() };

      if (!response.ok) {
        const detail = payload.detail || payload.message || `请求失败（HTTP ${response.status}）`;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }

      return payload;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("请求超时，请确认本地服务是否正常运行。");
      }
      if (error instanceof TypeError) {
        throw new Error("无法连接本地服务，请先启动后端后再试。");
      }
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }

  async function streamChat(
    body,
    onStart,
    onDelta,
    onReplace,
    onToolStart,
    onToolResult,
    onPlan,
    onStepStart,
    onStepResult,
    onTaskResult,
    onAgentStart,
    onAgentResult
  ) {
    const controller = new AbortController();
    let timer = 0;
    const armTimeout = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => controller.abort(), requestTimeoutMs);
    };
    armTimeout();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          Accept: "application/x-ndjson",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        const contentType = response.headers.get("content-type") || "";
        const payload = contentType.includes("application/json")
          ? await response.json()
          : { detail: await response.text() };
        throw new Error(payload.detail || `请求失败（HTTP ${response.status}）`);
      }
      if (!response.body) throw new Error("浏览器不支持读取流式回答。");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let completed = false;

      const consumeLine = (line) => {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === "start") onStart(event);
        else if (event.type === "delta" && typeof event.text === "string") {
          onDelta(event.text);
        }
        else if (event.type === "replace" && typeof event.text === "string") {
          onReplace(event.text);
        }
        else if (event.type === "tool_start") onToolStart(event);
        else if (event.type === "tool_result") onToolResult(event);
        else if (event.type === "plan") onPlan(event);
        else if (event.type === "step_start") onStepStart(event);
        else if (event.type === "step_result") onStepResult(event);
        else if (event.type === "task_result") onTaskResult(event);
        else if (event.type === "agent_start") onAgentStart(event);
        else if (event.type === "agent_result") onAgentResult(event);
        else if (event.type === "done") completed = true;
      };

      while (true) {
        const { value, done } = await reader.read();
        armTimeout();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        let newline = buffer.indexOf("\n");
        while (newline !== -1) {
          consumeLine(buffer.slice(0, newline));
          buffer = buffer.slice(newline + 1);
          newline = buffer.indexOf("\n");
        }
        if (done) break;
      }
      if (buffer.trim()) consumeLine(buffer);
      if (!completed) throw new Error("流式回答意外中断，请重试。");
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("请求超时，请确认本地服务是否正常运行。");
      }
      if (error instanceof TypeError) {
        throw new Error("无法连接本地服务，请先启动后端后再试。");
      }
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }

  function setStatus(element, state, value, title = "") {
    element.dataset.state = state;
    const valueNode = element.querySelector(".status-value");
    valueNode.textContent = value;
    element.title = title || value;
  }

  async function refreshHealth({ quiet = false } = {}) {
    setStatus(elements.modelStatus, "loading", "连接中…");
    setStatus(elements.knowledgeStatus, "loading", "读取中…");
    setStatus(elements.toolingStatus, "loading", "读取中…");
    if (elements.agentStatus) setStatus(elements.agentStatus, "loading", "读取中…");
    elements.healthButton.disabled = true;

    try {
      const health = await apiRequest("/api/health");
      const advertisedTimeout = Number(health.request_timeout_seconds);
      if (Number.isFinite(advertisedTimeout) && advertisedTimeout > 0) {
        requestTimeoutMs = Math.min(310_000, Math.max(10_000, advertisedTimeout * 1000));
      }
      const provider = health.provider || {};
      const knowledge = health.knowledge || {};
      const tooling = health.tooling || {};
      const multiAgent = health.multi_agent || {};
      const providerName = compactText(provider.name, "未命名模型");

      if (provider.configured) {
        setStatus(elements.modelStatus, "ok", providerName, `模型 ${providerName} 已配置`);
      } else {
        setStatus(
          elements.modelStatus,
          "warning",
          `${providerName} · 演示模式`,
          "模型尚未配置，当前可能使用本地规则应答"
        );
      }

      const documentCount = Number(knowledge.documents) || 0;
      const chunkCount = Number(knowledge.chunks) || 0;
      const knowledgeWarnings = Array.isArray(knowledge.warnings) ? knowledge.warnings : [];
      const knowledgeState = documentCount > 0 && chunkCount > 0
        ? knowledgeWarnings.length ? "warning" : "ok"
        : "warning";
      const knowledgeText = documentCount > 0
        ? `${documentCount} 篇资料 · ${chunkCount} 个片段`
        : "暂无可检索资料";
      setStatus(
        elements.knowledgeStatus,
        knowledgeState,
        knowledgeText,
        knowledgeWarnings.join("；") || knowledgeText
      );

      const toolCount = Number(tooling.tool_count) || 0;
      const toolingReady = Boolean(tooling.configured) && toolCount > 0;
      setStatus(
        elements.toolingStatus,
        toolingReady ? "ok" : "warning",
        toolingReady ? `${toolCount} 个工具 · 模拟数据` : "依赖未就绪",
        compactText(tooling.message, "MCP 工具状态不可用")
      );

      if (elements.agentStatus) {
        const roles = Array.isArray(multiAgent.roles) ? multiAgent.roles : [];
        const roleNames = roles.map((role) => typeof role === "string"
          ? role
          : compactText(role && (role.display_name || role.agent_id))).filter(Boolean);
        const agentsReady = roleNames.length >= 3;
        setStatus(
          elements.agentStatus,
          agentsReady ? "ok" : "warning",
          agentsReady ? `${roleNames.length} 个角色 · 协作就绪` : "协作状态未就绪",
          roleNames.length ? `多 Agent：${roleNames.join("、")}` : "未收到多 Agent 角色信息"
        );
      }

      if (!quiet && knowledgeWarnings.length) {
        showToast(knowledgeWarnings[0], "default");
      }
    } catch (error) {
      setStatus(elements.modelStatus, "error", "服务未连接", error.message);
      setStatus(elements.knowledgeStatus, "error", "状态不可用", error.message);
      setStatus(elements.toolingStatus, "error", "状态不可用", error.message);
      if (elements.agentStatus) setStatus(elements.agentStatus, "error", "状态不可用", error.message);
      if (!quiet) showToast(error.message, "error");
    } finally {
      elements.healthButton.disabled = false;
    }
  }

  function scrollToLatest() {
    window.requestAnimationFrame(() => {
      elements.chatMain.scrollTo({ top: elements.chatMain.scrollHeight, behavior: "smooth" });
    });
  }

  function createMessageShell(role, label) {
    const article = createElement("article", `message message-${role}`);
    const avatar = createElement("div", "avatar", role === "user" ? "我" : "AI");
    avatar.setAttribute("aria-hidden", "true");

    const content = createElement("div", "message-content");
    const heading = createElement("div", "message-heading");
    heading.append(
      createElement("strong", "", label),
      createElement("span", "", formatTime())
    );
    content.appendChild(heading);
    article.append(avatar, content);
    return { article, content };
  }

  function appendUserMessage(text) {
    const { article, content } = createMessageShell("user", "你");
    const bubble = createElement("div", "message-bubble");
    bubble.appendChild(createElement("p", "answer-text", text));
    content.appendChild(bubble);
    elements.messageList.appendChild(article);
    return article;
  }

  function appendLoadingMessage() {
    const { article, content } = createMessageShell("assistant", "华工校园助手");
    article.dataset.loading = "true";
    const bubble = createElement("div", "message-bubble loading-bubble");
    bubble.setAttribute("aria-label", "正在查找资料并生成回答");
    const dots = createElement("div", "loading-dots");
    dots.append(createElement("span"), createElement("span"), createElement("span"));
    bubble.appendChild(dots);
    content.appendChild(bubble);
    elements.messageList.appendChild(article);
    return article;
  }

  function appendAssistantMessage() {
    const { article, content } = createMessageShell("assistant", "华工校园助手");
    const agentTrace = createElement("div", "agent-trace");
    agentTrace.hidden = true;
    const planTrace = createElement("div", "plan-trace");
    planTrace.hidden = true;
    const toolTrace = createElement("div", "tool-trace");
    toolTrace.hidden = true;
    const bubble = createElement("div", "message-bubble");
    bubble.hidden = true;
    const answerText = createElement("p", "answer-text", "");
    bubble.appendChild(answerText);

    content.append(agentTrace, planTrace, toolTrace, bubble);
    elements.messageList.appendChild(article);
    return {
      article,
      answerText,
      bubble,
      agentTrace,
      agentCard: null,
      agents: new Map(),
      reviewRequired: false,
      reviewState: "not_required",
      deferredAnswer: "",
      deferredHasAnswer: false,
      deferredTaskResults: [],
      planTrace,
      plans: new Map(),
      toolTrace,
      toolCards: new Map(),
    };
  }

  const agentDefinitions = [
    { id: "planner", displayName: "Planner Agent", role: "规划" },
    { id: "executor", displayName: "Executor Agent", role: "执行" },
    { id: "reviewer", displayName: "Reviewer Agent", role: "复核" },
  ];

  const agentStates = {
    pending: { label: "等待", indicator: "·" },
    running: { label: "工作中", indicator: "↻" },
    completed: { label: "已完成", indicator: "✓" },
    failed: { label: "失败", indicator: "!" },
    rejected: { label: "复核未通过", indicator: "!" },
    interrupted: { label: "连接中断", indicator: "!" },
  };

  function ensureAgentCard(message) {
    if (message.agentCard) return message.agentCard;
    const card = createElement("section", "multi-agent-card");
    card.dataset.state = "running";
    card.setAttribute("aria-label", "多 Agent 协作");
    const heading = createElement("div", "agent-card-heading");
    const title = createElement("strong", "agent-card-title", "多 Agent 协作");
    const progress = createElement("span", "agent-card-progress", "准备协作");
    heading.append(title, progress);
    const list = createElement("ol", "agent-stage-list");
    agentDefinitions.forEach((definition, index) => {
      const row = createElement("li", "agent-stage");
      row.dataset.agentId = definition.id;
      row.dataset.state = "pending";
      const indicator = createElement("span", "agent-stage-indicator", String(index + 1));
      indicator.setAttribute("aria-hidden", "true");
      const copy = createElement("div", "agent-stage-copy");
      const rowHeading = createElement("div", "agent-stage-heading");
      const identity = createElement("span", "agent-stage-identity");
      const name = createElement("strong", "agent-stage-name", definition.displayName);
      const role = createElement("span", "agent-stage-role", definition.role);
      identity.append(name, role);
      const status = createElement("span", "agent-stage-status", agentStates.pending.label);
      rowHeading.append(identity, status);
      const detail = createElement("p", "agent-stage-detail", "等待上一角色完成");
      copy.append(rowHeading, detail);
      row.append(indicator, copy);
      list.appendChild(row);
      message.agents.set(definition.id, { row, indicator, name, role, status, detail });
    });
    card.append(heading, list);
    message.agentTrace.appendChild(card);
    message.agentTrace.hidden = false;
    message.agentCard = { card, progress };
    return message.agentCard;
  }

  function refreshAgentCard(message) {
    if (!message.agentCard) return;
    const entries = [...message.agents.values()];
    const states = entries.map((entry) => entry.row.dataset.state);
    const completed = states.filter((state) => state === "completed").length;
    if (states.includes("rejected")) {
      message.agentCard.card.dataset.state = "rejected";
      message.agentCard.progress.textContent = "复核未通过";
    } else if (states.some((state) => ["failed", "interrupted"].includes(state))) {
      message.agentCard.card.dataset.state = "failed";
      message.agentCard.progress.textContent = "协作失败";
    } else if (completed === entries.length && message.reviewState === "approved") {
      message.agentCard.card.dataset.state = "completed";
      message.agentCard.progress.textContent = "复核通过 · 可以展示结果";
    } else {
      message.agentCard.card.dataset.state = "running";
      message.agentCard.progress.textContent = `协作中 · ${completed}/${entries.length}`;
    }
  }

  function updateAgentEntry(entry, status, event) {
    const state = Object.prototype.hasOwnProperty.call(agentStates, status) ? status : "failed";
    entry.row.dataset.state = state;
    entry.status.textContent = agentStates[state].label;
    entry.indicator.textContent = agentStates[state].indicator;
    const displayName = compactText(event.display_name);
    const role = compactText(event.role);
    if (displayName) entry.name.textContent = displayName;
    if (role) entry.role.textContent = role;
    const fallback = state === "running" ? "正在处理当前阶段…"
      : state === "completed" ? "当前阶段已完成。"
        : state === "rejected" ? "Reviewer 未批准本次结果，结果不会展示。"
          : state === "interrupted" ? "未收到完整结果，请重试。" : "当前阶段执行失败。";
    entry.detail.textContent = compactText(event.message, fallback);
  }

  function flushReviewedResults(message) {
    if (message.reviewState !== "approved") return;
    message.deferredTaskResults.splice(0).forEach((event) => showTaskResult(message, event));
    if (message.deferredHasAnswer) {
      message.bubble.hidden = false;
      message.answerText.textContent = message.deferredAnswer;
      message.deferredAnswer = "";
      message.deferredHasAnswer = false;
    }
  }

  function showAgentEvent(message, event, starting) {
    const payload = eventPayload(event);
    const agentId = compactText(payload.agent_id);
    if (!agentDefinitions.some((definition) => definition.id === agentId)) return;
    ensureAgentCard(message);
    message.reviewRequired = true;
    if (message.reviewState === "not_required") message.reviewState = "pending";
    const entry = message.agents.get(agentId);
    let state = starting ? "running" : compactText(payload.status, "completed");
    if (agentId === "reviewer" && !starting && (payload.approved === false || state === "failed")) {
      state = "rejected";
      message.reviewState = "rejected";
      message.deferredAnswer = "";
      message.deferredHasAnswer = false;
      message.deferredTaskResults = [];
    } else if (agentId === "reviewer" && !starting && state === "completed") {
      message.reviewState = "approved";
    }
    updateAgentEntry(entry, state, payload);
    refreshAgentCard(message);
    flushReviewedResults(message);
  }

  const stepStates = {
    pending: { label: "等待执行", indicator: "·" },
    running: { label: "进行中", indicator: "↻" },
    completed: { label: "已完成", indicator: "✓" },
    failed: { label: "失败", indicator: "!" },
    skipped: { label: "已跳过", indicator: "–" },
    blocked: { label: "已中止", indicator: "!" },
    interrupted: { label: "连接中断", indicator: "!" },
  };

  function eventPayload(event) {
    return event.payload && typeof event.payload === "object" ? event.payload : event;
  }

  function formatConstraints(constraints) {
    if (!constraints || typeof constraints !== "object") return "";
    const labels = {
      student_id: "演示编号", demo_student_defaulted: "演示编号默认",
      course_query: "课程检索", no_attendance: "点名要求", low_workload: "工作量",
      limit: "备选数", weekday: "星期", time_window: "时间范围", time_range: "时间范围",
      time_mode: "时间筛选", period: "时段", time_interpretation: "时间含义",
      min_rating: "最低评分", total_budget: "总预算", price_limit: "单本预算", max_price: "最高单价",
      book_names: "书名列表", selection: "选择规则", campus: "校区", resolved_campus: "确认校区", category: "类别",
    };
    return Object.entries(constraints)
      .filter(([key, value]) => value !== undefined && value !== ""
        && (value !== null || ["total_budget", "price_limit", "max_price"].includes(key)))
      .map(([key, value]) => {
        let display;
        if (value === null) display = "未限制";
        else if (key === "no_attendance" && typeof value === "boolean") display = value ? "明确不点名" : "未限制";
        else if (key === "low_workload" && typeof value === "boolean") display = value ? "低工作量" : "未限制";
        else if (key === "time_mode" && ["within", "avoid"].includes(value)) display = value === "within" ? "指定时段" : "排除时段";
        else if (typeof value === "boolean") display = value ? "是" : "否";
        else if (Array.isArray(value)) display = value.join(key === "time_range" ? "–" : "、");
        else if (["total_budget", "price_limit", "max_price"].includes(key) && typeof value === "number") display = `${value} 元`;
        else if (key === "min_rating" && typeof value === "number") display = `${value}/5`;
        else display = typeof value === "object" ? JSON.stringify(value) : String(value);
        return `${labels[key] || key}：${display}`;
      })
      .join(" · ");
  }

  function refreshPlanState(plan) {
    const entries = [...plan.steps.values()];
    const complete = entries.filter((step) => step.row.dataset.state === "completed").length;
    const hasFailure = entries.some((step) => ["failed", "blocked", "interrupted"].includes(step.row.dataset.state));
    const finished = entries.length > 0 && entries.every((step) => ["completed", "skipped"].includes(step.row.dataset.state));
    if (plan.result) {
      plan.card.dataset.state = plan.result.status;
      plan.progress.textContent = plan.result.status === "completed" ? `任务完成 · ${plan.result.itemCount} 项结果`
        : plan.result.status === "no_result" ? "无匹配结果" : "任务失败";
    } else {
      plan.card.dataset.state = hasFailure ? "failed" : "running";
      plan.progress.textContent = hasFailure ? `执行未完成 · ${complete}/${entries.length}`
        : finished ? "步骤完成 · 等待汇总" : `执行中 · ${complete}/${entries.length}`;
    }
  }

  function updatePlanStep(step, status, message = "") {
    const state = Object.prototype.hasOwnProperty.call(stepStates, status) ? status : "pending";
    step.row.dataset.state = state;
    step.status.textContent = stepStates[state].label;
    step.indicator.textContent = stepStates[state].indicator;
    if (message) step.detail.textContent = message;
    step.detail.hidden = !step.detail.textContent;
  }

  function addPlanStep(plan, definition) {
    const stepId = compactText(definition.step_id, `step-${plan.steps.size + 1}`);
    if (plan.steps.has(stepId)) return plan.steps.get(stepId);
    const row = createElement("li", "plan-step");
    const indicator = createElement("span", "plan-step-indicator");
    indicator.setAttribute("aria-hidden", "true");
    const copy = createElement("div", "plan-step-copy");
    const heading = createElement("div", "plan-step-heading");
    const title = createElement("strong", "", compactText(definition.title, "执行步骤"));
    const status = createElement("span", "plan-step-status");
    heading.append(title, status);
    const detail = createElement("p", "plan-step-detail");
    copy.append(heading, detail);
    row.append(indicator, copy);
    plan.list.appendChild(row);
    const step = { row, title, status, indicator, detail };
    plan.steps.set(stepId, step);
    updatePlanStep(step, definition.status || "pending", compactText(definition.message));
    return step;
  }

  function showPlan(message, event) {
    const payload = eventPayload(event);
    const definition = payload.plan && typeof payload.plan === "object" ? payload.plan : payload;
    const planId = compactText(definition.plan_id, "plan-main");
    let plan = message.plans.get(planId);
    if (!plan) {
      const card = createElement("section", "execution-plan");
      card.setAttribute("aria-label", "任务执行计划");
      const heading = createElement("div", "plan-heading");
      const title = createElement("strong", "plan-title");
      const progress = createElement("span", "plan-progress");
      heading.append(title, progress);
      const constraints = createElement("p", "plan-constraints");
      const list = createElement("ol", "plan-step-list");
      const summary = createElement("p", "plan-task-summary");
      summary.hidden = true;
      card.append(heading, constraints, list, summary);
      message.planTrace.appendChild(card);
      plan = { card, title, progress, constraints, list, summary, result: null, steps: new Map() };
      message.plans.set(planId, plan);
    }
    plan.title.textContent = compactText(definition.title, "任务执行计划");
    plan.constraints.textContent = formatConstraints(definition.constraints);
    plan.constraints.hidden = !plan.constraints.textContent;
    if (Array.isArray(definition.steps)) definition.steps.forEach((step) => {
      if (step && typeof step === "object") addPlanStep(plan, step);
    });
    message.planTrace.hidden = false;
    refreshPlanState(plan);
    return plan;
  }

  function showStepEvent(message, event, starting) {
    const payload = eventPayload(event);
    const fallbackId = message.plans.size === 1 ? message.plans.keys().next().value : "plan-main";
    const planId = compactText(payload.plan_id, fallbackId);
    const plan = message.plans.get(planId) || showPlan(message, { plan_id: planId });
    const step = addPlanStep(plan, payload);
    if (compactText(payload.title)) step.title.textContent = payload.title;
    updatePlanStep(step, starting ? "running" : compactText(payload.status, "completed"), compactText(payload.message));
    refreshPlanState(plan);
  }

  function showTaskResult(message, event) {
    const payload = eventPayload(event);
    const fallbackId = message.plans.size === 1 ? message.plans.keys().next().value : "plan-main";
    const planId = compactText(payload.plan_id, fallbackId);
    const plan = message.plans.get(planId) || showPlan(message, { plan_id: planId });
    const status = ["completed", "no_result", "failed"].includes(payload.status) ? payload.status : "failed";
    const count = Number(payload.item_count);
    plan.result = { status, itemCount: Number.isFinite(count) && count >= 0 ? Math.floor(count) : 0 };
    const fallback = status === "completed" ? "任务执行完成。"
      : status === "no_result" ? "已完成查询，但没有符合全部条件的结果。" : "本次任务未能完成，请查看步骤提示后重试。";
    plan.summary.textContent = compactText(payload.message, fallback);
    plan.summary.hidden = false;
    refreshPlanState(plan);
  }

  function showReviewedTaskResult(message, event) {
    if (message.reviewRequired && message.reviewState === "pending") {
      message.deferredTaskResults.push(event);
      return;
    }
    if (message.reviewState === "rejected") return;
    showTaskResult(message, event);
  }

  function showAnswerDelta(message, text) {
    if (message.reviewRequired && message.reviewState === "pending") {
      message.deferredAnswer += text;
      message.deferredHasAnswer = true;
      return;
    }
    if (message.reviewState === "rejected") return;
    message.bubble.hidden = false;
    message.answerText.textContent += text;
  }

  function showAnswerReplace(message, text) {
    if (message.reviewRequired && message.reviewState === "pending") {
      message.deferredAnswer = text;
      message.deferredHasAnswer = true;
      return;
    }
    if (message.reviewState === "rejected") return;
    message.bubble.hidden = false;
    message.answerText.textContent = text;
  }

  function markTraceInterrupted(message) {
    if (message.agentCard) {
      message.agents.forEach((entry) => {
        if (["pending", "running"].includes(entry.row.dataset.state)) {
          updateAgentEntry(entry, "interrupted", { message: "未收到完整协作结果，请重试。" });
        }
      });
      message.reviewState = "interrupted";
      message.deferredAnswer = "";
      message.deferredHasAnswer = false;
      message.deferredTaskResults = [];
      refreshAgentCard(message);
    }
    message.plans.forEach((plan) => {
      plan.steps.forEach((step) => {
        if (["pending", "running"].includes(step.row.dataset.state)) {
          updatePlanStep(step, "interrupted", "未收到完整执行结果，请重试。");
        }
      });
      refreshPlanState(plan);
    });
    message.toolCards.forEach((entry) => {
      if (entry.card.dataset.state !== "running") return;
      entry.card.dataset.state = "failed";
      entry.state.textContent = "连接中断";
      entry.indicator.textContent = "!";
      entry.result.textContent = "未收到工具返回，不能确认本次调用结果。";
    });
  }

  function formatToolArguments(argumentsValue) {
    if (!argumentsValue || typeof argumentsValue !== "object") return "";
    return Object.entries(argumentsValue)
      .map(([key, value]) => `${key}=${String(value)}`)
      .join(" · ");
  }

  function showToolStart(message, event) {
    const callId = compactText(event.call_id, `tool-${message.toolCards.size + 1}`);
    const card = createElement("div", "tool-call-card");
    card.dataset.state = "running";
    const heading = createElement("div", "tool-call-heading");
    heading.append(
      createElement("span", "tool-call-indicator", "↻"),
      createElement("strong", "", compactText(event.display_name, "校园工具")),
      createElement("span", "tool-call-state", "正在调用")
    );
    const args = createElement(
      "p",
      "tool-call-arguments",
      formatToolArguments(event.arguments) || "无公开参数"
    );
    const result = createElement("p", "tool-call-result", "等待工具返回…");
    card.append(heading, args, result);
    message.toolTrace.hidden = false;
    message.toolTrace.appendChild(card);
    message.toolCards.set(callId, { card, result, state: heading.querySelector(".tool-call-state"), indicator: heading.querySelector(".tool-call-indicator") });
  }

  function showToolResult(message, event) {
    const callId = compactText(event.call_id);
    const entry = message.toolCards.get(callId);
    if (!entry) return;
    const succeeded = event.status === "completed";
    entry.card.dataset.state = succeeded ? "completed" : "failed";
    entry.state.textContent = succeeded ? "调用完成" : "调用失败";
    entry.indicator.textContent = succeeded ? "✓" : "!";
    const count = Number(event.item_count) || 0;
    const mode = event.data_mode === "demo" ? "模拟数据" : "本地知识库";
    entry.result.textContent = `${mode} · ${count} 条结果 · ${compactText(event.message, "已返回")}`;
  }

  function appendErrorMessage(error, originalMessage) {
    const { article, content } = createMessageShell("assistant", "连接提示");
    const bubble = createElement("div", "message-bubble error-bubble");
    bubble.append(
      createElement("p", "error-title", "这次没有成功收到回答"),
      createElement("p", "answer-text", error.message || "本地服务暂时不可用，请稍后重试。")
    );
    const retry = createElement("button", "retry-button", "重新发送");
    retry.type = "button";
    retry.addEventListener("click", () => {
      article.remove();
      submitMessage(originalMessage);
    });
    bubble.appendChild(retry);
    content.appendChild(bubble);
    elements.messageList.appendChild(article);
  }

  function setSending(value) {
    isSending = value;
    elements.messageInput.disabled = value;
    elements.sendButton.disabled = value || !elements.messageInput.value.trim();
    elements.sendButton.setAttribute("aria-label", value ? "正在发送" : "发送消息");
  }

  function updateSendButton() {
    elements.sendButton.disabled = isSending || !elements.messageInput.value.trim();
  }

  function resizeTextarea() {
    elements.messageInput.style.height = "auto";
    elements.messageInput.style.height = `${Math.min(elements.messageInput.scrollHeight, 150)}px`;
  }

  async function submitMessage(rawMessage) {
    const message = compactText(rawMessage);
    if (!message || isSending) return;

    elements.welcomePanel.hidden = true;
    appendUserMessage(message);
    elements.messageInput.value = "";
    resizeTextarea();
    setSending(true);
    const loading = appendLoadingMessage();
    let streamedMessage = null;
    const ensureStreamedMessage = () => {
      if (!streamedMessage) {
        loading.remove();
        streamedMessage = appendAssistantMessage();
      }
      return streamedMessage;
    };
    scrollToLatest();

    try {
      const body = { message };
      if (sessionId) body.session_id = sessionId;
      await streamChat(
        body,
        (event) => {
          if (compactText(event.session_id)) {
            sessionId = event.session_id;
            sessionStorage.setItem(SESSION_KEY, sessionId);
          }
        },
        (text) => {
          const current = ensureStreamedMessage();
          showAnswerDelta(current, text);
          scrollToLatest();
        },
        (text) => {
          const current = ensureStreamedMessage();
          showAnswerReplace(current, text);
          scrollToLatest();
        },
        (event) => {
          showToolStart(ensureStreamedMessage(), event);
          scrollToLatest();
        },
        (event) => {
          showToolResult(ensureStreamedMessage(), event);
          scrollToLatest();
        },
        (event) => {
          showPlan(ensureStreamedMessage(), event);
          scrollToLatest();
        },
        (event) => {
          showStepEvent(ensureStreamedMessage(), event, true);
          scrollToLatest();
        },
        (event) => {
          showStepEvent(ensureStreamedMessage(), event, false);
          scrollToLatest();
        },
        (event) => {
          showReviewedTaskResult(ensureStreamedMessage(), event);
          scrollToLatest();
        },
        (event) => {
          showAgentEvent(ensureStreamedMessage(), event, true);
          scrollToLatest();
        },
        (event) => {
          showAgentEvent(ensureStreamedMessage(), event, false);
          scrollToLatest();
        }
      );
      if (!streamedMessage || (!compactText(streamedMessage.answerText.textContent)
        && streamedMessage.reviewState !== "rejected")) {
        throw new Error("服务返回了空回答，请重试。");
      }
    } catch (error) {
      loading.remove();
      if (streamedMessage) {
        if (streamedMessage.agentCard || streamedMessage.plans.size || streamedMessage.toolCards.size) markTraceInterrupted(streamedMessage);
        else streamedMessage.article.remove();
      }
      appendErrorMessage(error, message);
    } finally {
      setSending(false);
      elements.messageInput.focus();
      scrollToLatest();
    }
  }

  async function clearSession() {
    if (isSending) {
      showToast("请等待当前回答完成后再清空会话。", "default");
      return;
    }

    elements.clearButton.disabled = true;
    try {
      if (sessionId) {
        await apiRequest("/api/session/clear", {
          method: "POST",
          body: JSON.stringify({ session_id: sessionId }),
        });
      }
      sessionId = "";
      sessionStorage.removeItem(SESSION_KEY);
      elements.messageList.replaceChildren();
      elements.welcomePanel.hidden = false;
      elements.chatMain.scrollTo({ top: 0, behavior: "smooth" });
      showToast("会话已清空。", "success");
    } catch (error) {
      showToast(`清空失败：${error.message}`, "error");
    } finally {
      elements.clearButton.disabled = false;
      elements.messageInput.focus();
    }
  }

  async function rebuildKnowledge() {
    if (elements.rebuildButton.dataset.busy === "true") return;
    elements.rebuildButton.dataset.busy = "true";
    elements.rebuildButton.disabled = true;
    setStatus(elements.knowledgeStatus, "loading", "正在重建…");

    try {
      const result = await apiRequest("/api/knowledge/rebuild", {
        method: "POST",
        body: "{}",
      });
      const status = compactText(result.status, "已完成");
      showToast(`知识库重建：${status}`, "success");
      await refreshHealth({ quiet: true });
    } catch (error) {
      setStatus(elements.knowledgeStatus, "error", "重建失败", error.message);
      showToast(`知识库重建失败：${error.message}`, "error");
    } finally {
      elements.rebuildButton.dataset.busy = "false";
      elements.rebuildButton.disabled = false;
    }
  }

  elements.chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitMessage(elements.messageInput.value);
  });

  elements.messageInput.addEventListener("input", () => {
    resizeTextarea();
    updateSendButton();
  });

  elements.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      elements.chatForm.requestSubmit();
    }
  });

  document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => submitMessage(button.dataset.question));
  });

  elements.clearButton.addEventListener("click", clearSession);
  elements.rebuildButton.addEventListener("click", rebuildKnowledge);
  elements.healthButton.addEventListener("click", () => refreshHealth());

  updateSendButton();
  refreshHealth({ quiet: true });
})();
