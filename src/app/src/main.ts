// main.ts — Main application logic mirroring _sse_cli_main from sse_cli.py

import { KimixClient } from "./client";
import { renderMessagePart, fmtArg, fmtTs } from "./renderer";
import { MessagePartType } from "./types";
import type { Message, Session } from "./types";

// ── Application State ──────────────────────────────────────────────

let client: KimixClient | null = null;
let session: Session | null = null;
let pollTimer: ReturnType<typeof setInterval> | null = null;
let seenMessageCount = 0;
let emptyPolls = 0;
let connected = false;
let debugMode = false;

// ── DOM Elements ──────────────────────────────────────────────────

const hostInput = document.getElementById("host") as HTMLInputElement;
const portInput = document.getElementById("port") as HTMLInputElement;
const connectBtn = document.getElementById("connect-btn") as HTMLButtonElement;
const disconnectBtn = document.getElementById("disconnect-btn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLElement;
const sessionIdEl = document.getElementById("session-id") as HTMLElement;
const outputEl = document.getElementById("output") as HTMLElement;
const promptInput = document.getElementById("prompt") as HTMLInputElement;
const sendBtn = document.getElementById("send-btn") as HTMLButtonElement;
const debugCheck = document.getElementById("debug-check") as HTMLInputElement;

// Command buttons
const btnNew = document.getElementById("btn-new") as HTMLButtonElement;
const btnAbort = document.getElementById("btn-abort") as HTMLButtonElement;
const btnStatus = document.getElementById("btn-status") as HTMLButtonElement;
const btnSessions = document.getElementById("btn-sessions") as HTMLButtonElement;
const btnMessages = document.getElementById("btn-messages") as HTMLButtonElement;
const btnClear = document.getElementById("btn-clear") as HTMLButtonElement;
const btnCompact = document.getElementById("btn-compact") as HTMLButtonElement;
const btnExport = document.getElementById("btn-export") as HTMLButtonElement;

// ── Output Helpers ────────────────────────────────────────────────

function log(text: string, cls: string = ""): void {
  const line = document.createElement("div");
  line.className = `log-line ${cls}`;
  line.textContent = text;
  outputEl.appendChild(line);
  scrollToBottom();
}

function logHtml(html: string, cls: string = ""): void {
  const line = document.createElement("div");
  line.className = `log-line ${cls}`;
  line.innerHTML = html;
  outputEl.appendChild(line);
  scrollToBottom();
}

function appendPart(partEl: HTMLElement): void {
  // Append inline to the last log line if it's a text part, otherwise new line
  const lastLine = outputEl.lastElementChild;
  if (
    lastLine &&
    partEl.classList.contains("part-text") &&
    lastLine.classList.contains("has-inline")
  ) {
    lastLine.appendChild(partEl);
  } else if (partEl.classList.contains("part-text")) {
    const wrapper = document.createElement("div");
    wrapper.className = "log-line has-inline";
    wrapper.appendChild(partEl);
    outputEl.appendChild(wrapper);
  } else {
    const wrapper = document.createElement("div");
    wrapper.className = "log-line";
    wrapper.appendChild(partEl);
    outputEl.appendChild(wrapper);
  }
  scrollToBottom();
}

function scrollToBottom(): void {
  outputEl.scrollTop = outputEl.scrollHeight;
}

function setStatus(text: string, ok: boolean = true): void {
  statusEl.textContent = text;
  statusEl.className = ok ? "status-ok" : "status-error";
}

function setConnected(isConnected: boolean): void {
  connected = isConnected;
  hostInput.disabled = isConnected;
  portInput.disabled = isConnected;
  connectBtn.disabled = isConnected;
  disconnectBtn.disabled = !isConnected;
  promptInput.disabled = !isConnected;
  sendBtn.disabled = !isConnected;
  debugCheck.disabled = isConnected;

  for (const btn of [
    btnNew,
    btnAbort,
    btnStatus,
    btnSessions,
    btnMessages,
    btnClear,
    btnCompact,
    btnExport,
  ]) {
    (btn as HTMLButtonElement).disabled = !isConnected;
  }
}

// ── API Actions ───────────────────────────────────────────────────

async function doConnect(): Promise<void> {
  const host = hostInput.value.trim() || "127.0.0.1";
  const port = parseInt(portInput.value.trim(), 10) || 4096;

  debugMode = debugCheck.checked;
  client = new KimixClient(host, port);

  log(`[SSE CLI] Connecting to http://${host}:${port}...`, "info");

  try {
    const healthy = await client.healthCheck();
    if (!healthy) {
      log(`[SSE CLI] Server not healthy at http://${host}:${port}`, "error");
      client = null;
      return;
    }

    session = await client.createSession("SSE Web debug session");
    log(`[SSE CLI] Created session: ${session.id}`, "info");
    sessionIdEl.textContent = session.id;
    setConnected(true);
    setStatus("Connected", true);

    if (debugMode) {
      log(`[SSE CLI] Debug mode ON`, "debug");
    }

    log(
      "[SSE CLI] Commands: /exit /new /abort /status /sessions /messages /clear /compact /export",
      "info"
    );

    // Start SSE polling
    seenMessageCount = 0;
    emptyPolls = 0;
    startPolling();
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    log(`[SSE CLI] Connection failed: ${msg}`, "error");
    client = null;
    session = null;
  }
}

async function doDisconnect(): Promise<void> {
  stopPolling();
  if (client) {
    await client.deleteSession(session?.id || "").catch(() => {});
    client = null;
  }
  session = null;
  sessionIdEl.textContent = "—";
  setConnected(false);
  setStatus("Disconnected", false);
  log("[SSE CLI] Disconnected.", "info");
}

function startPolling(): void {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollMessages, 500);
}

function stopPolling(): void {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollMessages(): Promise<void> {
  if (!client || !session) return;

  try {
    const messages = await client.getMessages(session.id, 50);

    let newCount = 0;
    for (let i = seenMessageCount; i < messages.length; i++) {
      newCount++;
      const msg = messages[i];
      for (const part of msg.parts) {
        const el = renderMessagePart(part);
        appendPart(el);
      }
    }
    seenMessageCount = messages.length;

    if (newCount === 0) {
      emptyPolls++;
      if (emptyPolls >= 4) {
        // 4 * 0.5s = 2s of no new messages
        try {
          const status = await client.getSessionStatus(session.id);
          const sessionStatus = status.type || "idle";
          if (sessionStatus === "idle" || sessionStatus === "error") {
            log(
              `[SSE CLI] Session ${sessionStatus}, stream ended.`,
              "info"
            );
            // Fetch final messages
            const finalMessages = await client.getMessages(session.id);
            for (const msg of finalMessages) {
              for (const part of msg.parts) {
                const el = renderMessagePart(part);
                appendPart(el);
              }
            }
            seenMessageCount = 0;
            emptyPolls = 0;
            stopPolling();
          }
        } catch {
          // ignore
        }
        emptyPolls = 0;
      }
    } else {
      emptyPolls = 0;
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    log(`[SSE CLI] get_messages error: ${msg}`, "error");
  }
}

async function sendPrompt(text: string): Promise<void> {
  if (!client || !session) return;

  // Check for slash commands
  if (text.startsWith("/")) {
    await handleCommand(text);
    return;
  }

  log(`> ${text}`, "user-input");

  const ok = await client.sendPromptAsync(session.id, text);
  if (!ok) {
    log("[SSE CLI] Failed to send prompt", "error");
    return;
  }

  log("[SSE CLI] Streaming events...", "info");
  seenMessageCount = 0;
  emptyPolls = 0;
  startPolling();
}

// ── Command Handler ───────────────────────────────────────────────

async function handleCommand(cmd: string): Promise<void> {
  if (!client || !session) return;

  const taskStr = cmd.slice(1); // remove leading /
  let taskSplit: string[];
  const splitIdx = taskStr.indexOf(":");
  if (splitIdx >= 0) {
    taskSplit = [taskStr.slice(0, splitIdx), taskStr.slice(splitIdx + 1)];
  } else {
    taskSplit = [taskStr];
  }

  const command = taskSplit[0];

  try {
    switch (command) {
      case "help":
        log(
          "[SSE CLI] Commands: /exit /new /abort /status /sessions /messages /clear /compact /export",
          "info"
        );
        break;

      case "new":
        stopPolling();
        session = await client.createSession("SSE Web debug session");
        log(`[SSE CLI] New session: ${session.id}`, "info");
        sessionIdEl.textContent = session.id;
        seenMessageCount = 0;
        emptyPolls = 0;
        outputEl.innerHTML = "";
        break;

      case "abort": {
        const ok = await client.abortSession(session.id);
        log(`[SSE CLI] Abort: ${ok ? "ok" : "failed"}`, ok ? "info" : "error");
        break;
      }

      case "status": {
        const status = await client.getSessionStatus(session.id);
        log(`[SSE CLI] Status: ${JSON.stringify(status)}`, "info");
        break;
      }

      case "sessions": {
        const sessions = await client.listSessions();
        for (const s of sessions) {
          log(`  ${s.id}: ${s.title || ""}`, "info");
        }
        break;
      }

      case "messages": {
        const messages = await client.getMessages(session.id, 20);
        for (const m of messages) {
          const content = m.text_content.slice(0, 100);
          log(`  [${m.role}] ${content}...`, "info");
        }
        break;
      }

      case "clear": {
        const ok = await client.clearSession(session.id);
        log(
          `[SSE CLI] Clear: ${ok ? "ok" : "failed"}`,
          ok ? "info" : "error"
        );
        outputEl.innerHTML = "";
        seenMessageCount = 0;
        break;
      }

      case "compact": {
        let keep: number | undefined;
        if (taskSplit.length > 1) {
          keep = parseInt(taskSplit[1], 10);
          if (isNaN(keep)) {
            log("[SSE CLI] Usage: /compact[:N] (N = messages to keep)", "error");
            return;
          }
        }
        const ok = await client.compactSession(session.id, keep);
        log(
          `[SSE CLI] Compact: ${ok ? "ok" : "failed"}`,
          ok ? "info" : "error"
        );
        break;
      }

      case "export": {
        const outputPath = taskSplit.length > 1 ? taskSplit[1] : undefined;
        const result = await client.exportSession(session.id, outputPath);
        log(
          `[SSE CLI] Export: ${result.count || 0} messages -> ${result.output || "n/a"}`,
          "info"
        );
        break;
      }

      case "exit":
        await doDisconnect();
        break;

      default:
        log(`[SSE CLI] Unrecognized command: ${command}`, "error");
        break;
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    log(`[SSE CLI] Command failed: ${msg}`, "error");
  }
}

// ── Event Listeners ───────────────────────────────────────────────

connectBtn.addEventListener("click", () => {
  doConnect().catch(console.error);
});

disconnectBtn.addEventListener("click", () => {
  doDisconnect().catch(console.error);
});

sendBtn.addEventListener("click", () => {
  const text = promptInput.value.trim();
  if (text) {
    sendPrompt(text).catch(console.error);
    promptInput.value = "";
  }
});

promptInput.addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key === "Enter") {
    const text = promptInput.value.trim();
    if (text) {
      sendPrompt(text).catch(console.error);
      promptInput.value = "";
    }
  }
});

// Command button listeners
btnNew.addEventListener("click", () => handleCommand("/new"));
btnAbort.addEventListener("click", () => handleCommand("/abort"));
btnStatus.addEventListener("click", () => handleCommand("/status"));
btnSessions.addEventListener("click", () => handleCommand("/sessions"));
btnMessages.addEventListener("click", () => handleCommand("/messages"));
btnClear.addEventListener("click", () => handleCommand("/clear"));
btnCompact.addEventListener("click", () => handleCommand("/compact"));
btnExport.addEventListener("click", () => handleCommand("/export"));

// ── Initial State ─────────────────────────────────────────────────

setConnected(false);
log("[SSE CLI] Ready. Enter host/port and click Connect.", "info");
