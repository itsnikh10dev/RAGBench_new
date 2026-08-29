/* ============================================================
   RAGBench frontend logic
   No frameworks — vanilla JS talking to the FastAPI backend.
   ============================================================ */

(() => {
  "use strict";

  // ------------------------------------------------------------
  // Element references
  // ------------------------------------------------------------
  const statusPill = document.getElementById("status-pill");
  const statusLabel = document.getElementById("status-label");

  const fileInput = document.getElementById("file-input");
  const dropzone = document.getElementById("dropzone");
  const fileListEl = document.getElementById("file-list");

  const processBtn = document.getElementById("process-btn");
  const processBtnLabel = document.getElementById("process-btn-label");
  const progressBlock = document.getElementById("progress-block");
  const progressSteps = Array.from(document.querySelectorAll(".progress-step"));

  const indexStatusEl = document.getElementById("index-status");
  const indexStatusValue = document.getElementById("index-status-value");

  const askForm = document.getElementById("ask-form");
  const questionInput = document.getElementById("question-input");
  const askBtn = document.getElementById("ask-btn");
  const askBtnLabel = document.getElementById("ask-btn-label");
  const askHint = document.getElementById("ask-hint");

  const conversation = document.getElementById("conversation");
  const emptyState = document.getElementById("empty-state");
  const turnTemplate = document.getElementById("tpl-turn");

  // ------------------------------------------------------------
  // State
  // ------------------------------------------------------------
  let pendingFiles = [];  // File objects chosen but not yet processed
  let indexReady = false;
  let isProcessing = false;
  let isAsking = false;

  // ============================================================
  // STATUS
  // ============================================================
  async function refreshStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) throw new Error("status request failed");
      const data = await res.json();

      indexReady = !!data.faiss_ready;

      if (data.api_configured) {
        statusPill.dataset.state = "online";
        statusLabel.textContent = "System online";
      } else {
        statusPill.dataset.state = "offline";
        statusLabel.textContent = "API key missing";
      }

      indexStatusEl.dataset.ready = String(indexReady);
      indexStatusValue.textContent = indexReady ? "Ready" : "Not built";

      updateAskAvailability();
    } catch (err) {
      statusPill.dataset.state = "offline";
      statusLabel.textContent = "Backend unreachable";
    }
  }

  function updateAskAvailability() {
    const canAsk = indexReady && !isAsking;
    questionInput.disabled = !canAsk;
    askBtn.disabled = !canAsk || questionInput.value.trim() === "";

    if (!indexReady) {
      askHint.textContent = "Upload and process at least one PDF to enable questions.";
      askHint.classList.remove("error");
    } else {
      askHint.textContent = "Press Enter or click \u201cAsk RAGBench\u201d to submit.";
      askHint.classList.remove("error");
    }
  }

  // ============================================================
  // FILE SELECTION
  // ============================================================
  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function addFiles(fileListLike) {
    const incoming = Array.from(fileListLike || []);
    for (const f of incoming) {
      if (!f.name.toLowerCase().endsWith(".pdf")) continue;
      const alreadyAdded = pendingFiles.some(
        (existing) => existing.name === f.name && existing.size === f.size
      );
      if (!alreadyAdded) pendingFiles.push(f);
    }
    renderFileList();
  }

  function removeFile(index) {
    pendingFiles.splice(index, 1);
    renderFileList();
  }

  function renderFileList() {
    fileListEl.innerHTML = "";

    pendingFiles.forEach((f, index) => {
      const card = document.createElement("div");
      card.className = "file-card";
      card.innerHTML = `
        <span class="file-icon">\u{1F4C4}</span>
        <div class="file-meta">
          <div class="file-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</div>
          <div class="file-size">${formatSize(f.size)}</div>
        </div>
        <button type="button" class="file-remove" aria-label="Remove ${escapeHtml(f.name)}">&times;</button>
      `;
      card.querySelector(".file-remove").addEventListener("click", () => removeFile(index));
      fileListEl.appendChild(card);
    });

    processBtn.disabled = pendingFiles.length === 0 || isProcessing;
  }

  fileInput.addEventListener("change", (e) => {
    addFiles(e.target.files);
    fileInput.value = ""; // allow re-selecting the same file later
  });

  // Drag and drop
  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files);
  });

  // ============================================================
  // PROCESS DOCUMENTS  (upload -> process)
  // ============================================================
  function setProgress(state) {
    // state: "idle" | "running" | "done" | "error"
    progressBlock.hidden = state === "idle";
    progressSteps.forEach((el) => {
      if (state === "running") el.dataset.status = "active";
      else if (state === "done") el.dataset.status = "done";
      else el.dataset.status = "";
    });
  }

  async function processDocuments() {
    if (isProcessing) return; // guard against double-click
    if (pendingFiles.length === 0) return;

    isProcessing = true;
    processBtn.disabled = true;
    processBtnLabel.textContent = "Processing\u2026";
    setProgress("running");
    clearBackendError();

    try {
      // 1) Upload
      const formData = new FormData();
      pendingFiles.forEach((f) => formData.append("files", f));

      const uploadRes = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const uploadData = await uploadRes.json().catch(() => ({}));
      if (!uploadRes.ok) {
        throw new Error(uploadData.detail || "Upload failed.");
      }

      // 2) Process
      const processRes = await fetch("/api/process", { method: "POST" });
      const processData = await processRes.json().catch(() => ({}));
      if (!processRes.ok) {
        throw new Error(processData.detail || "Processing failed.");
      }

      setProgress("done");
      pendingFiles = [];
      renderFileList();
      await refreshStatus();
      processBtnLabel.textContent = `Processed \u2713 (${processData.chunks} chunks)`;
    } catch (err) {
      setProgress("error");
      showBackendError(err.message || "Could not process documents.");
      processBtnLabel.textContent = "Process Documents";
    } finally {
      isProcessing = false;
      processBtn.disabled = pendingFiles.length === 0;
      setTimeout(() => {
        if (!isProcessing) processBtnLabel.textContent = "Process Documents";
      }, 3500);
    }
  }

  processBtn.addEventListener("click", processDocuments);

  function showBackendError(message) {
    askHint.textContent = message;
    askHint.classList.add("error");
  }
  function clearBackendError() {
    updateAskAvailability();
  }

  // ============================================================
  // ASK
  // ============================================================
  questionInput.addEventListener("input", updateAskAvailability);

  askForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question || isAsking || !indexReady) return;

    isAsking = true;
    updateAskAvailability();
    askBtnLabel.textContent = "Thinking\u2026";
    askBtn.disabled = true;

    if (emptyState) emptyState.remove();

    const turnNode = turnTemplate.content.firstElementChild.cloneNode(true);
    const userBody = turnNode.querySelector(".msg-user .msg-body");
    const aiBody = turnNode.querySelector(".msg-ai .msg-body");
    const sourcesEl = turnNode.querySelector(".sources");
    const sourcesCount = turnNode.querySelector(".sources-count");
    const sourcesList = turnNode.querySelector(".sources-list");

    userBody.textContent = question;
    aiBody.innerHTML = `
      <div class="msg-loading">
        <span>Retrieving relevant context</span>
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>
    `;
    conversation.appendChild(turnNode);
    conversation.scrollTop = conversation.scrollHeight;

    questionInput.value = "";

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data.detail || "Unable to contact the RAG backend.");
      }

      aiBody.innerHTML = renderMarkdown(data.answer || "Answer is not available in the context.");

      if (Array.isArray(data.sources) && data.sources.length > 0) {
        sourcesEl.hidden = false;
        sourcesCount.textContent = `(${data.sources.length})`;
        sourcesList.innerHTML = data.sources
          .map(
            (s) => `
            <div class="source-card">
              <div class="source-head">
                <span class="source-index">#${s.index}</span>
                <span class="source-name">${escapeHtml(s.source || "Document")}</span>
              </div>
              <div class="source-text">${escapeHtml(truncate(s.content, 900))}</div>
            </div>`
          )
          .join("");
      }
    } catch (err) {
      aiBody.innerHTML = `<div class="msg-error">${escapeHtml(err.message || "Unable to contact the RAG backend.")}</div>`;
    } finally {
      isAsking = false;
      askBtnLabel.textContent = "Ask RAGBench";
      updateAskAvailability();
      conversation.scrollTop = conversation.scrollHeight;
    }
  });

  // ============================================================
  // Lightweight, safe markdown renderer
  // (headings, bold, lists, code blocks/inline code, paragraphs)
  // ============================================================
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function truncate(str, max) {
    if (!str) return "";
    return str.length > max ? str.slice(0, max).trim() + "\u2026" : str;
  }

  function renderMarkdown(raw) {
    if (!raw || !raw.trim()) {
      return `<p class="answer-fallback">Answer is not available in the context.</p>`;
    }

    // Escape first, then reintroduce safe markup.
    let text = escapeHtml(raw.trim());

    // Fenced code blocks
    text = text.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);

    const lines = text.split("\n");
    const html = [];
    let listBuffer = [];
    let listType = null;

    function flushList() {
      if (listBuffer.length) {
        const tag = listType === "ol" ? "ol" : "ul";
        html.push(`<${tag}>${listBuffer.join("")}</${tag}>`);
        listBuffer = [];
        listType = null;
      }
    }

    for (let line of lines) {
      const trimmed = line.trim();

      if (trimmed === "") {
        flushList();
        continue;
      }
      if (trimmed.startsWith("<pre>")) {
        flushList();
        html.push(trimmed);
        continue;
      }

      const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
      if (headingMatch) {
        flushList();
        const level = headingMatch[1].length;
        html.push(`<h${level}>${inline(headingMatch[2])}</h${level}>`);
        continue;
      }

      const bulletMatch = trimmed.match(/^[-*]\s+(.*)$/);
      if (bulletMatch) {
        if (listType !== "ul") flushList();
        listType = "ul";
        listBuffer.push(`<li>${inline(bulletMatch[1])}</li>`);
        continue;
      }

      const numberedMatch = trimmed.match(/^\d+[\.\)]\s+(.*)$/);
      if (numberedMatch) {
        if (listType !== "ol") flushList();
        listType = "ol";
        listBuffer.push(`<li>${inline(numberedMatch[1])}</li>`);
        continue;
      }

      flushList();
      html.push(`<p>${inline(trimmed)}</p>`);
    }
    flushList();

    return html.join("\n");
  }

  function inline(str) {
    return str
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  // ============================================================
  // Init
  // ============================================================
  refreshStatus();
  updateAskAvailability();
  renderFileList();
})();
