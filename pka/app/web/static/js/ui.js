(() => {
  const root = document.documentElement;
  const THEME_KEY = "pka-theme";
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");

  const applyTheme = (theme) => {
    const resolved = theme || (prefersDark.matches ? "dark" : "light");
    root.setAttribute("data-theme", resolved);
    localStorage.setItem(THEME_KEY, resolved);
  };

  applyTheme(localStorage.getItem(THEME_KEY));

  const themeButtons = ["theme-toggle", "theme-toggle-mobile", "theme-toggle-inline"]
    .map((id) => document.getElementById(id))
    .filter((btn) => btn instanceof HTMLElement);

  themeButtons.forEach((button) =>
    button.addEventListener("click", () => {
      const current = root.getAttribute("data-theme");
      applyTheme(current === "dark" ? "light" : "dark");
    })
  );

  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebar-toggle");
  if (sidebar && sidebarToggle) {
    const closeSidebar = () => sidebar.setAttribute("data-open", "false");
    sidebarToggle.addEventListener("click", () => {
      const open = sidebar.getAttribute("data-open") === "true";
      sidebar.setAttribute("data-open", open ? "false" : "true");
    });
    document.addEventListener("click", (event) => {
      if (sidebar.getAttribute("data-open") === "true") {
        if (!sidebar.contains(event.target) && event.target !== sidebarToggle) {
          closeSidebar();
        }
      }
    });
    window.addEventListener("resize", () => {
      if (window.innerWidth >= 960) {
        closeSidebar();
      }
    });
  }

  const textarea = document.getElementById("question");
  if (textarea instanceof HTMLTextAreaElement) {
    let resizeFrame = 0;
    const autoResize = () => {
      if (resizeFrame) cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => {
        textarea.style.height = "auto";
        textarea.style.height = `${textarea.scrollHeight}px`;
        resizeFrame = 0;
      });
    };
    textarea.addEventListener("input", autoResize);
    autoResize();
    try {
      textarea.focus({ preventScroll: true });
    } catch (err) {
      textarea.focus();
    }
  }

  const chatForm = document.getElementById("chat-form");
  const chatThread = document.getElementById("chat-thread");
  const emptyHero = document.getElementById("chat-empty-state");
  const typingIndicator = document.getElementById("typing-indicator");
  const composerStatus = document.getElementById("composer-status");
  const telemetryLatency = document.getElementById("chat-latency");
  const telemetryRunId = document.getElementById("chat-run-id");
  const COPY_ICON_SRC = "/static/assets/copy.svg";
  const attachmentInput = document.getElementById("chat-attachment-input");
  const imageInput = document.getElementById("chat-image-input");
  const attachmentList = document.getElementById("composer-attachments");
  const attachButton = document.getElementById("composer-attach");
  const screenshotButton = document.getElementById("composer-screenshot");
  const attachments = [];
  const ATTACHMENT_LIMIT = 5;
  const MAX_PREVIEW_CHARS = 4000;
  const TEXT_EXTENSIONS = new Set([
    "txt",
    "md",
    "markdown",
    "json",
    "csv",
    "log",
    "eml",
    "mbox",
    "html",
    "htm",
    "xml",
    "yaml",
    "yml",
    "rtf",
  ]);

  const formatBytes = (bytes) => {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / 1024 ** exponent;
    return `${value % 1 === 0 ? value : value.toFixed(1)} ${units[exponent]}`;
  };

  const isTextLike = (file) => {
    if (!file) return false;
    if (file.type && file.type.startsWith("text/")) return true;
    const name = file.name || "";
    const ext = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
    return TEXT_EXTENSIONS.has(ext);
  };

  const hideEmpty = () => {
    if (emptyHero && !emptyHero.classList.contains("hidden")) {
      emptyHero.classList.add("hidden");
    }
  };

  const toggleTyping = (active) => {
    if (!typingIndicator) return;
    typingIndicator.classList.toggle("hidden", !active);
  };

  const setComposerStatus = (text) => {
    if (!composerStatus) return;
    composerStatus.textContent = text || "";
    composerStatus.classList.toggle("active", Boolean(text));
  };

  const updateTelemetry = (latency, runId) => {
    if (telemetryLatency) telemetryLatency.textContent = latency ? `${latency} ms` : "--";
    if (telemetryRunId) telemetryRunId.textContent = runId ? String(runId).slice(0, 8) : "--";
  };

  let scrollFrame = 0;
  const scrollToBottom = () => {
    if (!chatThread) return;
    if (scrollFrame) cancelAnimationFrame(scrollFrame);
    scrollFrame = requestAnimationFrame(() => {
      try {
        chatThread.scrollTo({ top: chatThread.scrollHeight, behavior: "smooth" });
      } catch (err) {
        chatThread.scrollTop = chatThread.scrollHeight;
      }
      scrollFrame = 0;
    });
  };

  const escapeHtml = (text) =>
    text.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char] || char));

  const formatParagraphs = (text) => {
    const safe = escapeHtml(text);
    return safe.split(/\n{2,}/).map((block) => `<p class="leading-relaxed">${block.replace(/\n/g, "<br />")}</p>`).join("");
  };

  const readFilePreview = (file) =>
    new Promise((resolve) => {
      if (!isTextLike(file)) {
        resolve(null);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        const value = typeof reader.result === "string" ? reader.result : "";
        resolve(value.slice(0, MAX_PREVIEW_CHARS));
      };
      reader.onerror = () => resolve(null);
      reader.readAsText(file);
    });

  const createAttachmentId = () =>
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `att-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;

  const renderAttachmentList = () => {
    if (!attachmentList) return;
    if (!attachments.length) {
      attachmentList.innerHTML = "";
      attachmentList.classList.add("hidden");
      return;
    }
    attachmentList.classList.remove("hidden");
    attachmentList.innerHTML = "";
    attachments.forEach((item) => {
      const chip = document.createElement("div");
      chip.className = "attachment-chip";
      const label = document.createElement("span");
      label.textContent = `${item.name} · ${formatBytes(item.size)}${item.preview ? "" : " (preview not available)"}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove ${item.name}`);
      remove.dataset.attachmentId = item.id;
      remove.innerHTML = "&times;";
      chip.append(label, remove);
      attachmentList.appendChild(chip);
    });
  };

  const summarizeAttachmentsForPrompt = (list) => {
    if (!list.length) return "";
    return list
      .map((item, index) => {
        const header = `Attachment ${index + 1}: ${item.name} (${formatBytes(item.size)})`;
        if (item.preview) {
          return `${header}\n${item.preview}`;
        }
        return `${header}\n[Preview unavailable for this ${item.kind} attachment]`;
      })
      .join("\n\n");
  };

  const renderUserAttachmentsMarkup = (list) => {
    if (!list.length) return "";
    const items = list
      .map(
        (item, index) =>
          `<li><span>${index + 1}. ${escapeHtml(item.name)}</span><span>${escapeHtml(formatBytes(item.size))}</span></li>`,
      )
      .join("");
    return `<div class="message-attachments"><h4>Attachments</h4><ul>${items}</ul></div>`;
  };

  const addAttachment = async (file, kind) => {
    if (!file || attachments.length >= ATTACHMENT_LIMIT) return;
    const preview = await readFilePreview(file);
    attachments.push({
      id: createAttachmentId(),
      name: file.name || "Untitled",
      size: file.size || 0,
      type: file.type || "",
      kind,
      preview,
    });
  };

  const handleFileSelection = (fileList, kind) => {
    if (!fileList || !fileList.length) return;
    const openSlots = ATTACHMENT_LIMIT - attachments.length;
    if (openSlots <= 0) return;
    const files = Array.from(fileList).slice(0, openSlots);
    Promise.all(files.map((file) => addAttachment(file, kind)))
      .then(() => renderAttachmentList())
      .catch((error) => console.error("Failed to process attachment:", error));
  };

  const clearAttachments = () => {
    attachments.length = 0;
    renderAttachmentList();
    if (attachmentInput) attachmentInput.value = "";
    if (imageInput) imageInput.value = "";
  };

  if (attachButton instanceof HTMLButtonElement && attachmentInput instanceof HTMLInputElement) {
    attachButton.addEventListener("click", () => attachmentInput.click());
    attachmentInput.addEventListener("change", (event) => {
      handleFileSelection(event.target.files, "document");
      attachmentInput.value = "";
    });
  }

  if (screenshotButton instanceof HTMLButtonElement && imageInput instanceof HTMLInputElement) {
    screenshotButton.addEventListener("click", () => imageInput.click());
    imageInput.addEventListener("change", (event) => {
      handleFileSelection(event.target.files, "image");
      imageInput.value = "";
    });
  }

  attachmentList?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const attachmentId = target.dataset.attachmentId;
    if (!attachmentId) return;
    const index = attachments.findIndex((item) => item.id === attachmentId);
    if (index >= 0) {
      attachments.splice(index, 1);
      renderAttachmentList();
    }
  });

  const renderSources = (sources) => {
    if (!Array.isArray(sources) || sources.length === 0) return "";
    const items = sources
      .map((source) => {
        const id = escapeHtml(String(source.id ?? ""));
        const loc = escapeHtml(String(source.loc ?? ""));
        return `<li>${id}<span>${loc}</span></li>`;
      })
      .join("");
    return `<div class="message-sources"><h4>Sources</h4><ul>${items}</ul></div>`;
  };

  const attachMessageActions = (article, text) => {
    if (!article || !text) return;
    let actions = article.querySelector(".message-actions");
    if (!actions) {
      actions = document.createElement("div");
      actions.className = "message-actions";
      article.appendChild(actions);
    }
    actions.innerHTML = "";
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.setAttribute("aria-label", "Copy message");

    const renderCopyIcon = () => {
      copyBtn.innerHTML = "";
      const icon = document.createElement("img");
      icon.src = COPY_ICON_SRC;
      icon.alt = "";
      icon.width = 14;
      icon.height = 14;
      copyBtn.appendChild(icon);
    };

    renderCopyIcon();

    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(text).then(
        () => {
          copyBtn.textContent = "Copied";
          setTimeout(renderCopyIcon, 1500);
        },
        () => {
          copyBtn.textContent = "Error";
          setTimeout(renderCopyIcon, 1500);
        }
      );
    });
    actions.appendChild(copyBtn);
  };

  const createMessageElement = (role, html, { loading = false, copyText = "" } = {}) => {
    const article = document.createElement("article");
    article.className = `message-card ${role === "user" ? "message-user" : "message-assistant"}`;
    if (loading) article.classList.add("is-loading");
    article.innerHTML = html;
    chatThread?.appendChild(article);
    if (!loading && copyText) attachMessageActions(article, copyText);
    scrollToBottom();
    return article;
  };

  const renderAssistantMessage = (payload) => {
    const answer = payload?.answer ?? {};
    const blocks = [];

    if (typeof answer.answer === "string" && answer.answer.trim()) {
      blocks.push(`<div class="message-body">${formatParagraphs(answer.answer)}</div>`);
    }

    if (Array.isArray(answer.bullets) && answer.bullets.length) {
      const items = answer.bullets.map((item) => `<li>${escapeHtml(String(item))}</li>`).join("");
      blocks.push(`<ul class="message-list">${items}</ul>`);
    }

    if (answer.abstain) {
      blocks.unshift('<p class="message-banner">Assistant abstained</p>');
    }

    const sources = renderSources(answer.sources);
    if (sources) blocks.push(sources);

    const metaItems = [];
    if (payload?.latency_ms) metaItems.push(`${payload.latency_ms} ms`);
    metaItems.push("Generated locally", "Private session");
    if (payload?.run_id) metaItems.push(`Run ${String(payload.run_id).slice(0, 8)}`);
    const metaMarkup = metaItems
      .map((item) => `<span>${escapeHtml(String(item))}</span>`)
      .join("<span>|</span>");
    blocks.push(`<div class="message-meta">${metaMarkup}</div>`);

    return blocks.join("");
  };

  const suggestionTriggers = document.querySelectorAll("[data-suggest]");
  suggestionTriggers.forEach((node) => {
    node.addEventListener("click", () => {
      const suggestion = node.getAttribute("data-suggest") || "";
      if (textarea instanceof HTMLTextAreaElement && suggestion) {
        textarea.value = suggestion;
        textarea.dispatchEvent(new Event("input"));
        textarea.focus();
      }
    });
  });

  if (chatForm instanceof HTMLFormElement && chatThread) {
    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!(textarea instanceof HTMLTextAreaElement)) return;
      const question = textarea.value.trim();
      if (!question) return;

      const attachmentsSnapshot = attachments.map((item) => ({ ...item }));
      const attachmentsMarkup = renderUserAttachmentsMarkup(attachmentsSnapshot);
      const attachmentsSummary = summarizeAttachmentsForPrompt(attachmentsSnapshot);
      const payloadQuestion = attachmentsSummary
        ? `${question}\n\n[Attachments]\n${attachmentsSummary}`
        : question;

      hideEmpty();
      const userBlocks = [`<p class="leading-relaxed">${escapeHtml(question)}</p>`];
      if (attachmentsMarkup) userBlocks.push(attachmentsMarkup);
      createMessageElement("user", userBlocks.join(""), { copyText: payloadQuestion });

      chatForm.reset();
      textarea.dispatchEvent(new Event("input"));
      clearAttachments();

      const loaderHtml =
        '<div class="message-loader"><div class="loading-dots" aria-hidden="true"><span></span><span></span><span></span></div><span>Thinking...</span></div>';
      const assistantNode = createMessageElement("assistant", loaderHtml, { loading: true });
      toggleTyping(true);
      setComposerStatus("NestAi is thinking...");
      updateTelemetry(null, null);

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: payloadQuestion, mode: "synthesize" }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const payload = await response.json();
        assistantNode.classList.remove("is-loading");
        assistantNode.innerHTML = renderAssistantMessage(payload);
        attachMessageActions(assistantNode, payload?.answer?.answer ?? "");
        updateTelemetry(payload?.latency_ms ?? null, payload?.run_id ?? null);
        const latencyLabel = payload?.latency_ms ? `${payload.latency_ms} ms` : "complete";
        setComposerStatus(`Answered in ${latencyLabel}`);
      } catch (error) {
        assistantNode.classList.remove("is-loading");
        assistantNode.innerHTML = `<p class="leading-relaxed message-error">Request failed: ${
          error instanceof Error ? escapeHtml(error.message) : "Unknown error"
        }. Check your Ollama daemon and model configuration.</p>`;
        setComposerStatus("Request failed - check Ollama");
      } finally {
        toggleTyping(false);
      }
    });
  }

  const librarySearch = document.getElementById("library-search");
  const libraryList = document.getElementById("library-list");
  const libraryEmpty = document.getElementById("library-empty");
  if (librarySearch instanceof HTMLInputElement && libraryList) {
    const cards = Array.from(libraryList.querySelectorAll(".library-card"));
    const handleSearch = () => {
      const query = librarySearch.value.trim().toLowerCase();
      let visible = 0;
      cards.forEach((card) => {
        const title = card.dataset.title || "";
        const summary = card.dataset.summary || "";
        const match = !query || title.includes(query) || summary.includes(query);
        card.style.display = match ? "" : "none";
        if (match) visible += 1;
      });
      if (libraryEmpty) libraryEmpty.classList.toggle("hidden", visible > 0);
    };
    librarySearch.addEventListener("input", handleSearch);
    handleSearch();
  }

  const SETTINGS_KEY = "pka-settings";
  const defaults = { model: "qwen2.5:3b-instruct", strict: true, reranker: false, temperature: 0 };
  const settingsForm = document.getElementById("settings-form");
  const settingsFeedback = document.getElementById("settings-feedback");

  if (settingsForm instanceof HTMLFormElement) {
    let stored = {};
    try {
      stored = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null") || {};
    } catch {
      stored = {};
    }
    const current = { ...defaults, ...stored };

    settingsForm.querySelectorAll("[data-setting-key]").forEach((input) => {
      const key = input.getAttribute("data-setting-key");
      if (!key) return;
      if (input instanceof HTMLInputElement && input.type === "checkbox") {
        input.checked = Boolean(current[key]);
      } else if (input instanceof HTMLInputElement && input.type === "range") {
        input.value = String(current[key]);
      } else if (input instanceof HTMLSelectElement) {
        input.value = String(current[key]);
      }
    });

    settingsForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const values = { ...defaults };
      settingsForm.querySelectorAll("[data-setting-key]").forEach((input) => {
        const key = input.getAttribute("data-setting-key");
        if (!key) return;
        if (input instanceof HTMLInputElement && input.type === "checkbox") {
          values[key] = input.checked;
        } else if (input instanceof HTMLInputElement && input.type === "range") {
          values[key] = Number.parseFloat(input.value);
        } else if (input instanceof HTMLSelectElement) {
          values[key] = input.value;
        }
      });
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(values));
      if (settingsFeedback) {
        settingsFeedback.classList.remove("hidden");
        settingsFeedback.textContent = "Saved locally.";
        setTimeout(() => settingsFeedback.classList.add("hidden"), 1500);
      }
    });
  }
})();
