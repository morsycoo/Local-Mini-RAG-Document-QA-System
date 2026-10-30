(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    conversationId: null,
    conversations: [],
    busy: false,
    activeRequests: new Set(),
    sessionReady: false,
    closing: false
  };

  const transcript = $("transcript");
  const messages = $("messages");
  const welcome = $("welcome");
  const input = $("message-input");
  const sendButton = $("send-button");
  const notice = $("notice");
  const toast = $("toast");
  let toastTimer;
  let navigationVersion = 0;

  async function api(path, options = {}) {
    const response = await fetch(path, {
      credentials: "same-origin",
      ...options,
      headers: {
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {})
      }
    });

    let data = {};
    try { data = await response.json(); } catch (_) {}

    if (!response.ok) {
      const error = new Error(data.error || data.detail || `HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function showToast(message) {
    toast.textContent = message;
    toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.hidden = true; }, 4200);
  }

  function setConnection(connected, label) {
    $("connection-dot").classList.toggle("offline", !connected);
    $("connection-label").textContent = label;
  }

  function showNotice(message) {
    notice.textContent = message || "";
    notice.hidden = !message;
  }

  function isConversationBusy(conversationId) {
    return Boolean(
      conversationId && state.activeRequests.has(conversationId)
    );
  }

  function syncBusyUI() {
    const busy = isConversationBusy(state.conversationId);
    state.busy = busy;
    sendButton.disabled = busy || !state.conversationId;
    input.disabled = false;
    $("send-label").textContent = busy
      ? "\u062c\u0627\u0631\u064a \u0627\u0644\u0625\u0631\u0633\u0627\u0644"
      : "\u0625\u0631\u0633\u0627\u0644";
  }

  function setBusy(busy, conversationId = state.conversationId) {
    if (conversationId) {
      if (busy) {
        state.activeRequests.add(conversationId);
      } else {
        state.activeRequests.delete(conversationId);
      }
    }
    syncBusyUI();
  }

  function scrollToLatest() {
    const nearBottom = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 160;
    if (nearBottom) transcript.scrollTop = transcript.scrollHeight;
  }

  function renderMessageContent(body, content) {
    body.textContent = content;
    if (window.marked && window.DOMPurify) {
      const html = window.marked.parse(content, {
        gfm: true,
        breaks: false
      });
      body.innerHTML = window.DOMPurify.sanitize(html, {
        USE_PROFILES: { html: true }
      });
    }
  }

  function addMessage(role, content, options = {}) {
    const article = document.createElement("article");
    article.className = `message ${role}`;
    if (state.conversationId) {
      article.dataset.conversationId = state.conversationId;
    }

    const label = document.createElement("div");
    label.className = "message-role";
    label.textContent = role === "user" ? "\u0623\u0646\u062a" : "\u062d\u064e\u062f\u0651\u064f\u0648\u062a\u064e\u0629";
    article.appendChild(label);

    const body = document.createElement("div");
    body.className = "message-content";
    body.dir = "auto";
    body.textContent = content;
    if (role === "assistant" && !options.pending) {
      renderMessageContent(body, content);
    }
    article.appendChild(body);

    if (options.pending) {
      body.classList.add("pending");
      body.textContent = "\u062c\u0627\u0631\u064d \u062a\u062c\u0647\u064a\u0632 \u0627\u0644\u0631\u062f...";
      article.dataset.pending = "true";
    }

    if (role === "assistant" && !options.pending) {
      const actions = document.createElement("div");
      actions.className = "message-actions";
      const copy = document.createElement("button");
      copy.type = "button";
      copy.textContent = "\u0646\u0633\u062e \u0627\u0644\u0646\u0635";
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(content);
          showToast("\u062a\u0645 \u0646\u0633\u062e \u0627\u0644\u0646\u0635.");
        } catch (_) {
          showToast("\u062a\u0639\u0630\u0631 \u0627\u0644\u0646\u0633\u062e. \u064a\u0645\u0643\u0646\u0643 \u062a\u062d\u062f\u064a\u062f \u0627\u0644\u0646\u0635 \u0648\u0646\u0633\u062e\u0647 \u064a\u062f\u0648\u064a\u064b\u0627.");
        }
      });
      actions.appendChild(copy);

      if (options.continuationAvailable === true) {
        const continueButton = document.createElement("button");
        continueButton.type = "button";
        continueButton.className = "continue-answer-button";
        continueButton.textContent = "\u0643\u0645\u0651\u0644 \u0627\u0644\u0625\u062c\u0627\u0628\u0629";
        continueButton.addEventListener("click", () => {
          continueMessage(article, continueButton);
        });
        actions.appendChild(continueButton);
      }

      article.appendChild(actions);
    }

    messages.appendChild(article);
    welcome.hidden = true;
    scrollToLatest();
    return article;
  }

  function showWelcome() {
    messages.replaceChildren();
    welcome.hidden = false;
  }

  function renderConversationList() {
    const list = $("conversation-list");
    list.replaceChildren();

    for (const item of state.conversations) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "conversation-item";
      button.classList.toggle("active", item.conversation_id === state.conversationId);
      button.textContent = item.title || "\u0645\u062d\u0627\u062f\u062b\u0629 \u062c\u062f\u064a\u062f\u0629";
      button.title = item.title || "\u0645\u062d\u0627\u062f\u062b\u0629 \u062c\u062f\u064a\u062f\u0629";
      button.addEventListener("click", () => openConversation(item.conversation_id));
      list.appendChild(button);
    }
  }

  async function refreshConversations() {
    const data = await api("/api/conversations");
    state.conversations = data.conversations || [];
    renderConversationList();
  }

  async function ensureSession() {
    try {
      await api("/api/conversations");
      state.sessionReady = true;
      return;
    } catch (error) {
      if (![401, 404].includes(error.status)) throw error;
    }

    await api("/api/session", { method: "POST" });
    state.sessionReady = true;
  }

  async function createConversation() {
    const navigation = ++navigationVersion;
    const data = await api("/api/conversations", {
      method: "POST",
      body: JSON.stringify({})
    });
    if (navigation !== navigationVersion) return;
    state.conversationId = data.conversation_id;
    syncBusyUI();
    $("current-title").textContent = data.title || "\u0645\u062d\u0627\u062f\u062b\u0629 \u062c\u062f\u064a\u062f\u0629";
    showWelcome();
    await refreshConversations();
    closeSidebar();
    input.focus();
  }

  async function openConversation(id) {
    const navigation = ++navigationVersion;
    showNotice("");
    try {
      const data = await api(`/api/conversations/${encodeURIComponent(id)}`);
      if (navigation !== navigationVersion) return;
      state.conversationId = data.conversation_id;
      syncBusyUI();
      $("current-title").textContent = data.title || "\u0645\u062d\u0627\u062f\u062b\u0629 \u062c\u062f\u064a\u062f\u0629";
      messages.replaceChildren();
      welcome.hidden = true;

      if (!data.messages || data.messages.length === 0) {
        showWelcome();
      } else {
        for (const message of data.messages) {
          addMessage(
            message.role === "user" ? "user" : "assistant",
            message.content,
            {
              continuationAvailable:
                message.role === "assistant" &&
                data.continuation_available === true &&
                message === data.messages[data.messages.length - 1]
            }
          );
        }
      }
      renderConversationList();
      closeSidebar();
      transcript.scrollTop = transcript.scrollHeight;
    } catch (error) {
      if (navigation !== navigationVersion) return;
      handleError(error, "\u062a\u0639\u0630\u0631 \u0625\u0643\u0645\u0627\u0644 \u0627\u0644\u0639\u0645\u0644\u064a\u0629.");
    }
  }

  async function continueMessage(article, button) {
    const conversationId = article.dataset.conversationId || state.conversationId;
    if (
      !conversationId ||
      state.conversationId !== conversationId ||
      isConversationBusy(conversationId)
    ) return;

    const body = article.querySelector(".message-content");
    if (!body) return;

    const oldLabel = button.textContent;
    button.disabled = true;
    button.textContent = "\u062c\u0627\u0631\u064a \u0627\u0633\u062a\u0643\u0645\u0627\u0644...";
    setBusy(true, conversationId);
    showNotice("");

    try {
      const data = await api(
        `/api/conversations/${encodeURIComponent(conversationId)}/continue`,
        { method: "POST" }
      );

      if (state.conversationId === conversationId) {
        if (body.isConnected) {
          renderMessageContent(body, data.content || "");

          if (data.continuation_available === true) {
            button.textContent = "\u0643\u0645\u0651\u0644 \u0627\u0644\u0625\u062c\u0627\u0628\u0629";
            button.disabled = false;
          } else {
            button.remove();
          }
        } else {
          // Navigation may have replaced the original DOM node while the
          // continuation was running. Reload the current history instead of
          // mutating a detached message element.
          await openConversation(conversationId);
        }
      }

      await refreshConversations();
      if (state.conversationId === conversationId) {
        scrollToLatest();
      }
    } catch (error) {
      if (body.isConnected) {
        button.disabled = false;
        button.textContent = oldLabel;
      }
      if (state.conversationId === conversationId) {
        handleError(error, "\u062a\u0639\u0630\u0651\u0631 \u0627\u0633\u062a\u0643\u0645\u0627\u0644 \u0627\u0644\u0625\u062c\u0627\u0628\u0629.");
      }
    } finally {
      setBusy(false, conversationId);
    }
  }

  async function sendMessage(text, retryOptions = {}) {
    const content = text.trim();
    if (!content) return;
    if (
      retryOptions.conversationId &&
      state.conversationId !== retryOptions.conversationId
    ) return;
    if (isConversationBusy(state.conversationId)) return;

    if (!state.conversationId) {
      await createConversation();
    }

    const conversationId = retryOptions.conversationId || state.conversationId;
    if (
      !conversationId ||
      state.conversationId !== conversationId ||
      isConversationBusy(conversationId)
    ) return;

    input.value = "";
    resizeInput();
    const userArticle = retryOptions.reuseUserArticle &&
      retryOptions.reuseUserArticle.isConnected
        ? retryOptions.reuseUserArticle
        : addMessage("user", content);
    const pending = addMessage("assistant", "", { pending: true });
    setBusy(true, conversationId);
    showNotice("");

    try {
      const data = await api(
        `/api/conversations/${encodeURIComponent(conversationId)}/messages`,
        { method: "POST", body: JSON.stringify({ content }) }
      );

      pending.remove();

      if (state.conversationId !== conversationId) {
        await refreshConversations();
        return;
      }

      if (data.done) {
        if (data.title) $("current-title").textContent = data.title;
        addMessage("assistant", data.content || "", {
          continuationAvailable: data.continuation_available === true
        });
        setConnection(true, "\u062c\u0627\u0647\u0632");
      } else {
        addMessage(
          "assistant",
          "\u062d\u062f\u062b \u062e\u0637\u0623. \u062d\u0627\u0648\u0644 \u0625\u0631\u0633\u0627\u0644 \u0631\u0633\u0627\u0644\u062a\u0643 \u0645\u0631\u0629 \u0623\u062e\u0631\u0649."
        );
      }

      await refreshConversations();
    } catch (error) {
      pending.remove();

      if (state.conversationId !== conversationId) {
        await refreshConversations();
        return;
      }

      const failed = addMessage(
        "assistant",
        "\u0644\u0645 \u064a\u0645\u0643\u0646 \u0625\u0643\u0645\u0627\u0644 \u0627\u0644\u0637\u0644\u0628. \u062a\u062d\u0642\u0642 \u0645\u0646 \u0627\u0644\u0627\u062a\u0635\u0627\u0644 \u0648\u062d\u0627\u0648\u0644 \u0645\u0631\u0629 \u0623\u062e\u0631\u0649."
      );

      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "text-button";
      retry.textContent = "\u0625\u0639\u0627\u062f\u0629 \u0627\u0644\u0645\u062d\u0627\u0648\u0644\u0629";
      retry.addEventListener("click", () => {
        failed.remove();
        sendMessage(content, {
          conversationId,
          reuseUserArticle: userArticle
        });
      });

      const actions = document.createElement("div");
      actions.className = "message-actions";
      actions.appendChild(retry);
      failed.appendChild(actions);
      handleError(error, "\u062a\u0639\u0630\u0631 \u0625\u0631\u0633\u0627\u0644 \u0627\u0644\u0631\u0633\u0627\u0644\u0629.");
    } finally {
      setBusy(false, conversationId);
      if (state.conversationId === conversationId) {
        input.focus();
      }
    }
  }

  function handleError(error, fallback) {
    const message = error.status === 429
      ? "\u062a\u062c\u0627\u0648\u0632\u062a \u062d\u062f\u0648\u062f \u0627\u0644\u0637\u0644\u0628\u0627\u062a. \u0627\u0646\u062a\u0638\u0631 \u0642\u0644\u064a\u0644\u064b\u0627 \u062b\u0645 \u062d\u0627\u0648\u0644 \u0645\u062c\u062f\u062f\u064b\u0627."
      : error.status === 503
        ? "\u0627\u0644\u062e\u062f\u0645\u0629 \u063a\u064a\u0631 \u0645\u062a\u0627\u062d\u0629. \u062a\u0623\u0643\u062f \u0645\u0646 \u0623\u0646 \u0627\u0644\u0646\u0645\u0648\u0630\u062c \u064a\u0639\u0645\u0644 \u0628\u0634\u0643\u0644 \u0635\u062d\u064a\u062d."
        : error.status === 401 || error.status === 404
          ? "\u062a\u0639\u0630\u0631 \u0627\u0644\u0648\u0635\u0648\u0644 \u0644\u0644\u0645\u062d\u062a\u0648\u0649. \u062d\u0627\u0648\u0644 \u0628\u062f\u0621 \u0645\u062d\u0627\u062f\u062b\u0629 \u062c\u062f\u064a\u062f\u0629."
          : fallback;

    showNotice(message);
    setConnection(false, "\u063a\u064a\u0631 \u0645\u062a\u0635\u0644");
    showToast(message);
  }

  function resizeInput() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  }

  function closeSidebar() {
    $("sidebar").classList.remove("open");
  }

  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(input.value);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  input.addEventListener("input", resizeInput);

  $("new-chat").addEventListener("click", async () => {
    try {
      await createConversation();
    } catch (error) {
      handleError(error, "\u062a\u0639\u0630\u0631 \u062d\u0630\u0641 \u0627\u0644\u0645\u062d\u0627\u062f\u062b\u0629.");
    }
  });

  $("menu-toggle").addEventListener("click", () => {
    $("sidebar").classList.toggle("open");
  });

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      input.value = button.dataset.prompt || "";
      resizeInput();
      input.focus();
    });
  });

  $("end-session").addEventListener("click", async () => {
    if (state.activeRequests.size > 0 || state.closing) return;
    if (!confirm("\u0647\u0644 \u062a\u0631\u064a\u062f \u0625\u0646\u0647\u0627\u0621 \u0627\u0644\u062c\u0644\u0633\u0629 \u0648\u062d\u0630\u0641 \u0627\u0644\u0645\u062d\u0627\u062f\u062b\u0627\u062a \u0627\u0644\u0645\u0624\u0642\u062a\u0629\u061f")) return;

    state.closing = true;
    try {
      await api("/api/session", { method: "DELETE" });
      state.sessionReady = false;
      state.conversationId = null;
      state.conversations = [];
      renderConversationList();
      showWelcome();
      $("current-title").textContent = "\u0645\u062d\u0627\u062f\u062b\u0629 \u062c\u062f\u064a\u062f\u0629";
      await ensureSession();
      await createConversation();
      showToast("\u062a\u0645 \u0625\u0646\u0647\u0627\u0621 \u0627\u0644\u062c\u0644\u0633\u0629 \u0648\u0628\u062f\u0621 \u062c\u0644\u0633\u0629 \u062c\u062f\u064a\u062f\u0629.");
    } catch (error) {
      handleError(error, "\u062a\u0639\u0630\u0631 \u0625\u0646\u0647\u0627\u0621 \u0627\u0644\u062c\u0644\u0633\u0629.");
    } finally {
      state.closing = false;
    }
  });

  window.addEventListener("pagehide", (event) => {
    if (!event.persisted && state.sessionReady && !state.closing) {
      fetch("/api/session/close", {
        method: "POST",
        credentials: "same-origin",
        keepalive: true
      }).catch(() => {});
    }
  });

  async function init() {
    try {
      await ensureSession();
      await refreshConversations();
      setConnection(true, "\u062c\u0627\u0647\u0632");

      if (state.conversations.length) {
        await openConversation(state.conversations[0].conversation_id);
      } else {
        await createConversation();
      }
    } catch (error) {
      handleError(error, "\u062a\u0639\u0630\u0631 \u062a\u0634\u063a\u064a\u0644 \u0627\u0644\u062a\u0637\u0628\u064a\u0642. \u062a\u0623\u0643\u062f \u0645\u0646 \u0623\u0646 HADOTA \u064a\u0639\u0645\u0644.");
    }
  }

  init();
})();
