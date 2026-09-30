/*
 * SFSA Assistant chat UI. Served by the API itself and shown in an iframe on
 * the wiki, so every call below is same-origin and carries the member's wiki
 * login cookie. Rules this file keeps:
 *   - text from the server is only ever set as plain text, never as HTML;
 *   - a link is rendered only if it is https and on an allow-listed host;
 *   - nothing is stored in the browser: the conversation lives in memory only.
 */
(function () {
  "use strict";

  var logEl = document.getElementById("log");
  var form = document.getElementById("form");
  var input = document.getElementById("q");
  var sendBtn = document.getElementById("send");
  var statusEl = document.getElementById("status");
  var noticeEl = document.getElementById("notice");
  var whoEl = document.getElementById("who");

  var turns = [];            // [{role, content}], last few sent along for context
  var citationHosts = [];
  var busy = false;
  var blocked = false;       // not logged in / service down: form stays locked
  var timer = null;

  function lock(locked) {
    input.disabled = locked;
    sendBtn.disabled = locked;
  }

  function notice(text) {
    noticeEl.textContent = text;
    noticeEl.hidden = !text;
  }

  function block(text) {
    blocked = true;
    lock(true);
    notice(text);
  }

  function needLogin() {
    block("Please log in to the SFSA wiki, then reopen this assistant.");
  }

  function unavailable() {
    block("The assistant is unavailable right now. Please try again shortly.");
  }

  function linkFor(source) {
    if (!source.url) { return null; }
    try {
      var u = new URL(source.url);
      var ok = u.protocol === "https:" &&
        (source.kind === "web" || citationHosts.indexOf(u.hostname) !== -1);
      if (!ok) { return null; }
      var a = document.createElement("a");
      a.href = u.href;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = source.label;
      return a;
    } catch (e) {
      return null;
    }
  }

  function addMessage(kind, text, sources) {
    var wrap = document.createElement("div");
    wrap.className = "msg " + kind;
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);

    if (sources && sources.length) {
      var box = document.createElement("div");
      box.className = "sources";
      box.appendChild(document.createTextNode("Sources: "));
      sources.forEach(function (s, i) {
        if (i > 0) { box.appendChild(document.createTextNode(", ")); }
        box.appendChild(linkFor(s) || document.createTextNode(s.label));
      });
      wrap.appendChild(box);
    }
    logEl.appendChild(wrap);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function startStatus() {
    var seconds = 0;
    statusEl.textContent = "Thinking… (answers can take up to a minute or two)";
    timer = setInterval(function () {
      seconds += 1;
      statusEl.textContent = "Thinking… " + seconds + "s";
    }, 1000);
  }

  function stopStatus() {
    if (timer) { clearInterval(timer); timer = null; }
    statusEl.textContent = "";
  }

  function finish() {
    busy = false;
    stopStatus();
    if (!blocked) { lock(false); input.focus(); }
  }

  function send(question) {
    return fetch("/chat", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, conversation_history: turns.slice(-10) })
    }).then(function (r) {
      if (r.status === 200) {
        return r.json().then(function (d) {
          addMessage("assistant", d.response, d.sources);
          turns.push({ role: "user", content: question });
          turns.push({ role: "assistant", content: d.response });
          turns = turns.slice(-10);
        });
      }
      if (r.status === 401) { needLogin(); return null; }
      return r.json().catch(function () { return {}; }).then(function (d) {
        addMessage("error", typeof d.detail === "string" ? d.detail : "Something went wrong. Please try again.");
      });
    }).catch(function () {
      addMessage("error", "Couldn't reach the assistant. Check your connection and try again.");
    });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (busy || blocked) { return; }
    var question = input.value.trim();
    if (!question) { return; }
    busy = true;
    lock(true);
    input.value = "";
    addMessage("user", question);
    startStatus();
    send(question).then(finish, finish);
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit ? form.requestSubmit() : sendBtn.click();
    }
  });

  fetch("/session", { credentials: "same-origin" }).then(function (r) {
    if (r.status === 401) { needLogin(); return; }
    if (r.status !== 200) { unavailable(); return; }
    return r.json().then(function (s) {
      whoEl.textContent = s.user;
      citationHosts = s.citation_hosts || [];
      if (s.max_question_chars) { input.maxLength = s.max_question_chars; }
      lock(false);
      input.focus();
    });
  }).catch(unavailable);
})();
