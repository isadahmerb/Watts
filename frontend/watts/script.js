
const API_URL = "http://localhost:8000/chat";   // URL


function buildRequestBody(message, context) {
  return { message: message, context: context };
}

let lastQuestion = null;   


function extractReply(data) {
  return data.response ?? data.reply ?? data.message ?? "";
}

// ===========================================================

const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const messages = document.getElementById("messages");

function addMessage(text, type) {
  const el = document.createElement("div");
  el.className = "msg " + type;
  el.textContent = text;          
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function addTyping() {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  const context = lastQuestion;
  lastQuestion = text;

  document.body.classList.add("started");
  addMessage(text, "user");
  input.value = "";
  sendBtn.disabled = true;

  const typing = addTyping();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildRequestBody(text, context)),
    });

    if (!res.ok) throw new Error("Erro " + res.status);

    const data = await res.json();
    typing.remove();
    addMessage(extractReply(data) || "(resposta vazia)", "bot");
  } catch (err) {
    typing.remove();
    addMessage("Não consegui falar com o servidor. Tente novamente em instantes.", "bot error");
    console.error(err);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

input.focus();