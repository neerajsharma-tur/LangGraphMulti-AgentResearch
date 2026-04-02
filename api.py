from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uuid

from graph.builder import build_graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="LangGraph Research Agent")
graph = build_graph()

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f0f2f5; height: 100vh; display: flex; flex-direction: column; }
  header { background: #0f172a; color: #fff; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header .title { font-size: 18px; font-weight: 600; }
  header .subtitle { font-size: 12px; color: #94a3b8; margin-top: 2px; }
  #messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
  .msg { max-width: 82%; padding: 12px 16px; border-radius: 14px; line-height: 1.6; font-size: 14px; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: #0f172a; color: #fff; border-bottom-right-radius: 4px; }
  .msg.agent { align-self: flex-start; background: #fff; color: #1e293b; border: 1px solid #e2e8f0; border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .msg.clarify { align-self: flex-start; background: #fefce8; color: #713f12; border: 1px solid #fde68a; border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .msg.clarify .clarify-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #a16207; margin-bottom: 6px; display: flex; align-items: center; gap: 4px; }
  .msg.agent h1, .msg.agent h2, .msg.agent h3 { margin: 12px 0 6px; color: #0f172a; }
  .msg.agent h1 { font-size: 17px; } .msg.agent h2 { font-size: 15px; } .msg.agent h3 { font-size: 14px; font-weight: 600; }
  .msg.agent ul, .msg.agent ol { margin: 6px 0 6px 20px; }
  .msg.agent li { margin: 4px 0; }
  .msg.agent a { color: #2563eb; text-decoration: none; } .msg.agent a:hover { text-decoration: underline; }
  .msg.agent p { margin: 6px 0; }
  .msg.agent strong { font-weight: 600; }
  .msg.agent code { background: #f1f5f9; padding: 2px 5px; border-radius: 3px; font-size: 13px; font-family: monospace; }
  .meta-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; padding-top: 10px; border-top: 1px solid #e2e8f0; }
  .badge { font-size: 11px; font-weight: 500; padding: 3px 8px; border-radius: 999px; }
  .badge.confidence { background: #dcfce7; color: #166534; }
  .badge.attempts { background: #dbeafe; color: #1e40af; }
  .badge.validation { background: #f3e8ff; color: #6b21a8; }
  .badge.validation.PASS { background: #dcfce7; color: #166534; }
  .badge.validation.FAIL { background: #fee2e2; color: #991b1b; }
  .loader { align-self: flex-start; display: flex; gap: 6px; padding: 12px 16px; background: #fff; border-radius: 14px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .loader span { width: 8px; height: 8px; background: #94a3b8; border-radius: 50%; animation: bounce 1.4s ease-in-out infinite; }
  .loader span:nth-child(2) { animation-delay: 0.2s; } .loader span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,80%,100% { transform: scale(0); } 40% { transform: scale(1); } }
  #input-bar { display: flex; gap: 8px; padding: 16px 24px; background: #fff; border-top: 1px solid #e2e8f0; }
  #input-bar input { flex: 1; padding: 10px 14px; border: 1px solid #cbd5e1; border-radius: 10px; font-size: 14px; outline: none; background: #f8fafc; transition: border-color 0.15s, background 0.15s; }
  #input-bar input:focus { border-color: #0f172a; background: #fff; }
  #input-bar button { padding: 10px 22px; background: #0f172a; color: #fff; border: none; border-radius: 10px; font-size: 14px; font-weight: 500; cursor: pointer; transition: background 0.15s; white-space: nowrap; }
  #input-bar button:hover:not(:disabled) { background: #1e293b; }
  #input-bar button:disabled { opacity: 0.45; cursor: not-allowed; }
  #clarify-hint { display: none; font-size: 12px; color: #a16207; background: #fefce8; border: 1px solid #fde68a; padding: 6px 12px; border-radius: 8px; margin: 0 24px 0; text-align: center; }
</style>
</head>
<body>
<header>
  <div>
    <div class="title">Research Agent</div>
    <div class="subtitle">Multi-agent · Clarity · Research · Validation · Synthesis</div>
  </div>
</header>
<div id="clarify-hint">Awaiting your clarification — answer the question above to continue research</div>
<div id="messages"></div>
<div id="input-bar">
  <input id="prompt" type="text" placeholder="Ask a research question..." autocomplete="off" />
  <button id="send" onclick="send()">Send</button>
</div>
<script>
  const msgs = document.getElementById('messages');
  const inp  = document.getElementById('prompt');
  const btn  = document.getElementById('send');
  const hint = document.getElementById('clarify-hint');

  let pendingThreadId = null;
  let awaitingClarification = false;

  function markdownToHtml(md) {
    let html = md
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^# (.+)$/gm, '<h1>$1</h1>')
      .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank">$1</a>');
    html = html.replace(/^(\\d+)\\. (.+)$/gm, '<li>$2</li>');
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\\/li>\\n?)+)/g, '<ul>$1</ul>');
    html = html.replace(/\\n{2,}/g, '</p><p>');
    html = '<p>' + html + '</p>';
    html = html.replace(/<p>\\s*(<h[123]>)/g, '$1').replace(/(<\\/h[123]>)\\s*<\\/p>/g, '$1');
    html = html.replace(/<p>\\s*(<ul>)/g, '$1').replace(/(<\\/ul>)\\s*<\\/p>/g, '$1');
    html = html.replace(/<p>\\s*<\\/p>/g, '');
    return html;
  }

  function addMsg(text, cls) {
    const d = document.createElement('div');
    d.className = 'msg ' + cls;
    if (cls === 'clarify') {
      d.innerHTML = '<div class="clarify-label">&#9888; Clarification needed</div>' + markdownToHtml(text);
    } else if (cls === 'agent') {
      d.innerHTML = markdownToHtml(text);
    } else {
      d.textContent = text;
    }
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
    return d;
  }

  function addAgentWithMeta(text, data) {
    const d = document.createElement('div');
    d.className = 'msg agent';
    d.innerHTML = markdownToHtml(text);
    const hasMeta = data.confidence_score != null || data.research_attempts != null || data.validation_result != null;
    if (hasMeta) {
      const badges = document.createElement('div');
      badges.className = 'meta-badges';
      if (data.confidence_score != null) {
        badges.innerHTML += '<span class="badge confidence">Confidence: ' + data.confidence_score + '/10</span>';
      }
      if (data.research_attempts != null) {
        badges.innerHTML += '<span class="badge attempts">Attempts: ' + data.research_attempts + '</span>';
      }
      if (data.validation_result != null) {
        const v = data.validation_result.toUpperCase().includes('PASS') ? 'PASS' : data.validation_result.toUpperCase().includes('FAIL') ? 'FAIL' : '';
        badges.innerHTML += '<span class="badge validation ' + v + '">Validation: ' + data.validation_result + '</span>';
      }
      d.appendChild(badges);
    }
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function showLoader() {
    const d = document.createElement('div');
    d.className = 'loader'; d.id = 'loader';
    d.innerHTML = '<span></span><span></span><span></span>';
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
  }
  function hideLoader() { const l = document.getElementById('loader'); if (l) l.remove(); }

  function setInputState(disabled) {
    btn.disabled = disabled;
    inp.disabled = disabled;
  }

  async function send() {
    const text = inp.value.trim();
    if (!text) return;
    addMsg(text, 'user');
    inp.value = '';
    setInputState(true);
    showLoader();

    const body = awaitingClarification
      ? { query: '', clarification: text, thread_id: pendingThreadId }
      : { query: text };

    try {
      const res = await fetch('/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      hideLoader();
      if (!res.ok) {
        addMsg('Server error: ' + (data.detail || res.statusText), 'agent');
        setInputState(false); inp.focus(); return;
      }

      if (data.needs_clarification) {
        pendingThreadId = data.thread_id;
        awaitingClarification = true;
        hint.style.display = 'block';
        inp.placeholder = 'Type your clarification...';
        addMsg(data.clarifying_question || 'Can you clarify your question?', 'clarify');
      } else {
        pendingThreadId = null;
        awaitingClarification = false;
        hint.style.display = 'none';
        inp.placeholder = 'Ask a research question...';
        addAgentWithMeta(data.answer || data.detail || 'No response', data);
      }
    } catch (e) {
      hideLoader();
      addMsg('Error: ' + e.message, 'agent');
    }
    setInputState(false);
    inp.focus();
  }

  inp.addEventListener('keydown', e => { if (e.key === 'Enter' && !btn.disabled) send(); });
  inp.focus();
</script>
</body>
</html>"""


class ResearchRequest(BaseModel):
    query: str
    clarification: Optional[str] = None
    thread_id: Optional[str] = None


class ResearchResponse(BaseModel):
    thread_id: str
    answer: Optional[str] = None
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    confidence_score: Optional[int] = None
    research_attempts: Optional[int] = None
    validation_result: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    return CHAT_HTML


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    try:
        thread_id = req.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Resuming an existing thread with a clarification answer
        if req.thread_id and req.clarification:
            snapshot = graph.get_state(config)
            if snapshot.next and "ask_user" in snapshot.next:
                graph.update_state(
                    config,
                    {"messages": [HumanMessage(content=req.clarification)]},
                )
            graph.invoke(None, config=config)
        else:
            # New query — start fresh
            graph.invoke(
                {
                    "messages": [HumanMessage(content=req.query)],
                    "clarification_count": 0,
                    "research_attempts": 0,
                },
                config=config,
            )

            snapshot = graph.get_state(config)
            if snapshot.next and "ask_user" in snapshot.next:
                clarifying_question = snapshot.values["messages"][-1].content
                return ResearchResponse(
                    thread_id=thread_id,
                    needs_clarification=True,
                    clarifying_question=clarifying_question,
                )

        final = graph.get_state(config).values
        return ResearchResponse(
            thread_id=thread_id,
            answer=final["messages"][-1].content,
            needs_clarification=False,
            confidence_score=final.get("confidence_score"),
            research_attempts=final.get("research_attempts"),
            validation_result=final.get("validation_result"),
        )
    except Exception as e:
        print(f"[/research] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
