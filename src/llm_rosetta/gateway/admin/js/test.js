// Side-effect-only module: exposes functions via window.* for inline HTML handlers.
import { S, _TEST_TIMEOUT_MS } from './state.js';
import { t } from './i18n.js';
import { api, showToast, esc } from './core.js';

// ===================== Test Model =====================
// Embedded test image (resized PNG transparency demo from Wikimedia Commons, public domain)
const TEST_IMAGE_URL = 'data:image/jpeg;base64,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';

function getTestLabel(type) { return t('test.' + type); }

function buildTestPayload(model, type, extraParam) {
  const base = {model, max_tokens: 256};
  switch(type) {
    case 'text':
      return {...base, messages:[{role:'user', content:'Say "Hello! The gateway test is working." and nothing else.'}]};
    case 'stream':
      return {...base, stream:true, messages:[{role:'user', content:'Count from 1 to 5, one number per line.'}]};
    case 'tools':
      return {...base,
        messages:[{role:'user', content:"What's the weather in San Francisco?"}],
        tools:[{type:'function', function:{
          name:'get_weather',
          description:'Get current weather for a location',
          parameters:{type:'object', properties:{location:{type:'string',description:'City name'}}, required:['location']}
        }}]
      };
    case 'vision':
      return {...base,
        messages:[{role:'user', content:[
          {type:'text', text:'Describe what you see in this image in one sentence.'},
          {type:'image_url', image_url:{url: TEST_IMAGE_URL}}
        ]}]
      };
    case 'reasoning':
      return {...base, reasoning_effort: 'low', messages:[{role:'user', content:'What is 15 * 37? Show your reasoning briefly.'}]};
    case 'embedding':
      return {model, input: 'Hello, world!'};
    case 'embed_batch':
      return {model, input: ['Hello, world!', 'The quick brown fox jumps over the lazy dog.', 'Embeddings are useful for semantic search.']};
    case 'embed_multimodal':
      return {model, input: [{type: 'text', text: 'Describe this image.'}, {type: 'image_url', image_url: {url: TEST_IMAGE_URL}}]};
    case 'matryoshka':
      return {model, input: 'Hello, world!', dimensions: extraParam};
    case 'rerank':
      return {model, query: 'What is machine learning?', documents: ['Machine learning is a subset of artificial intelligence.', 'The weather today is sunny.', 'Deep learning uses neural networks with many layers.']};
    case 'rerank_batch':
      return {model, query: 'What is machine learning?', documents: ['Machine learning is a subset of artificial intelligence.', 'The weather today is sunny.', 'Deep learning uses neural networks with many layers.', 'Python is a popular programming language.', 'Gradient descent is an optimization algorithm.'], top_n: 3};
  }
}

function toggleTestMenu(btn) {
  // Close any other open menus first
  document.querySelectorAll('.test-menu.open').forEach(m => m.classList.remove('open'));
  const menu = btn.parentElement.querySelector('.test-menu');
  menu.classList.toggle('open');
  // Close on outside click
  const close = (e) => { if (!menu.contains(e.target) && e.target !== btn) { menu.classList.remove('open'); document.removeEventListener('click', close); }};
  setTimeout(() => document.addEventListener('click', close), 0);
}

function truncatePayloadForDisplay(payload) {
  // Deep clone and truncate base64 data URIs for readable display
  const str = JSON.stringify(payload, null, 2);
  return str.replace(/(data:[^;]+;base64,)([A-Za-z0-9+/=]{80})[A-Za-z0-9+/=]+/g, '$1$2...[truncated]');
}

// --- Test request lifecycle ---
// Async task-based testing for non-stream requests.  The browser
// POSTs to /admin/api/test which returns immediately with a task_id;
// we then poll /admin/api/test/<id> until done.  This completely
// avoids the browser connection-pool exhaustion problem because the
// long-running upstream call lives in the gateway's httpx pool, not
// in the browser's 6-connection HTTP/1.1 limit.

function _abortPendingTest() {
  // Cancel browser-side stream fetch
  if (S._testAbortCtrl) { S._testAbortCtrl.abort(); S._testAbortCtrl = null; }
  // Cancel server-side async task
  if (S._testTaskId) {
    api.del('/admin/api/test/' + S._testTaskId).catch(() => {});
    S._testTaskId = null;
  }
  // Stop polling
  if (S._testPollTimer) { clearInterval(S._testPollTimer); S._testPollTimer = null; }
  // Stop elapsed timer from previous test
  if (S._testElapsedTimer) { clearInterval(S._testElapsedTimer); S._testElapsedTimer = null; }
}

function _newTestSignal() {
  _abortPendingTest();
  S._testAbortCtrl = new AbortController();
  const ctrl = S._testAbortCtrl;
  setTimeout(() => ctrl.abort(), _TEST_TIMEOUT_MS);
  return ctrl.signal;
}


function promptMatryoshka(model) {
  S._matryoshkaModel = model;
  document.getElementById('matryoshkaDim').value = '256';
  openModal('matryoshkaModal');
  document.getElementById('matryoshkaDim').focus();
}
function confirmMatryoshka() {
  const dim = parseInt(document.getElementById('matryoshkaDim').value, 10);
  if (isNaN(dim) || dim < 1) { showToast(t('error.invalidDimension'), 'error'); return; }
  window.closeModal('matryoshkaModal');
  runTest(S._matryoshkaModel, 'matryoshka', dim);
}

async function runTest(model, type, extraParam) {
  // Close any open dropdown
  document.querySelectorAll('.test-menu.open').forEach(m => m.classList.remove('open'));

  const output = document.getElementById('testOutput');
  const meta = document.getElementById('testMeta');
  const reqDetails = document.getElementById('testReqDetails');
  const reqBody = document.getElementById('testReqBody');
  const resDetails = document.getElementById('testResDetails');
  const resBody = document.getElementById('testResBody');
  const imgDetails = document.getElementById('testImageDetails');
  const imgBody = document.getElementById('testImageBody');
  document.getElementById('testModalTitle').textContent = `Test: ${getTestLabel(type)}`;
  meta.innerHTML = `<div class="meta-item"><strong>Model:</strong> ${esc(model)}</div>`;
  output.textContent = '';
  output.className = 'test-output';
  reqDetails.removeAttribute('open');
  resDetails.removeAttribute('open');
  imgDetails.removeAttribute('open');
  resBody.textContent = '';
  resDetails.style.display = 'none';
  // Show image preview for vision tests
  if (type === 'vision' || type === 'embed_multimodal') {
    imgDetails.style.display = '';
    imgBody.innerHTML = `<img src="${TEST_IMAGE_URL}" alt="test image">`;
  } else {
    imgDetails.style.display = 'none';
    imgBody.innerHTML = '';
  }
  openModal('testModal');

  const payload = buildTestPayload(model, type, extraParam);
  reqBody.textContent = truncatePayloadForDisplay(payload);

  const t0 = performance.now();
  _abortPendingTest();  // cancel any previous test

  if (type === 'embedding' || (type !== 'stream')) {
    // --- Async task path (embedding + all non-streaming tests) ---
    // Browser never holds a long connection; the gateway's httpx pool
    // handles the upstream call.
    const isEmbedType = ['embedding','embed_batch','embed_multimodal','matryoshka'].includes(type);
    const isRerankType = ['rerank','rerank_batch'].includes(type);
    const endpoint = isRerankType ? '/v1/rerank' : isEmbedType ? '/v1/embeddings' : '/v1/chat/completions';
    output.innerHTML = `<span class="test-spinner"></span>${t('test.sending')}`;
    // Show Cancel button and live elapsed timer
    const cancelBtn = document.getElementById('testCancelBtn');
    if (cancelBtn) cancelBtn.style.display = '';
    let _elapsedSec = 0;
    if (S._testElapsedTimer) clearInterval(S._testElapsedTimer);
    S._testElapsedTimer = setInterval(() => {
      _elapsedSec++;
      output.innerHTML = `<span class="test-spinner"></span>${t('test.sending')} ${_elapsedSec}s`;
    }, 1000);
    const _stopElapsed = () => { clearInterval(S._testElapsedTimer); S._testElapsedTimer = null; if (cancelBtn) cancelBtn.style.display = 'none'; };
    try {
      const startResp = await api.post('/admin/api/test', {endpoint, payload});
      const taskId = startResp.task_id;
      S._testTaskId = taskId;

      // Poll for result
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          clearInterval(S._testPollTimer);
          S._testPollTimer = null;
          _stopElapsed();
          // Cancel the server-side task on timeout
          if (S._testTaskId) {
            api.del('/admin/api/test/' + S._testTaskId).catch(() => {});
            S._testTaskId = null;
          }
          reject(new DOMException('Test timed out', 'AbortError'));
        }, _TEST_TIMEOUT_MS);

        S._testPollTimer = setInterval(async () => {
          try {
            const r = await api.post('/admin/api/test/' + taskId + '/poll');
            if (r.status === 'pending') return; // keep polling
            clearInterval(S._testPollTimer);
            S._testPollTimer = null;
            clearTimeout(timeout);
            S._testTaskId = null;

            if (r.status === 'cancelled') {
              _stopElapsed();
              reject(new DOMException('Request cancelled', 'AbortError'));
              return;
            }
            if (r.status === 'error') {
              _stopElapsed();
              reject(new Error(r.error || 'Unknown error'));
              return;
            }

            // --- done ---
            _stopElapsed();
            const elapsed = ((performance.now() - t0)/1000).toFixed(2);
            const body = r.body;
            const statusCode = r.status_code || 200;

            resDetails.style.display = '';
            resBody.textContent = typeof body === 'string' ? body : JSON.stringify(body, null, 2);

            if (statusCode >= 400) {
              const errMsg = body?.error?.message || JSON.stringify(body?.error || body);
              output.textContent = `Error ${statusCode}: ${errMsg}`;
              meta.innerHTML += `<div class="meta-item"><strong>Time:</strong> ${elapsed}s</div><div class="meta-item"><strong>Status:</strong> <span style="color:var(--red)">${statusCode}</span></div>`;
              resolve();
              return;
            }

            meta.innerHTML += `<div class="meta-item"><strong>Time:</strong> ${elapsed}s</div><div class="meta-item"><strong>Status:</strong> <span style="color:var(--green)">200 OK</span></div>`;

            if (isEmbedType) {
              if (body?.usage) {
                meta.innerHTML += `<div class="meta-item"><strong>Tokens:</strong> ${body.usage.prompt_tokens || '?'} in</div>`;
              }
              const dim = body?.data?.[0]?.embedding?.length;
              output.textContent = dim ? `Embedding OK — ${dim} dimensions` : t('test.emptyResponse');
            } else if (isRerankType) {
              // Rerank result
              const results = body?.results || body?.data || [];
              if (results.length > 0) {
                output.textContent = `Rerank OK — ${results.length} result${results.length !== 1 ? 's' : ''}\n\n` +
                  results.map((r, i) => `#${i+1} index=${r.index ?? i} score=${(r.relevance_score ?? r.score ?? 0).toFixed(4)}`).join('\n');
              } else {
                output.textContent = t('test.emptyResponse');
              }
            } else {
              // Chat completion result
              if (body?.usage) {
                meta.innerHTML += `<div class="meta-item"><strong>Tokens:</strong> ${body.usage.prompt_tokens || '?'} in / ${body.usage.completion_tokens || '?'} out</div>`;
              }
              const choice = body?.choices?.[0];
              if (type === 'tools' && choice?.message?.tool_calls) {
                const tc = choice.message.tool_calls[0];
                output.textContent = `Tool call: ${tc.function?.name || tc.name || '?'}(${tc.function?.arguments || JSON.stringify(tc.arguments) || '{}'})`;
                if (choice.message.content) output.textContent += '\n\nContent: ' + choice.message.content;
              } else {
                const content = choice?.message?.content;
                output.textContent = (content != null && content !== '') ? content : t('test.emptyResponse');
              }
            }
            resolve();
          } catch(pollErr) {
            _stopElapsed();
            // Network error during poll — stop polling
            clearInterval(S._testPollTimer);
            S._testPollTimer = null;
            clearTimeout(timeout);
            reject(pollErr);
          }
        }, 500);
      });
    } catch(e) {
      if (e.name === 'AbortError') {
        output.textContent = 'Request aborted (timeout or cancelled)';
      } else {
        output.textContent = `Error: ${e.message}`;
      }
    }
    return;
  }

  // --- Stream test: direct browser fetch (SSE needs live connection) ---
  // type === 'stream' is the only path that reaches here (others return above)
  const signal = _newTestSignal();
  const streamHeaders = {'Content-Type':'application/json', 'Connection':'close'};
  if (S.internalToken) streamHeaders['Authorization'] = `Bearer ${S.internalToken}`;

  output.classList.add('streaming');
  output.innerHTML = `<span class="test-spinner"></span>${t('test.connecting')}`;
  try {
    const res = await fetch('/v1/chat/completions', {
      method:'POST', headers:streamHeaders, body:JSON.stringify(payload), signal
    });
    if (!res.ok) {
      const errText = await res.text();
      output.className = 'test-output';
      output.textContent = `Error ${res.status}: ${errText}`;
      meta.innerHTML += `<div class="meta-item"><strong>Time:</strong> ${((performance.now()-t0)/1000).toFixed(2)}s</div><div class="meta-item"><strong>Status:</strong> <span style="color:var(--red)">${res.status}</span></div>`;
      resDetails.style.display = '';
      resBody.textContent = errText;
      return;
    }
    output.textContent = '';
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let buffer = '';
    let rawChunks = [];
    while(true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') { rawChunks.push('[DONE]'); continue; }
        rawChunks.push(data);
        try {
          const chunk = JSON.parse(data);
          const delta = chunk.choices?.[0]?.delta?.content || '';
          fullText += delta;
          output.textContent = fullText;
          output.scrollTop = output.scrollHeight;
        } catch(e) {}
      }
    }
    output.className = 'test-output';
    if (!fullText) output.textContent = t('test.emptyResponse');
    const elapsed = ((performance.now() - t0)/1000).toFixed(2);
    meta.innerHTML += `<div class="meta-item"><strong>Time:</strong> ${elapsed}s</div><div class="meta-item"><strong>Status:</strong> <span style="color:var(--green)">OK (stream)</span></div>`;
    resDetails.style.display = '';
    resBody.textContent = rawChunks.map(c => c === '[DONE]' ? c : JSON.stringify(JSON.parse(c), null, 2)).join('\n---\n');
  } catch(e) {
    output.className = 'test-output';
    output.textContent = e.name === 'AbortError' ? 'Request aborted (timeout or cancelled)' : `Network error: ${e.message}`;
  }
}

Object.assign(window, {
  getTestLabel, buildTestPayload, toggleTestMenu,
  promptMatryoshka, confirmMatryoshka, runTest,
});
