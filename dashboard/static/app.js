'use strict';
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 0) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(d) : '-';
const PHASES = ['locate', 'approach', 'grasp', 'twist', 'release', 'regrasp', 'check', 'retreat'];
const OUTCOMES = {
  opened: ['Opened', '--ok'], loosened: ['Loosened', '--loosened'], no_progress: ['No progress', '--neutral'],
  slip: ['Slip', '--warn'], misaligned: ['Misaligned', '--misaligned'], aborted: ['Aborted', '--bad'], fault: ['Fault', '--bad'],
};
const outColor = o => `var(${(OUTCOMES[o] || [0, '--neutral'])[1]})`;
const badge = (o, live) => live ? `<span class="badge live" style="--c:var(--accent)">In progress</span>`
  : `<span class="badge" style="--c:${outColor(o)}">${esc((OUTCOMES[o] || [o || 'unknown'])[0])}</span>`;
const J7_LOAD_HOT = 500;

const S = { cfg: null, learn: null, learnV: '', agent: null, agentV: '', seenCalls: 0, liveHist: [], doc: 'summary', traceKey: '' };

async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(j.error || r.status), { body: j });
  return j;
}
function setSrc(name, ok) { const el = $(`.src[data-src="${name}"]`); if (el) { el.classList.toggle('on', ok); el.classList.toggle('off', !ok); } }
function loop(fn, ms) { const tick = async () => { try { await fn(); } catch (e) { /* surfaced via status dots */ } setTimeout(tick, document.hidden ? ms * 4 : ms); }; tick(); }

/* ---------- live: cameras + joints ---------- */
function initCams() {
  const row = $('#cams');
  row.innerHTML = S.cfg.cameras.map(c => `<figure class="cam" id="cam-${c.role}"><img alt="${c.role} camera"><figcaption>${c.role}</figcaption><div class="off">${c.role} camera offline</div></figure>`).join('');
  const state = {};
  S.cfg.cameras.forEach(c => {
    const fig = $(`#cam-${c.role}`), img = $('img', fig);
    state[c.role] = 'stream';
    img.onload = () => { fig.classList.remove('offline'); setSrc('relay', true); };
    img.onerror = () => {
      if (state[c.role] === 'stream') { state[c.role] = 'snap'; snap(); return; }
      fig.classList.add('offline'); setSrc('relay', false);
      setTimeout(snap, 3000);
    };
    const snap = () => { img.src = `${S.cfg.relay}/snapshot/${c.hash}?_=${Date.now()}`; };
    img.addEventListener('load', () => { if (state[c.role] === 'snap') setTimeout(snap, 700); });
    img.src = `${S.cfg.relay}/stream/${c.hash}`;
  });
}

function initJoints() {
  $('#joints').innerHTML = [1, 2, 3, 4, 5, 6, 7, 8].map(i => `<div class="joint j${i}" id="j${i}">
    <b>j${i}${i === 7 ? ' roll' : i === 8 ? ' jaw' : ''}</b><div class="jbar"><div class="pos"></div><div class="goal"></div></div>
    <span class="num pos-v">-</span><div class="lbar" title="load"><i></i></div></div>`).join('');
}

async function pollLive() {
  let live;
  try { live = await getJSON('/api/live'); } catch (e) { setSrc('recorder', false); $('#live-age').textContent = 'arm telemetry unavailable (recorder /live)'; return; }
  setSrc('recorder', true);
  const bus = Object.values(live.buses || {}).find(b => b.role === 'follower') || Object.values(live.buses || {})[0];
  if (!bus) return;
  for (let i = 1; i <= 8; i++) {
    const j = bus['j' + i], el = $('#j' + i);
    if (!j || !el) continue;
    const n = j.norm ?? 0, g = (j.goal - j.range_min) / (j.range_max - j.range_min);
    $('.pos', el).style.height = `${Math.max(0, Math.min(1, n)) * 100}%`;
    $('.goal', el).style.bottom = `calc(${Math.max(0, Math.min(1, g)) * 100}% - 1px)`;
    $('.pos-v', el).textContent = n.toFixed(2);
    const lb = $('.lbar', el); lb.title = `load ${j.load}`;
    $('i', lb).style.width = `${Math.min(1, Math.abs(j.load) / 1000) * 100}%`;
    lb.classList.toggle('hot', Math.abs(j.load) >= J7_LOAD_HOT);
  }
  const grip = bus.gripper ?? bus.j8?.norm, j7 = bus.j7;
  const gg = $('#g-grip'); $('.g-val', gg).textContent = grip == null ? '-' : `${Math.round(grip * 100)}%`; $('b', gg).style.width = `${(grip ?? 0) * 100}%`;
  const gj = $('#g-j7');
  if (j7) {
    $('.g-val', gj).textContent = j7.load; $('b', gj).style.width = `${Math.min(1, Math.abs(j7.load) / 1000) * 100}%`;
    gj.classList.toggle('hot', Math.abs(j7.load) >= J7_LOAD_HOT);
    S.liveHist.push({ t: live.wall || Date.now() / 1000, pos: j7.norm, load: j7.load });
    const cut = (live.wall || Date.now() / 1000) - 60; while (S.liveHist.length && S.liveHist[0].t < cut) S.liveHist.shift();
  }
  $('#live-age').textContent = `telemetry ${new Date((live.wall || Date.now() / 1000) * 1000).toLocaleTimeString()}`;
  const cur = currentAttempt();
  if (cur && !cur.done && cur.phase === 'twist' && !(cur.files || []).includes('twist_trace.json')) drawLiveTrace();
}

/* ---------- learn state ---------- */
const currentAttempt = () => { const a = S.learn?.attempts || []; return a[a.length - 1]; };

async function pollLearn() {
  let st;
  try { st = await getJSON('/api/state?v=' + encodeURIComponent(S.learnV)); } catch (e) { setSrc('records', false); return; }
  setSrc('records', !!st.exists || !!st.unchanged);
  if (st.unchanged) { renderCurrentClock(); return; }
  S.learn = st; S.learnV = st.version;
  renderCurrent(); renderCharts(); renderTimeline(); renderDocTabs();
}

function paramRows(a, prev) {
  const keys = [...new Set([...Object.keys(a.params || {}), ...Object.keys(prev?.params || {})])];
  if (!keys.length) return '<p class="empty">No parameters recorded for this attempt.</p>';
  return `<table class="params"><thead><tr><th>parameter</th><th style="text-align:right">value</th></tr></thead><tbody>${keys.map(k => {
    const ch = (a.changed || []).includes(k), v = a.params?.[k], w = prev?.params?.[k];
    return `<tr class="${ch ? 'changed' : ''}"><td>${esc(k)}</td><td class="num">${ch && w !== undefined ? `<span class="was">${esc(fmtV(w))}</span> ` : ''}${esc(fmtV(v))}</td></tr>`;
  }).join('')}</tbody></table>`;
}
const fmtV = v => v === undefined ? '(removed)' : typeof v === 'number' ? String(+v.toFixed(3)) : typeof v === 'object' ? JSON.stringify(v) : String(v);

function renderCurrent() {
  const a = currentAttempt(), body = $('#cur-body');
  const n = S.learn.attempts.length;
  $('#learn-meta').textContent = n ? `${n} attempt${n > 1 ? 's' : ''} · ${S.learn.attempts.filter(x => x.outcome === 'opened').length} opened` : '';
  if (!a) {
    $('#cur-meta').textContent = '';
    body.innerHTML = `<p class="empty">No attempts yet. The first attempt appears here as soon as the operator writes <code>episode_begin</code> to <code>${esc(S.learn.learn_dir)}/episodes.jsonl</code>.</p>`;
    return;
  }
  const prev = S.learn.attempts[S.learn.attempts.length - 2];
  const done = new Map(); a.phases.forEach(p => done.set(p.phase, p));
  const counts = {}; a.phases.forEach(p => counts[p.phase] = (counts[p.phase] || 0) + 1);
  const last = a.phases[a.phases.length - 1];
  body.innerHTML = `
    <div class="cur-title"><span class="cur-n">#${a.attempt}</span>${badge(a.outcome, !a.done)}
      ${a.done ? `<span class="stat">rotation <b>${fmt(a.rotation_achieved_deg)}°</b></span><span class="stat">lift <b>${fmt(a.cap_lift_mm, 1)} mm</b></span>` : `<span class="stat">phase <b>${esc(a.phase || 'starting')}</b></span>`}</div>
    <div class="phases">${PHASES.map(p => { const r = done.get(p); return `<div class="ph ${r ? esc(r.status) : ''} ${!a.done && a.phase === p ? 'now' : ''}">${p}${counts[p] > 1 ? `<small>×${counts[p]}</small>` : ''}</div>`; }).join('')}</div>
    ${last?.note ? `<p class="phase-note">${esc(last.phase)}: ${esc(last.note)}</p>` : ''}
    <p class="cur-hyp"><span>Hypothesis${a.based_on?.length ? ` · builds on #${a.based_on.map(esc).join(', #')}` : ''}</span>${esc(a.hypothesis || 'not stated')}</p>
    ${paramRows(a, prev)}
    <div class="trace" id="trace"></div>`;
  renderCurrentClock();
  S.traceKey = '';
  loadTrace(a);
}
function renderCurrentClock() {
  const a = currentAttempt(); if (!a) return;
  const t0 = a.t_begin, t1 = a.done ? a.t_end : Date.now() / 1000;
  $('#cur-meta').textContent = t0 ? `${a.done ? 'took' : 'running'} ${Math.max(0, Math.round(t1 - t0))} s · started ${new Date(t0 * 1000).toLocaleTimeString()}` : '';
}

async function loadTrace(a) {
  const el = $('#trace'); if (!el) return;
  if (!(a.files || []).includes('twist_trace.json')) { if (!a.done && a.phase === 'twist') drawLiveTrace(); return; }
  try {
    const tr = await getJSON(`/api/trace/${a.attempt}`);
    drawTrace(el, tr.columns, `Twist trace, attempt ${a.attempt}`);
  } catch (e) { el.innerHTML = ''; }
}
function drawLiveTrace() {
  const el = $('#trace'); if (!el || S.liveHist.length < 2) return;
  const t0 = S.liveHist[0].t;
  drawTrace(el, { t: S.liveHist.map(r => r.t - t0), j7_pos: S.liveHist.map(r => r.pos), j7_load: S.liveHist.map(r => r.load) }, 'Live wrist roll (last 60 s)');
}

function pickCol(cols, re) { const k = Object.keys(cols).find(k => re.test(k) && cols[k].some(v => typeof v === 'number')); return k ? [k, cols[k]] : null; }
function drawTrace(el, cols, title) {
  const tc = pickCol(cols, /^(t|time|ts|t_s|elapsed)/i);
  const series = [pickCol(cols, /j7.*(pos|norm|angle)|(pos|norm).*j7/i), pickCol(cols, /j7.*goal|goal.*j7/i), pickCol(cols, /j7.*load|load.*j7/i), pickCol(cols, /tcp.*z|^z$/i)].filter(Boolean);
  if (!series.length) { el.innerHTML = ''; return; }
  const n = series[0][1].length, xs = tc ? tc[1] : [...Array(n).keys()];
  const W = 520, H = 46, G = 12, colors = ['var(--ink-2)', 'var(--ink-3)', 'var(--accent)', 'var(--loosened)'];
  let y = 0, out = '';
  series.forEach(([k, v], i) => {
    const pts = v.map((val, j) => [xs[j], val]).filter(p => typeof p[0] === 'number' && typeof p[1] === 'number');
    if (pts.length < 2) return;
    const x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0])) || 1;
    const lo = Math.min(...pts.map(p => p[1])), hi = Math.max(...pts.map(p => p[1]));
    const sx = x => 80 + (x - x0) / ((x1 - x0) || 1) * (W - 90), sy = v => y + H - (v - lo) / ((hi - lo) || 1) * H;
    out += `<text x="0" y="${y + 12}">${esc(k)}</text><text x="0" y="${y + 26}">${+hi.toFixed(2)}</text><text x="0" y="${y + H}">${+lo.toFixed(2)}</text>
      <line class="grid" x1="80" x2="${W - 10}" y1="${y + H}" y2="${y + H}"/>
      <polyline fill="none" stroke="${colors[i]}" stroke-width="1.6" stroke-linejoin="round" points="${pts.map(p => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ')}"/>`;
    y += H + G;
  });
  el.innerHTML = `<h3>${esc(title)}</h3><svg viewBox="0 0 ${W} ${y}" role="img" aria-label="${esc(title)}">${out}</svg>`;
}

/* ---------- charts ---------- */
function bars(values, { unit = '', colors, h = 150 } = {}) {
  const W = 520, L = 34, B = 20, T = 8, n = values.length;
  const nums = values.filter(v => typeof v === 'number');
  const max = Math.max(1, ...nums.map(Math.abs)) * 1.1, bw = (W - L) / Math.max(n, 1);
  const ticks = [0, max / 2, max].map(v => +v.toPrecision(2));
  let s = ticks.map(t => { const yy = T + (h - T - B) * (1 - t / max); return `<line class="grid" x1="${L}" x2="${W}" y1="${yy}" y2="${yy}"/><text x="${L - 6}" y="${yy + 3}" text-anchor="end">${t}</text>`; }).join('');
  values.forEach((v, i) => {
    const x = L + i * bw + bw * 0.18, w = bw * 0.64, label = S.learn.attempts[i].attempt;
    if (typeof v === 'number') {
      const bh = Math.max(1.5, (h - T - B) * Math.abs(v) / max);
      s += `<rect x="${x}" y="${h - B - bh}" width="${w}" height="${bh}" rx="3" fill="${colors ? colors[i] : 'var(--ink-3)'}"><title>#${label}: ${v}${unit}</title></rect>`;
    } else s += `<text x="${x + w / 2}" y="${h - B - 4}" text-anchor="middle">-</text>`;
    s += `<text x="${x + w / 2}" y="${h - 5}" text-anchor="middle">${label}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${h}" role="img">${s}</svg>`;
}

function renderCharts() {
  const A = S.learn.attempts;
  if (!A.length) {
    ['#ch-rot', '#ch-load', '#ch-params'].forEach(s => $(s).innerHTML = '<p class="empty">Charts fill in as attempts finish.</p>');
    $('#legend').innerHTML = ''; return;
  }
  $('#ch-rot').innerHTML = bars(A.map(a => a.rotation_achieved_deg), { unit: '°', colors: A.map(a => a.done ? outColor(a.outcome) : 'var(--accent)') });
  const used = [...new Set(A.map(a => a.outcome).filter(Boolean))];
  $('#legend').innerHTML = used.map(o => `<span style="--c:${outColor(o)}"><i></i>${esc((OUTCOMES[o] || [o])[0])}</span>`).join('');
  $('#ch-load').innerHTML = A.some(a => typeof a.peak_j7_load === 'number') ? bars(A.map(a => a.peak_j7_load), { colors: A.map(a => (a.peak_j7_load ?? 0) >= J7_LOAD_HOT ? 'var(--bad)' : 'var(--accent)') })
    : '<p class="empty">No j7 load measurements yet (episode_end measurements or twist_trace.json).</p>';
  const keys = [...new Set(A.flatMap(a => Object.keys(a.params || {})))];
  $('#ch-params').innerHTML = keys.map(k => paramChart(k, A)).join('') || '<p class="empty">No parameters recorded yet.</p>';
}
function paramChart(k, A) {
  const vals = A.map(a => a.params?.[k]), W = 200, H = 54, P = 6;
  const nums = vals.filter(v => typeof v === 'number');
  const last = vals.filter(v => v !== undefined).pop();
  let body;
  if (nums.length) {
    const lo = Math.min(...nums), hi = Math.max(...nums), sx = i => P + i * (W - 2 * P) / Math.max(A.length - 1, 1), sy = v => H - P - (v - lo) / ((hi - lo) || 1) * (H - 2 * P);
    const pts = vals.map((v, i) => typeof v === 'number' ? [sx(i), sy(v), v, i] : null).filter(Boolean);
    body = `<polyline fill="none" stroke="var(--ink-3)" stroke-width="1.4" points="${pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ')}"/>` +
      pts.map(p => `<circle cx="${p[0]}" cy="${p[1]}" r="${A[p[3]].changed.includes(k) ? 3.6 : 2.4}" fill="${A[p[3]].changed.includes(k) ? 'var(--accent)' : 'var(--ink-3)'}"><title>#${A[p[3]].attempt}: ${p[2]}</title></circle>`).join('') +
      `<text x="${W}" y="10" text-anchor="end">${+hi.toPrecision(3)}</text><text x="${W}" y="${H}" text-anchor="end">${+lo.toPrecision(3)}</text>`;
  } else {
    body = vals.map((v, i) => `<text x="${P + i * (W - 2 * P) / Math.max(A.length - 1, 1)}" y="${H / 2}" text-anchor="middle" ${A[i].changed.includes(k) ? 'style="fill:var(--accent-ink);font-weight:600"' : ''}>${esc(v === undefined ? '' : String(v).slice(0, 4))}</text>`).join('');
  }
  return `<div class="sm"><h4>${esc(k)}<span>now ${esc(fmtV(last))}</span></h4><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(k)} per attempt">${body}</svg></div>`;
}

/* ---------- timeline ---------- */
function thumb(a, names, label) {
  const f = (a.files || []).find(x => names.includes(x)) || (a.files || []).find(x => x.startsWith(label + '_') && /\.(jpe?g|png)$/i.test(x));
  if (!f) return '';
  const url = `/files/attempt_${a.attempt}/${encodeURIComponent(f)}?m=${S.learnV}`;
  return `<a href="${url}" target="_blank" rel="noopener"><img loading="lazy" src="${url}" alt="${label} attempt ${a.attempt}"><span>${label}</span></a>`;
}
function renderTimeline() {
  const A = S.learn.attempts, el = $('#timeline');
  if (!A.length) { el.innerHTML = '<li class="empty">The history of attempts builds up here: outcome, rotation, what the agent learned and what it changes next.</li>'; return; }
  el.innerHTML = A.slice().reverse().map((a, idx, rev) => {
    const prev = rev[idx + 1];
    const changes = (a.changed || []).map(k => `<span class="chip">${esc(k)} ${esc(fmtV(prev?.params?.[k]))} → ${esc(fmtV(a.params?.[k]))}</span>`).join('');
    const nxt = a.next_change && typeof a.next_change === 'object' ? Object.entries(a.next_change).map(([k, v]) => `<span class="chip next">${esc(k)} → ${esc(fmtV(v))}</span>`).join('') : (a.next_change ? `<span class="chip next">${esc(a.next_change)}</span>` : '');
    const judge = a.judges?.length ? a.judges[a.judges.length - 1] : null;
    return `<li class="att" id="att-${a.attempt}">
      <div class="att-n">#${a.attempt}</div>
      <div>
        <div class="att-head">${badge(a.outcome, !a.done)}
          <span class="stat">rotation <b>${fmt(a.rotation_achieved_deg)}°</b></span>
          <span class="stat">cap lift <b>${fmt(a.cap_lift_mm, 1)} mm</b></span>
          <span class="stat">slips <b>${a.slip_events ?? '-'}</b></span>
          <span class="stat">peak j7 load <b>${fmt(a.peak_j7_load)}</b></span>
          ${a.t_begin && a.t_end ? `<span class="stat">${Math.round(a.t_end - a.t_begin)} s</span>` : ''}</div>
        ${a.hypothesis ? `<p><span class="lbl">tried</span>${esc(a.hypothesis)}</p>` : ''}
        ${changes ? `<div class="chips">${changes}</div>` : ''}
        ${a.lesson ? `<p><span class="lbl">learned</span>${esc(a.lesson)}</p>` : (a.done ? '' : `<p class="stat">${esc(a.phase || 'starting')} …</p>`)}
        ${judge ? `<p><span class="lbl">judge (${esc(judge.source)})</span>${esc(judge.verdict)}</p>` : ''}
        ${nxt ? `<div class="chips"><span class="lbl">next</span>${nxt}</div>` : ''}
        <div class="att-links">
          ${a.recording_id ? `<a href="${esc(S.cfg.recorder)}/recordings/${encodeURIComponent(a.recording_id)}/viewer" target="_blank" rel="noopener">Replay recording</a>` : ''}
          ${S.learn.reports.includes(a.attempt) ? `<a href="#doc" data-doc="${a.attempt}">Read report</a>` : ''}
        </div>
      </div>
      <div class="thumbs">${thumb(a, ['before_wrist.jpg'], 'before')}${thumb(a, ['after_wrist.jpg'], 'after')}</div>
    </li>`;
  }).join('');
}

/* ---------- agent stream ---------- */
async function pollAgent() {
  let st;
  try { st = await getJSON('/api/agent?v=' + encodeURIComponent(S.agentV)); } catch (e) { setSrc('agent', false); return; }
  if (st.unchanged) return;
  S.agentV = st.version; S.agent = st;
  setSrc('agent', st.actions.length > 0 || !!st.operator);
  const op = st.operator;
  $('#op-text').innerHTML = op?.latest ? md(op.latest) : `<p class="empty">${st.operator_path ? `Waiting for assistant text in <code>${esc(st.operator_path)}</code>.` : 'Set <code>OPERATOR_LOG</code> to the Devin export/log file to show the operator\u2019s reasoning here.'}</p>`;
  const opEl = $('#op-text'); opEl.scrollTop = opEl.scrollHeight;
  const calls = st.actions, fresh = Math.max(0, calls.length - S.seenCalls);
  $('#calls-count').textContent = calls.length ? `last ${calls.length}` : '';
  $('#agent-meta').textContent = calls.length ? `latest ${calls[calls.length - 1].tool} at ${calls[calls.length - 1].t}` : '';
  const openSet = new Set([...document.querySelectorAll('#calls details[open]')].map(d => d.dataset.k));
  $('#calls').innerHTML = calls.length ? calls.slice().reverse().map((c, i) => {
    const k = `${c.t}|${c.tool}|${calls.length - 1 - i}`;
    return `<li><details class="call ${c.error ? 'err' : ''} ${S.seenCalls && i < fresh ? 'fresh' : ''}" data-k="${esc(k)}" ${openSet.has(k) ? 'open' : ''}>
    <summary><span class="t num">${esc(c.t)}</span><span class="tool">${esc(c.tool)}</span><span class="args">${esc(c.args === '{}' ? '' : c.args)}</span><span class="dur num">${c.s != null ? fmt(c.s, 1) + ' s' : ''}</span></summary>
    <pre>${esc(c.args)}\n\n${esc(c.error ? 'ERROR: ' + c.error + '\n' : '')}${esc(c.result)}</pre></details></li>`;
  }).join('') : `<li class="empty">No tool calls yet in <code>${esc(st.actions_path)}</code>.</li>`;
  S.seenCalls = calls.length;
}

/* ---------- docs ---------- */
function renderDocTabs() {
  const L = S.learn, tabs = [];
  if (L.summary) tabs.push(['summary', 'Summary']);
  L.reports.forEach(n => tabs.push([String(n), `Attempt ${n}`]));
  const dl = $('#dl'); dl.setAttribute('aria-disabled', L.summary ? 'false' : 'true');
  if (!tabs.length) { $('#doc-tabs').innerHTML = ''; $('#doc').innerHTML = '<p class="empty">SUMMARY.md and per-attempt reports appear here once the first attempt is written up.</p>'; return; }
  if (!tabs.some(t => t[0] === S.doc)) S.doc = tabs[0][0];
  $('#doc-tabs').innerHTML = tabs.map(([k, l]) => `<button class="tab" role="tab" type="button" data-doc="${k}" aria-selected="${k === S.doc}">${esc(l)}</button>`).join('');
  loadDoc();
}
async function loadDoc() {
  const k = S.doc, path = k === 'summary' ? 'SUMMARY.md' : `attempt_${k}/report.md`, base = k === 'summary' ? '' : `attempt_${k}/`;
  try {
    const r = await fetch(`/files/${path}?m=${S.learnV}`, { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    const txt = await r.text();
    if (S.doc === k) $('#doc').innerHTML = md(txt, base);
  } catch (e) { $('#doc').innerHTML = `<p class="empty">Could not load ${esc(path)}.</p>`; }
}
document.addEventListener('click', e => {
  const t = e.target.closest('[data-doc]'); if (!t) return;
  if (t.tagName === 'A') e.preventDefault();
  S.doc = t.dataset.doc; renderDocTabs();
  if (t.tagName === 'A') $('.docs').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

/* ---------- tiny markdown ---------- */
function inline(s, base) {
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (_, c) => (codes.push(c), `\u0000${codes.length - 1}\u0000`));
  s = esc(s)
    .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, src) => `<img alt="${alt}" src="${/^(https?:|\/)/.test(src) ? src : '/files/' + base + src}">`)
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, href) => `<a href="${/^(https?:|\/|#)/.test(href) ? href : '/files/' + base + href}" target="_blank" rel="noopener">${t}</a>`)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${esc(codes[i])}</code>`);
}
function md(src, base = '') {
  const lines = String(src).replace(/\r/g, '').split('\n'), out = [];
  let i = 0;
  const isTableSep = l => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
  const cells = l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
  while (i < lines.length) {
    const l = lines[i];
    if (/^```/.test(l)) { const buf = []; i++; while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]); i++; out.push(`<pre><code>${esc(buf.join('\n'))}</code></pre>`); continue; }
    let m;
    if ((m = l.match(/^(#{1,6})\s+(.*)/))) { const h = Math.min(4, m[1].length); out.push(`<h${h}>${inline(m[2], base)}</h${h}>`); i++; continue; }
    if (l.includes('|') && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const head = cells(l); i += 2; const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) rows.push(cells(lines[i++]));
      out.push(`<table><thead><tr>${head.map(c => `<th>${inline(c, base)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${inline(c, base)}</td>`).join('')}</tr>`).join('')}</tbody></table>`); continue;
    }
    if (/^\s*([-*+]|\d+\.)\s+/.test(l)) {
      const ordered = /^\s*\d+\./.test(l), buf = [];
      while (i < lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i])) buf.push(lines[i++].replace(/^\s*([-*+]|\d+\.)\s+/, ''));
      out.push(`<${ordered ? 'ol' : 'ul'}>${buf.map(b => `<li>${inline(b, base)}</li>`).join('')}</${ordered ? 'ol' : 'ul'}>`); continue;
    }
    if (/^>\s?/.test(l)) { const buf = []; while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, '')); out.push(`<blockquote>${inline(buf.join(' '), base)}</blockquote>`); continue; }
    if (/^\s*(---|\*\*\*)\s*$/.test(l)) { out.push('<hr>'); i++; continue; }
    if (!l.trim()) { i++; continue; }
    const buf = []; while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|```|>|\s*([-*+]|\d+\.)\s)/.test(lines[i]) && !(lines[i].includes('|') && isTableSep(lines[i + 1] || ''))) buf.push(lines[i++]);
    out.push(`<p>${inline(buf.join(' '), base)}</p>`);
  }
  return out.join('\n');
}

/* ---------- boot ---------- */
$('#theme').addEventListener('click', () => {
  const root = document.documentElement, dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('r3-theme', root.dataset.theme); } catch (e) { }
});
(async () => {
  S.cfg = await getJSON('/api/config');
  initCams(); initJoints();
  loop(pollLearn, 1500); loop(pollAgent, 1500); loop(pollLive, 500);
})();
