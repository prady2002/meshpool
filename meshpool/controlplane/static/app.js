'use strict';
/* MeshPool dashboard — Use / Contribute / Wallet / Explore.
   Zero-build vanilla JS. Talks to the same REST API the CLI uses. */
const $ = (id) => document.getElementById(id);

// ---------- session (zero-signup local sign-in) ----------
let SESSION = { token: null, owner: 'default', gating: false, tier: 0, balance: 0, deviceId: null };

const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let d; try { d = (await r.json()).detail; } catch (e) {}
    const err = new Error(d || (r.status + ' ' + r.statusText));
    err.status = r.status;
    throw err;
  }
  return r.status === 204 ? null : r.json();
};
const sHdr = () => (SESSION.token ? { 'X-MeshPool-Session': SESSION.token } : {});
const post = (path, body, auth) => api(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', ...(auth ? sHdr() : {}) },
  body: JSON.stringify(body || {}),
});
// Consume calls attach the session and transparently re-auth once if it expired
// (e.g. after a coordinator restart) so the user never sees a spurious 401.
async function postAuthed(path, body) {
  try { return await post(path, body, true); }
  catch (e) {
    if (e.status === 401 && SESSION.gating) { await authLocal(); return await post(path, body, true); }
    throw e;
  }
}
async function authLocal() {
  try {
    const d = await post('/api/v1/auth/local');
    SESSION = { token: d.session_token, owner: d.owner, gating: d.gating,
                tier: d.tier, balance: d.balance, deviceId: d.device_id };
  } catch (e) { /* stay anonymous; LAN mode needs no session */ }
  renderChip();
  return SESSION;
}
function renderChip() {
  const b = $('balance');
  if (SESSION.gating) { b.textContent = fmtCr(SESSION.balance); $('walletChip').style.display = ''; }
  else { $('walletChip').style.display = 'none'; }
}

// ---------- small helpers ----------
function toast(msg, isErr) { const t = $('toast'); t.textContent = msg; t.className = 'toast show' + (isErr ? ' err' : ''); clearTimeout(toast._t); toast._t = setTimeout(() => t.className = 'toast', 3200); }
function copyText(s) { navigator.clipboard?.writeText(s.trim()).then(() => toast('Copied to clipboard')); }
const fmtMB = (mb) => mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB';
const fmtCr = (c) => (c == null ? '—' : (Math.abs(c) >= 100 ? Math.round(c) : (+c).toFixed(1)));
const esc = (s) => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const shortOwner = (o) => String(o || '').replace(/^dev:dev_/, '').replace(/^acct:/, '@').slice(0, 14);

// ---------- tab switching ----------
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + name));
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === name));
  if (name === 'wallet') loadWallet();
  if (name === 'explore') { loadCluster(); loadLedger(); }
  if (name === 'contribute') { loadCluster(); loadOnboarding(); }
  if (name === 'use') { loadCluster(); loadJobs(); loadDeps(); loadModel(); }
}
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => showView(t.dataset.view)));

// ---------- shared cluster load (stats + node lists) ----------
let LAST_NODES = [];
async function loadCluster() {
  try {
    const d = await api('/api/v1/cluster'); const s = d.summary;
    LAST_NODES = d.nodes;
    const online = d.nodes.filter(n => n.status === 'online');
    const gflops = online.reduce((a, n) => a + (n.capability.flops_estimate || 0), 0);
    const poolCards = [
      ['Machines online', `${s.online}<small> / ${s.total_nodes}</small>`],
      ['Pooled cores', s.total_cores],
      ['Pooled memory', fmtMB(s.total_ram_mb).replace(' ', '<small> ') + '</small>'],
      ['Aggregate compute', `${gflops.toFixed(0)}<small> GFLOPS</small>`],
    ];
    const statHtml = poolCards.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
    if ($('poolStats')) $('poolStats').innerHTML = statHtml;
    if ($('exploreStats')) $('exploreStats').innerHTML = statHtml;

    // Explore: all machines
    $('nodeCnt').textContent = s.online ? `${s.online} live` : '';
    $('nodes').innerHTML = d.nodes.length
      ? d.nodes.slice().sort((a, b) => a.name.localeCompare(b.name)).map(poolNodeCard).join('')
      : `<div class="empty">No machines yet — open <b>Contribute</b> to add one.</div>`;

    // Contribute: only the machines this owner runs
    const mine = d.nodes.filter(n => n.owner === SESSION.owner || (!SESSION.gating && n.owner === 'default'));
    const myOnline = mine.filter(n => n.status === 'online').length;
    $('myNodeCnt').textContent = mine.length ? `${myOnline} online` : '';
    $('myNodes').innerHTML = mine.length
      ? mine.sort((a, b) => a.name.localeCompare(b.name)).map(myNodeCard).join('')
      : `<div class="empty">You're not contributing any machines yet.<br>Use the pairing token on the right to add this or another computer.</div>`;
    if ($('contribStats')) $('contribStats').innerHTML = [
      ['Your machines', mine.length],
      ['Online now', myOnline],
      ['Credit balance', SESSION.gating ? fmtCr(SESSION.balance) : '\u221e'],
      ['Reputation', (WALLET.reputation != null ? WALLET.reputation.toFixed(2) : '1.00')],
    ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  } catch (e) { /* transient */ }
}

// ---------- node cards ----------
function nodeSpecs(n) {
  const cap = n.capability;
  const gpu = cap.gpu ? `GPU ${fmtMB(cap.vram_mb)}` : 'CPU-only';
  return `<div class="specs"><span>${cap.cpu_cores} cores</span><span>${fmtMB(cap.ram_mb)}</span>
    <span>${gpu}</span><span>~<b>${(cap.flops_estimate || 0).toFixed(0)}</b> GFLOPS</span>
    <span>trust <b>${(n.trust_score ?? 1).toFixed(2)}</b></span><span>${(n.link.latency_ms || 0).toFixed(0)}ms</span></div>`;
}
function poolNodeCard(n) {
  const cpu = Math.min(100, n.load.cpu_load_pct || 0);
  const via = n.reachable_via === 'tunnel' ? '<span class="pill via">via tunnel</span>' : '';
  return `<div class="node ${n.status === 'offline' ? 'off' : ''}">
    <div class="h"><span class="name">${esc(n.name)}</span>
      <span class="badge ${n.status}">${n.status}</span>${via}
      <span class="owner">${esc(shortOwner(n.owner))}</span></div>
    ${nodeSpecs(n)}
    <div class="bar cpu"><i style="width:${cpu}%"></i></div></div>`;
}
function myNodeCard(n) {
  const pol = n.policy, cap = n.capability, load = n.load;
  const cpu = Math.min(100, load.cpu_load_pct || 0);
  const slots = Math.max(1, Math.floor((pol.cpu_pct / 100) * cap.cpu_cores));
  const draining = n.status === 'draining';
  const act = n.status === 'offline' ? '' :
    (draining
      ? `<button class="btn sm" onclick="nodeAct('${n.id}','resume')">Resume donation</button>`
      : `<button class="btn d sm" onclick="nodeAct('${n.id}','reclaim')">\u23fb Reclaim now</button>`) +
    ` <select class="btn sm" style="width:auto;padding:4px 8px" onchange="setSchedule('${n.id}',this.value)" title="donation schedule">
        ${['always', 'idle_only', 'business_hours', 'outside_hours'].map(s => `<option value="${s}" ${pol.schedule === s ? 'selected' : ''}>${s.replace('_', '-')}</option>`).join('')}
      </select>`;
  const donate = n.status === 'offline' ? '' :
    `<label class="donate" title="live CPU donation cap — OS-enforced">
       <span>donate</span>
       <input type="range" min="0" max="100" step="5" value="${pol.cpu_pct}"
         oninput="this.nextElementSibling.textContent=this.value+'%'"
         onchange="setDonate('${n.id}',this.value)">
       <b>${pol.cpu_pct}%</b>
     </label>`;
  return `<div class="node ${n.status === 'offline' ? 'off' : ''}">
    <div class="h"><span class="name">${esc(n.name)}</span>
      <span class="badge ${n.status}">${n.status}</span>
      <span class="meta">${slots} slots · headroom ${n.headroom_cpu_pct}%</span></div>
    ${nodeSpecs(n)}
    <div class="bar cpu"><i style="width:${cpu}%"></i></div>
    <div class="actions">${donate}${act}</div></div>`;
}
async function nodeAct(id, action) { try { await post(`/api/v1/nodes/${id}/${action}`); toast(action === 'reclaim' ? 'Resources reclaimed' : 'Donation resumed'); loadCluster(); } catch (e) { toast(e.message, true); } }
async function setSchedule(id, schedule) { try { await post(`/api/v1/nodes/${id}/policy`, { schedule }); toast('Schedule \u2192 ' + schedule); } catch (e) { toast(e.message, true); } }
async function setDonate(id, pct) { try { await post(`/api/v1/nodes/${id}/policy`, { cpu_pct: Number(pct) }); toast('Donation \u2192 ' + pct + '%'); loadCluster(); } catch (e) { toast(e.message, true); } }

// ---------- USE: banner + batch ----------
function useBanner() {
  const el = $('useBanner'); if (!el) return;
  if (SESSION.gating && SESSION.balance <= 0) {
    el.innerHTML = `<div class="banner warn">You're out of compute-credits. Head to
      <a onclick="showView('contribute')">Contribute</a> to donate spare capacity and earn more.</div>`;
  } else if (SESSION.gating) {
    el.innerHTML = `<div class="banner">Signed in as <b>${esc(shortOwner(SESSION.owner))}</b> ·
      <b>${fmtCr(SESSION.balance)}</b> credits to spend. Every job you run debits credits and pays the machines that did the work.</div>`;
  } else {
    el.innerHTML = `<div class="banner">Local pool — no credit limits. Everything you submit runs across your machines.</div>`;
  }
}
const FUNC_HINTS = {
  sha_pow: 'Hash-based proof-of-work — a pure CPU burn, great for showing fan-out.',
  is_prime: 'Primality test on large numbers — quick, CPU-bound units.',
  embed: 'Turn text into embedding vectors — verifiable against a local oracle.',
  square: 'Square a number — trivial, for demoing the pipeline end-to-end.',
  sleep_echo: 'Sleep then echo — used to demo requeue/failover when a node dies.',
};
function describeFunc() {
  const f = $('jFunc').value;
  $('funcHint').textContent = FUNC_HINTS[f] || '';
  const defaults = { sha_pow: 24, is_prime: 24, embed: 20, square: 24, sleep_echo: 8 };
  $('jCount').value = defaults[f] || 24;
}
function payloadsFor(func, n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    if (func === 'sha_pow') out.push({ seed: 's' + i + Date.now(), rounds: 90000 });
    else if (func === 'is_prime') out.push({ n: 2000000011 + i * 2 });
    else if (func === 'embed') out.push({ text: 'meshpool distributed compute sample document number ' + i });
    else if (func === 'square') out.push({ x: i });
    else out.push({ seconds: 1.5, echo: i });
  }
  return out;
}
async function submitJob() {
  const func = $('jFunc').value, count = Math.max(1, +$('jCount').value || 1);
  const btn = $('jBtn'); btn.disabled = true;
  try {
    const r = await postAuthed('/api/v1/jobs', { name: func, func, payloads: payloadsFor(func, count), cpu_pct: 15, ram_mb: 128 });
    toast(`Submitted ${r.units} units across the mesh`); loadJobs(); refreshWalletSoon();
  } catch (e) { handleConsumeError(e); }
  finally { btn.disabled = false; }
}
const JOBS = new Map();
function mergeJob(j) {
  const prev = JOBS.get(j.id) || {};
  JOBS.set(j.id, { ...prev, ...j, progress: { ...(prev.progress || {}), ...(j.progress || {}) } });
}
function compactJson(v) {
  if (v == null) return '';
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  return esc(s.length > 140 ? s.slice(0, 137) + '…' : s);
}
function unitStream(j) {
  const units = j.units || [];
  const failed = units.filter(u => u.error || u.status === 'failed').slice(-3);
  const done = units.filter(u => u.status === 'done' && u.result != null).slice(-4);
  if (failed.length) {
    return `<div class="unit-stream">${failed.map(u =>
      `<div class="unit-line err"><b>${esc((u.id || '').slice(0, 8))}</b> ${esc(u.error || 'failed')}</div>`).join('')}</div>`;
  }
  if (done.length) {
    return `<div class="unit-stream">${done.map(u =>
      `<div class="unit-line ok"><b>${esc((u.id || '').slice(0, 8))}</b> ${compactJson(u.result)}</div>`).join('')}</div>`;
  }
  return `<div class="unit-stream muted">Waiting for first unit result…</div>`;
}
function jobRow(j) {
  const p = j.progress || {}, t = p.total || (j.units ? j.units.length : 1) || 1;
  const pct = (x) => (100 * x / t) + '%';
  const running = (p.running || 0) + (p.assigned || 0);
  const queued = p.pending || Math.max(0, t - (p.done || 0) - (p.failed || 0) - running);
  const state = String(j.status || 'running');
  const terminal = state.startsWith('completed');
  return `<div class="job ${terminal ? 'terminal' : ''}"><div class="h"><span class="fn">${esc(j.func || j.name || 'job')}</span>
    <span class="jid">${esc((j.id || '').slice(0, 12))}</span><span class="st ${state.includes('failure') ? 'bad' : (terminal ? 'ok' : '')}">${esc(state)}</span></div>
    <div class="pbar"><i class="done" style="width:${pct(p.done || 0)}"></i><i class="run" style="width:${pct(running)}"></i><i class="fail" style="width:${pct(p.failed || 0)}"></i></div>
    <div class="jstat"><span>${p.done || 0}/${t} done</span><span>${running} running</span>${p.failed ? `<span style="color:var(--err)">${p.failed} failed</span>` : ''}<span>${queued} queued</span></div>
    ${unitStream(j)}</div>`;
}
function renderJobs() {
  const jobs = Array.from(JOBS.values());
  const active = jobs.filter(j => j.status === 'running').length;
  $('jobCnt').textContent = active ? `${active} running` : '';
  if ($('jobLiveCnt')) $('jobLiveCnt').textContent = jobs.length ? `${active} active · ${jobs.length} recent` : '';
  $('jobs').innerHTML = jobs.length ? jobs.slice(0, 8).map(jobRow).join('') : `<div class="empty">No batch jobs yet.</div>`;
}
async function loadJobDetail(id) {
  try {
    const j = await api('/api/v1/jobs/' + encodeURIComponent(id));
    mergeJob({ ...j, progress: j.progress || progressFromUnits(j.units || []) });
    renderJobs();
  } catch (e) {}
}
function progressFromUnits(units) {
  const p = { pending: 0, assigned: 0, running: 0, done: 0, failed: 0, total: units.length };
  units.forEach(u => { if (p[u.status] != null) p[u.status] += 1; });
  return p;
}
function handleJobProgress(data) {
  if (!data || !data.id) return;
  const p = { ...data };
  p.pending = Math.max(0, (p.total || 0) - (p.done || 0) - (p.failed || 0) - (p.running || 0));
  p.assigned = 0;
  mergeJob({ id: data.id, status: 'running', progress: p });
  renderJobs();
  loadJobDetail(data.id);
}
function handleJobFinished(data) {
  if (!data || !data.id) return;
  mergeJob({ id: data.id, status: data.status || 'completed' });
  renderJobs();
  loadJobDetail(data.id);
  refreshWalletSoon();
}
async function loadJobs() {
  try {
    const d = await api('/api/v1/jobs');
    d.jobs.forEach(mergeJob);
    renderJobs();
    d.jobs.slice(0, 8).forEach(j => loadJobDetail(j.id));
  } catch (e) {}
}

// ---------- USE: hosting ----------
function depRow(d) {
  const reps = d.replicas || [];
  const healthy = reps.filter(r => r.healthy).length;
  return `<div class="job"><div class="h"><span class="fn">${esc(d.app)}</span>
    <span class="st">${healthy}/${reps.length} healthy</span></div>
    <div class="jstat">${reps.map(r => `<span class="pill" style="color:${r.healthy ? 'var(--ok)' : 'var(--err)'}">${esc(r.node_name || r.node_id.slice(0, 10))}</span>`).join('')}</div>
    <div style="margin-top:8px"><a class="btn sm" href="/app/${d.id}/" target="_blank">Open app \u2197</a>
      <button class="btn sm" onclick="showLogs('${d.id}')">Logs</button>
      <button class="btn sm d" onclick="rmDeploy('${d.id}')">Remove</button></div>
    <pre id="log-${d.id}" style="display:none;max-height:220px;overflow:auto;background:#0b0e14;color:#cbd5e1;padding:8px;border-radius:6px;margin-top:8px;font-size:12px;line-height:1.4;white-space:pre-wrap"></pre></div>`;
}
function onAppChange() { $('siteWrap').style.display = $('dApp').value === 'site' ? '' : 'none'; }
function utf8b64(s) { return btoa(unescape(encodeURIComponent(s))); }
// Upload a real multi-page app: POST the raw .zip bytes (no multipart) with the
// session header, re-authing once on 401 exactly like postAuthed does.
async function uploadSiteZip(file, replicas, envLines) {
  const buf = await file.arrayBuffer();
  let q = '?name=my-site&replicas=' + encodeURIComponent(replicas);
  for (const line of (envLines || [])) q += '&env=' + encodeURIComponent(line);
  const send = () => api('/api/v1/deployments/upload' + q, {
    method: 'POST', headers: { 'Content-Type': 'application/zip', ...sHdr() }, body: buf,
  });
  try { return await send(); }
  catch (e) {
    if (e.status === 401 && SESSION.gating) { await authLocal(); return await send(); }
    throw e;
  }
}
async function deployApp() {
  const app = $('dApp').value, replicas = Math.max(1, +$('dRep').value || 1);
  const btn = $('dBtn');
  if (app === 'site') {
    const zipInput = $('dZip'), file = zipInput && zipInput.files && zipInput.files[0];
    if (file) {
      const envEl = $('dEnv');
      const envLines = ((envEl && envEl.value) || '').split('\n')
        .map(s => s.trim()).filter(s => s.includes('='));
      btn.disabled = true;
      try {
        const res = await uploadSiteZip(file, replicas, envLines);
        toast(res && res.kind === 'proc'
          ? 'Your server app is deploying across the mesh'
          : 'Your app is deploying across the mesh');
        zipInput.value = '';
        loadDeps(); refreshWalletSoon();
      }
      catch (e) { handleConsumeError(e); }
      finally { btn.disabled = false; }
      return;
    }
  }
  let name = app + '-app', site = null;
  if (app === 'site') {
    const html = ($('dSite').value || '').trim();
    if (!html) { toast('Upload a .zip or paste some HTML to host your own site', true); return; }
    site = { 'index.html': utf8b64(html) };
    name = 'my-site';
  }
  btn.disabled = true;
  try {
    const body = { name, app, replicas };
    if (site) body.site = site;
    await postAuthed('/api/v1/deployments', body);
    toast(app === 'site' ? 'Your site is deploying across the mesh' : 'Deploying ' + replicas + ' replicas');
    loadDeps(); refreshWalletSoon();
  }
  catch (e) { handleConsumeError(e); }
  finally { btn.disabled = false; }
}
async function rmDeploy(id) { try { await api('/api/v1/deployments/' + id, { method: 'DELETE' }); loadDeps(); } catch (e) { toast(e.message, true); } }
// Show each replica's recent stdout/stderr so a deployer can debug a hosted app.
// Rendered via textContent (never innerHTML) so app-controlled log text can't inject markup.
async function showLogs(id) {
  const box = $('log-' + id);
  if (!box) return;
  if (box.style.display !== 'none') { box.style.display = 'none'; return; }  // toggle closed
  box.style.display = '';
  box.textContent = 'loading logs\u2026';
  try {
    const d = await api('/api/v1/deployments/' + id + '/logs?tail=200');
    const reps = d.replicas || [];
    if (!reps.length) { box.textContent = 'no replicas for this deployment'; return; }
    box.textContent = reps.map(r => {
      const head = `\u2500\u2500 ${r.replica_id} @ ${r.node_name || '?'} ${r.healthy ? '(healthy)' : '(unhealthy)'} \u2500\u2500`;
      const lines = (r.lines && r.lines.length) ? r.lines.join('\n') : '(no output captured \u2014 static site, or nothing logged yet)';
      return head + '\n' + lines;
    }).join('\n\n');
  } catch (e) { box.textContent = 'error: ' + e.message; }
}
async function loadDeps() {
  try {
    const d = await api('/api/v1/deployments');
    $('depCnt').textContent = d.deployments.length ? `${d.deployments.length}` : '';
    $('deps').innerHTML = d.deployments.length ? d.deployments.map(depRow).join('') : '';
  } catch (e) {}
}

// ---------- USE: pooled storage (files) ----------
const fmtBytes = (b) => {
  b = +b || 0;
  if (b >= 1073741824) return (b / 1073741824).toFixed(1) + ' GB';
  if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
  if (b >= 1024) return (b / 1024).toFixed(1) + ' KB';
  return b + ' B';
};
function fileRow(f) {
  const reps = f.replicas || [];
  const chunks = f.chunks || [];
  const st = f.status || 'stored';
  const col = st === 'stored' ? 'var(--ok)' : (st === 'lost' ? 'var(--err)' : 'var(--warn)');
  const nm = f.name || f.id.slice(0, 12);
  const dl = esc(nm).replace(/"/g, '&quot;');
  const dur = chunks.length
    ? `${chunks.length} chunks`
    : `${reps.length} cop${reps.length === 1 ? 'y' : 'ies'}`;
  return `<div class="job"><div class="h"><span class="fn">${esc(nm)}</span>
    <span class="st" style="color:${col}">${esc(st)} \u00b7 ${dur}</span></div>
    <div class="jstat"><span class="pill">${fmtBytes(f.size)}</span></div>
    <div style="margin-top:8px"><a class="btn sm" href="/api/v1/blobs/${f.id}" download="${dl}">Download \u2193</a>
      <button class="btn sm d" onclick="rmBlob('${f.id}')">Remove</button></div></div>`;
}
// Upload raw file bytes (no multipart) with the session header, re-authing once on
// 401 exactly like uploadSiteZip — the pool replicates it across nodes' spare disk.
async function uploadFile() {
  const input = $('fFile'), file = input && input.files && input.files[0];
  if (!file) { toast('Choose a file to upload to the pool', true); return; }
  const btn = $('fBtn');
  btn.disabled = true;
  try {
    const buf = await file.arrayBuffer();
    const q = '?name=' + encodeURIComponent(file.name || 'file');
    const send = () => api('/api/v1/blobs' + q, {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream', ...sHdr() }, body: buf,
    });
    let res;
    try { res = await send(); }
    catch (e) {
      if (e.status === 401 && SESSION.gating) { await authLocal(); res = await send(); }
      else throw e;
    }
    const spread = (res.chunks && res.chunks.length)
      ? (res.chunks.length + ' chunks across the pool')
      : ((res.replicas || []).length + ' machines');
    toast('Stored as ' + spread + ' — safe from any one dropping out');
    input.value = '';
    loadFiles(); refreshWalletSoon();
  }
  catch (e) { handleConsumeError(e); }
  finally { btn.disabled = false; }
}
async function rmBlob(id) { try { await api('/api/v1/blobs/' + id, { method: 'DELETE' }); loadFiles(); } catch (e) { toast(e.message, true); } }
async function loadFiles() {
  try {
    const d = await api('/api/v1/blobs');
    const files = d.blobs || [];
    $('fileCnt').textContent = files.length ? `${files.length}` : '';
    $('files').innerHTML = files.map(fileRow).join('');
  } catch (e) {}
}

// ---------- USE: inference ----------
let modelReady = false;
let AVAIL = { builtin: [], ollama: [] };
async function loadModelPicker() {
  try {
    const a = await api('/api/v1/models/available');
    AVAIL = a;
    const sel = $('modelPick'); if (!sel) return;
    const prev = sel.value;
    const opts = [];
    (a.ollama || []).forEach(o => {
      const n = o.nodes.length;
      opts.push(`<option value="${esc(o.model)}">${esc(o.model)} \u2014 Ollama \u00b7 ${n} node${n === 1 ? '' : 's'}</option>`);
    });
    (a.builtin || []).forEach(m => {
      const lbl = m === 'nano-gpt' ? `${esc(m)} \u2014 built-in demo (sharded)` : esc(m);
      opts.push(`<option value="${esc(m)}">${lbl}</option>`);
    });
    if (opts.length) sel.innerHTML = opts.join('');
    if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
    onModelPick();
  } catch (e) { /* keep the built-in default option */ }
}
function onModelPick() {
  const sel = $('modelPick'); if (!sel) return;
  const m = sel.value;
  const o = (AVAIL.ollama || []).find(x => x.model === m);
  let hint;
  if (o) {
    const n = o.nodes.length;
    hint = `Runs via Ollama on ${n} node${n === 1 ? '' : 's'} (${o.nodes.map(esc).join(', ')}) \u2014 replicated for throughput & failover.`;
  } else if (m === 'nano-gpt') {
    hint = 'Sharded layer-by-layer across the pool \u2014 bit-identical to a single machine, survives a node dying.';
  } else {
    hint = 'Sharded across the pool.';
  }
  $('modelHint').textContent = hint;
  // 'Bring your own weights' only applies to the built-in sharded net — Ollama models
  // are served from each node's local install, not from a pooled checkpoint.
  const byo = $('byoWrap');
  if (byo) {
    if (o) { byo.style.display = 'none'; const w = $('wFile'); if (w) w.value = ''; }
    else { byo.style.display = ''; }
  }
}
async function deployModel() {
  const model = ($('modelPick') && $('modelPick').value) || 'nano-gpt';
  const byo = $('byoWrap'), wIn = $('wFile');
  const wFile = byo && byo.style.display !== 'none' && wIn && wIn.files && wIn.files[0];
  $('modelBtn').disabled = true;
  $('modelState').textContent = wFile ? 'staging weights\u2026' : 'deploying\u2026';
  try {
    if (wFile) {
      const v = await uploadWeights(model, wFile);
      toast(`Deployed ${model} with your weights across ${v.stages.length} node${v.stages.length === 1 ? '' : 's'}`);
      if (wIn) wIn.value = '';
    } else {
      await postAuthed('/api/v1/models', { model });
      toast(`Deployed ${model} across the pool`);
    }
    await loadModel(); refreshWalletSoon();
  }
  catch (e) { handleConsumeError(e); $('modelBtn').disabled = false; $('modelState').textContent = ''; }
}
// Upload a trained checkpoint (.npz) as raw bytes with the session header, re-authing
// once on 401 exactly like uploadFile — the control plane stages it in the pool and
// ships each node only the slice of layers it runs.
async function uploadWeights(model, file) {
  const buf = await file.arrayBuffer();
  const send = () => api('/api/v1/models/upload?model=' + encodeURIComponent(model), {
    method: 'POST', headers: { 'Content-Type': 'application/octet-stream', ...sHdr() }, body: buf,
  });
  try { return await send(); }
  catch (e) {
    if (e.status === 401 && SESSION.gating) { await authLocal(); return await send(); }
    throw e;
  }
}
async function loadModel() {
  try {
    const d = await api('/api/v1/models');
    if (!d.deployed) { modelReady = false; $('modelBtn').style.display = ''; $('modelBtn').disabled = false; $('chatWrap').style.display = 'none'; $('topo').innerHTML = ''; return; }
    modelReady = true; $('modelBtn').style.display = 'none'; $('chatWrap').style.display = 'flex';
    const native = d.backend && d.backend !== 'meshshard';
    if (native) {
      const reps = d.replicas || d.stages.length;
      $('modelState').textContent = `${d.model} \u00b7 ${reps} replica${reps === 1 ? '' : 's'} \u00b7 ${d.backend}`;
      $('topo').innerHTML = `<div class="topo">${d.stages.map(s => `<div class="seg"><span class="lbl">${esc(s.node_name)}</span>
        <span class="blocks">full model \u00b7 ${esc(d.backend)}</span></div>`).join('')}</div>`;
    } else {
      const pooled = d.pooled ? ' \u00b7 your weights (pooled)' : '';
      $('modelState').textContent = `${d.model} \u00b7 ${d.stages.length} stages${pooled}`;
      $('topo').innerHTML = `<div class="topo">${d.stages.map(s => `<div class="seg"><span class="lbl">${esc(s.node_name)}</span>
        <span class="blocks">layers ${s.first}\u2013${s.last}</span></div>`).join('')}</div>`;
    }
  } catch (e) { modelReady = false; }
}
function addMsg(cls, html) { const m = document.createElement('div'); m.className = 'msg ' + cls; m.innerHTML = html; $('msgs').appendChild(m); $('msgs').scrollTop = 1e9; return m; }
async function sendChat() {
  const inp = $('chatIn'), text = inp.value.trim(); if (!text) return; inp.value = '';
  addMsg('u', esc(text));
  const pending = addMsg('a pending', 'thinking\u2026');
  try {
    const r = await postAuthed('/api/v1/inference/chat', { prompt: text, max_new_tokens: 64 });
    pending.className = 'msg a';
    const where = r.served_by ? esc(r.served_by) : `${r.stages.length} stage${r.stages.length === 1 ? '' : 's'}`;
    const back = r.backend ? ` \u00b7 ${esc(r.backend)}` : '';
    pending.innerHTML = esc(r.text) + `<span class="tps">${r.tokens} tokens \u00b7 ${r.tokens_per_sec.toFixed(1)} tok/s \u00b7 ${where}${back}</span>`;
    refreshWalletSoon();
  } catch (e) { pending.className = 'msg a'; pending.innerHTML = '<span style="color:var(--err)">' + esc(e.message) + '</span>'; }
  $('msgs').scrollTop = 1e9;
}

function handleConsumeError(e) {
  if (e.status === 402) toast('Out of credits — donate spare compute to earn more.', true);
  else if (e.status === 429) toast('Too many active jobs — let some finish first.', true);
  else if (e.status === 401) toast('Sign-in required. Reload the page on the machine running MeshPool.', true);
  else toast(e.message, true);
  useBanner();
}

// ---------- WALLET ----------
let WALLET = { reputation: 1 };
async function loadWallet() {
  try {
    const url = SESSION.gating ? '/api/v1/wallet' : '/api/v1/wallet?owner=default';
    const w = await api(url, { headers: sHdr() });
    WALLET = w; SESSION.balance = w.balance; renderChip();
    $('walletHero').innerHTML = `
      <div class="wcard big"><div class="k">Balance</div><div class="v">${fmtCr(w.balance)}<small> cr</small></div>
        <div class="sub">${SESSION.gating ? 'earn-to-spend active' : 'local pool · unlimited'}</div></div>
      <div class="wcard"><div class="k">Donated</div><div class="v up">+${fmtCr(w.donated)}</div><div class="sub">compute you gave</div></div>
      <div class="wcard"><div class="k">Used</div><div class="v down">-${fmtCr(w.consumed)}</div><div class="sub">compute you borrowed</div></div>
      <div class="wcard"><div class="k">Granted</div><div class="v">${fmtCr(w.granted)}</div><div class="sub">starter + bonuses</div></div>
      <div class="wcard"><div class="k">Reputation</div><div class="v">${(w.reputation ?? 1).toFixed(2)}</div><div class="sub">trust across your nodes</div></div>`;
    loadLedger();
  } catch (e) {
    $('walletHero').innerHTML = `<div class="banner warn">Wallet needs a session. Reload on the machine running MeshPool.</div>`;
  }
}
function txnRow(e) {
  const isCredit = e.kind === 'donated' || e.kind === 'grant';
  const label = { donated: 'Donated compute', consumed: 'Used compute', grant: 'Grant' }[e.kind] || e.kind;
  const when = new Date(e.ts_ms).toLocaleString();
  return `<div class="txn"><div class="tl"><span class="kind ${e.kind}">${label}</span>
    <span class="det">${esc(e.detail || '')}</span></div>
    <div class="tr"><span class="amt ${isCredit ? 'up' : 'down'}">${isCredit ? '+' : '-'}${fmtCr(e.compute_units)}</span>
    <span class="when">${when}</span></div></div>`;
}
async function loadLedger() {
  try {
    const d = await api('/api/v1/ledger');
    // Explore: per-machine contribution + trust
    if ($('ledger')) {
      $('ledger').innerHTML = d.by_node.length ? d.by_node.map(r =>
        `<tr><td>${esc(r.name)}</td><td class="num">${fmtCr(r.donated)}</td><td class="num">${fmtCr(r.consumed)}</td><td class="num">${(r.trust ?? 1).toFixed(2)}</td></tr>`).join('')
        : `<tr><td colspan="4" class="empty" style="padding:14px">No contributions recorded yet.</td></tr>`;
      const s = d.summary;
      $('trustTier').innerHTML = SESSION.tier >= 2
        ? `Global mesh (Tier 2): reputation + verification + earn-to-spend credits are active. Across the pool: <b>${fmtCr(s.donated)}</b> donated, <b>${fmtCr(s.consumed)}</b> used, <b>${fmtCr(s.granted)}</b> granted.`
        : `Local pool (Tier ${SESSION.tier}): full mutual trust — no verification overhead. Fair-share ledger is live.`;
    }
    // Wallet: my recent transactions + leaderboard
    if ($('txns')) {
      const mine = (d.recent || []).filter(e => e.owner === SESSION.owner || (!SESSION.gating && e.owner === 'default'));
      $('txns').innerHTML = mine.length ? mine.map(txnRow).join('') : `<div class="empty">No transactions yet. Run something in <a onclick="showView('use')">Use</a>, or donate in <a onclick="showView('contribute')">Contribute</a>.</div>`;
    }
    if ($('leaderboard')) {
      const top = (d.owners || []).slice().sort((a, b) => (b.donated + b.granted) - (a.donated + a.granted)).slice(0, 6);
      $('leaderboard').innerHTML = top.length ? top.map(o =>
        `<tr class="${o.owner === SESSION.owner ? 'me' : ''}"><td>${esc(shortOwner(o.owner))}</td><td class="num">${fmtCr(o.donated)}</td><td class="num">${fmtCr(o.consumed)}</td></tr>`).join('')
        : `<tr><td colspan="3" class="empty" style="padding:12px">No contributors yet.</td></tr>`;
    }
  } catch (e) {}
}
let walletT = 0;
function refreshWalletSoon() { clearTimeout(walletT); walletT = setTimeout(loadWallet, 1500); }

// ---------- WALLET: account linking ----------
async function startLink() {
  try {
    const d = await post('/api/v1/account/link/start', { session: SESSION.token });
    $('linkArea').innerHTML = `
      <div class="hint">Link code: <b class="mono">${esc(d.link_code)}</b> — enter a handle to finish.</div>
      <input id="linkHandle" placeholder="choose a handle (e.g. alice)" style="margin-top:8px">
      <button class="btn p block" onclick="completeLink('${d.pending_id}','${d.link_code}')">Finish linking</button>`;
  } catch (e) { toast(e.message, true); }
}
async function completeLink(pendingId, code) {
  const handle = ($('linkHandle').value || 'user').trim();
  try {
    const d = await api('/api/v1/account/link/complete', {
      method: 'POST', headers: { 'Content-Type': 'application/json', ...sHdr() },
      body: JSON.stringify({ pending_id: pendingId, proof: `${code}:${handle}` }),
    });
    toast('Linked as ' + d.handle);
    await authLocal(); loadWallet();
    $('linkArea').innerHTML = `<div class="banner">Linked as <b>@${esc(d.handle)}</b>. Credits &amp; reputation now pool across your machines.</div>`;
  } catch (e) { toast(e.message, true); }
}

// ---------- CONTRIBUTE: onboarding ----------
async function loadOnboarding() {
  try {
    const d = await api('/api/v1/onboarding');
    $('pairCode').textContent = d.invite_token;
    $('joinWin').textContent = d.join_windows || d.join_command;
    $('joinNix').textContent = d.join_unix || d.join_command;
  } catch (e) {}
}

// ---------- EXPLORE: events (SSE) ----------
const EMOJI = { job_submitted: '\u2699\ufe0f', job_progress: '\ud83d\udcc8', job_finished: '\u2705', units_requeued: '\u267b\ufe0f', node_status: '\ud83d\udd0c', node_offline: '\u26a0\ufe0f', deploy: '\ud83c\udf10', model_deployed: '\ud83e\udde0' };
function logEvent(kind, data) {
  if (kind === 'heartbeat' || kind === 'hello') return;
  const log = $('log'); if (!log) return;
  const e = document.createElement('div'); e.className = 'e';
  const t = new Date().toLocaleTimeString();
  e.innerHTML = `<span class="t">${t}</span> ${EMOJI[kind] || '\u2022'} <b>${kind}</b> ${data ? esc(JSON.stringify(data)).slice(0, 80) : ''}`;
  log.insertBefore(e, log.firstChild);
  while (log.children.length > 40) log.removeChild(log.lastChild);
}
function connectSSE() {
  const es = new EventSource('/api/v1/events');
  es.onopen = () => { $('dot').className = 'dot live'; $('connTxt').textContent = 'live'; };
  es.onerror = () => { $('dot').className = 'dot down'; $('connTxt').textContent = 'reconnecting\u2026'; };
  es.onmessage = (m) => {
    let d; try { d = JSON.parse(m.data); } catch (e) { return; }
    logEvent(d.kind, d.data);
    if (d.kind === 'job_progress') handleJobProgress(d.data);
    if (d.kind === 'job_finished') handleJobFinished(d.data);
    if (['job_submitted', 'units_requeued'].includes(d.kind)) { loadJobs(); refreshWalletSoon(); }
    if (['node_status', 'node_offline', 'heartbeat'].includes(d.kind)) loadClusterThrottled();
    if (['node_status', 'node_offline'].includes(d.kind)) loadModelPicker();
    if (['deploy', 'replica_health'].includes(d.kind)) loadDeps();
    if (d.kind && d.kind.indexOf('blob_') === 0) loadFiles();
  };
}
let clusterT = 0;
function loadClusterThrottled() { const now = Date.now(); if (now - clusterT > 1500) { clusterT = now; loadCluster(); } }

// ---------- boot ----------
function refreshAll() { loadCluster(); loadJobs(); loadDeps(); loadFiles(); loadModel(); useBanner(); }
(async function boot() {
  await authLocal();
  describeFunc();
  loadOnboarding();
  refreshAll();
  loadModelPicker();
  loadWallet();
  connectSSE();
  setInterval(loadCluster, 3500);
  setInterval(() => { const v = document.querySelector('.view.active'); if (v && v.id === 'view-use') loadJobs(); }, 2000);
})();
