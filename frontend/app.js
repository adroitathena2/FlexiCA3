/* =============================================================================
 * Paytriq — live-trace console
 * Vanilla ES2018. No frameworks, no bundler, no npm, no secrets in this file.
 * Loaded as a classic script so it also runs from file:// (ES modules do not).
 *
 * Design rules this file obeys, in order of priority:
 *   1. Never render a plausible-looking value the backend did not send.
 *      Missing -> an explicit "not sent" marker, never a zero, dash or guess.
 *   2. Never swallow a fetch error. Every request either succeeds, or paints a
 *      message where the data would have been, and/or raises the banner.
 *   3. Never put backend strings into innerHTML. All DOM is built with
 *      createElement + textContent.
 *   4. Real, fixture-backed and unavailable are three different things and are
 *      always styled differently.
 * ========================================================================== */
(function () {
  'use strict';

  /* ============================== constants ============================== */

  var LS = {
    base: 'paytriq.apiBase',
    run: 'paytriq.runId',
    event: 'paytriq.eventId',
    history: 'paytriq.runHistory'
  };

  var DEFAULT_BASE = 'http://localhost:18780';
  var START_CMD = 'uvicorn api.main:app --reload --port 18780';

  var AGENTS = {
    A1: { label: 'Discovery', short: 'Discovery' },
    A2: { label: 'Pricing', short: 'Pricing' },
    A3: { label: 'Outreach', short: 'Outreach' },
    A4: { label: 'Contract', short: 'Contract' },
    A5: { label: 'Compliance', short: 'Compliance' },
    A6: { label: 'Audit', short: 'Audit' },
    A7: { label: 'Arbiter', short: 'Arbiter' },
    ENV: { label: 'Sponsor (simulated counterparty)', short: 'sponsor' }
  };

  var KINDS = ['agent', 'llm', 'tool', 'decision', 'handoff', 'human', 'error'];
  var DISPLAY_KINDS = KINDS.concat(['unknown']);

  var KIND_COLOR = {
    agent: 'var(--k-agent)', llm: 'var(--k-llm)', tool: 'var(--k-tool)',
    decision: 'var(--k-decision)', handoff: 'var(--k-handoff)',
    human: 'var(--k-human)', error: 'var(--k-error)', unknown: '#5c6c7b'
  };

  /* Deterministic-fallback decision sources. A decision that names one of these
   * did not come from a model. The UI must never let that read as a model call. */
  var FALLBACK_SOURCES = { rules: 1, replay: 1 };
  var MODEL_SOURCES = { clef: 1, gemini: 1 };

  var enc = function (v) { return encodeURIComponent(String(v)); };
  var now = function () { return Date.now(); };

  var STAGES = [
    { key: 'discover', label: 'Discover', agent: 'A1', blurb: 'A1 · evidence-backed sponsor leads',
      path: function (id) { return '/api/events/' + enc(id) + '/discover'; } },
    { key: 'propose', label: 'Propose', agent: 'A2', blurb: 'A2 · priced tiers',
      path: function (id) { return '/api/events/' + enc(id) + '/propose'; } },
    { key: 'outreach', label: 'Outreach', agent: 'A3', blurb: 'A3 · outbound mail is gated',
      path: function (id) { return '/api/events/' + enc(id) + '/outreach'; } },
    { key: 'reply', label: 'Reply', agent: 'ENV', blurb: 'sponsor replies · A3 classifies intent',
      path: function (id) { return '/api/events/' + enc(id) + '/reply'; } },
    { key: 'contract', label: 'Contract', agent: 'A4', blurb: 'A4 · memorandum of understanding',
      path: function (id) { return '/api/events/' + enc(id) + '/contract'; } },
    { key: 'compliance', label: 'Compliance', agent: 'A5', blurb: 'A5 · promises vs evidence',
      path: function (id) { return '/api/events/' + enc(id) + '/compliance'; } },
    { key: 'audit', label: 'Audit', agent: 'A6', blurb: 'A6 · ROI report, every figure sourced',
      path: function (id) { return '/api/events/' + enc(id) + '/audit'; } }
  ];

  var PRESETS = [
    { key: 'yes', label: 'yes', text: 'Yes, that works for us. Please send the MoU across and we will sign this week.' },
    { key: 'pushback', label: 'pushback', text: 'We like the event, but 150000 is well over our campus marketing budget this term. Can you come down to 90000 and drop the panel from the deliverables?' },
    { key: 'interested', label: 'interested', text: 'Interested in principle. Can you share the audience profile and one past sponsorship report before we commit to anything?' },
    { key: 'no', label: 'no', text: 'No thank you. We have already signed with another college for this slot, so we will pass this time.' },
    { key: 'ambiguous', label: 'ambiguous', trap: true,
      text: 'Yesterday we thought the price was too high, but lets proceed.',
      note: 'The trap: this sentence contains the pushback keyword “too high” and the acceptance keyword “lets proceed”. A keyword router reads whichever it sees first and gets it wrong. The intent classifier has to resolve the temporal clause — the objection is withdrawn.' }
  ];

  var DEMO_EVENT = {
    name: 'TechFest 2026',
    location: 'VIT Pune, Karad Road',
    footfall: 5000,
    date: '2026-02-14',
    audience: 'Second and third year engineering students, 62% male, hiring-season heavy',
    categories_wanted: 'cafe, coaching, bookstore, printing',
    deliverables_offered: 'main-stage logo backdrop, 4000 Instagram story mentions, 2 reels, expo stall',
    budget_inr: 150000,
    contact_email: 'events@vit.edu.in'
  };

  /* Fields the server owns. They are not part of the create request. */
  var SERVER_FIELDS = { event_id: 1, created_at: 1, updated_at: 1, id: 1 };

  var FALLBACK_SCHEMA = {
    title: 'EventProfile',
    description: 'Built-in fallback: the backend did not serve a schema, so these field names are this console’s own declaration of the contract.',
    required: ['name', 'location', 'footfall', 'date', 'audience'],
    properties: {
      name: { type: 'string', minLength: 1, maxLength: 200, description: 'Name of the student event being sponsored.' },
      location: { type: 'string', minLength: 1, maxLength: 200, description: 'Campus / city the event runs in.' },
      footfall: { type: 'integer', minimum: 0, maximum: 10000000, description: 'Expected attendance.' },
      date: { type: 'string', minLength: 4, maxLength: 40, description: 'Event date, as the organiser writes it.' },
      audience: { type: 'string', minLength: 1, maxLength: 500, description: 'Who attends — this is what sponsors buy.' },
      categories_wanted: { type: 'array', items: { type: 'string' }, description: 'Comma-separated list of brand categories wanted.' },
      deliverables_offered: { type: 'array', items: { type: 'string' }, description: 'Comma-separated list of what the organiser can offer.' },
      budget_inr: { type: 'number', minimum: 0, description: 'Optional organiser budget in INR.' },
      contact_email: { type: 'string', description: 'Optional organiser contact address.' }
    }
  };

  /* ================================ state ================================ */

  var state = {
    base: DEFAULT_BASE,
    runId: '',
    eventId: '',
    conn: 'idle',
    paused: false,
    buffer: [],
    events: [],
    seen: Object.create(null),
    dupSuppressed: 0,
    malformed: 0,
    /* Highest trace seq received on the current stream. Sent back as
     * ?last_event_id= on a manual reconnect so the server resumes instead of
     * replaying; reset whenever the run changes. */
    lastSeq: null,
    filters: DISPLAY_KINDS.slice(),
    decisions: [],
    gates: [],
    approvals: Object.create(null),
    parked: null,
    threadId: '',
    stageState: {},
    health: null,
    summary: null,
    board: null,
    tree: '',
    disputes: [],
    lessons: [],
    handoffs: [],
    coordLoadedFor: '',
    schema: null,
    schemaFromBackend: false,
    formFields: [],
    gen: 0
  };

  var es = null;
  var refreshTimer = 0;
  var reduceMotion = false;

  /* Last successful load time per read-only panel, and whether the newest
   * refresh failed. A panel that still shows the previous snapshot must say so:
   * stale data presented as current is the failure mode this project exists to
   * avoid, and it applies to the UI as much as to the trace. */
  var fetchedAt = Object.create(null);
  var stale = Object.create(null);

  function markFresh(key) { fetchedAt[key] = Date.now(); stale[key] = false; }
  function markStale(key) { if (fetchedAt[key]) stale[key] = true; }

  function staleNote(key, what) {
    if (!stale[key]) return null;
    return el('p', { class: 'note note--warn' }, [
      'stale - not current. This is the last ' + what + ' the backend sent, at '
      + (fetchedAt[key] ? fmtClock(new Date(fetchedAt[key])) : 'an unknown time')
      + '. The most recent refresh failed, so this does not reflect the run as it stands now.'
    ]);
  }

  /* ============================== DOM helpers ============================= */

  function $(sel, root) { return (root || document).querySelector(sel); }

  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) {
      for (var k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        var v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'text') n.textContent = String(v);
        else if (k === 'class') n.className = v;
        else if (k === 'title') n.title = String(v);
        else if (k.indexOf('on') === 0 && typeof v === 'function') n.addEventListener(k.slice(2), v);
        else if (v === true) n.setAttribute(k, '');
        else n.setAttribute(k, String(v));
      }
    }
    append(n, kids);
    return n;
  }

  function append(parent, kids) {
    if (kids === null || kids === undefined || kids === false) return parent;
    if (Array.isArray(kids)) { kids.forEach(function (c) { append(parent, c); }); return parent; }
    parent.appendChild(kids instanceof Node ? kids : document.createTextNode(String(kids)));
    return parent;
  }

  function clear(node) { if (node) node.replaceChildren(); return node; }
  function frag() { return document.createDocumentFragment(); }

  function on(node, ev, fn) { if (node) node.addEventListener(ev, fn); return node; }

  /* CSS.escape is absent on a few older browsers; field names come from a JSON
   * schema, so escaping them is not optional. */
  var cssEscape = (window.CSS && typeof window.CSS.escape === 'function')
    ? function (s) { return window.CSS.escape(s); }
    : function (s) { return String(s).replace(/[^a-zA-Z0-9_-]/g, function (ch) { return '\\' + ch; }); };

  /* ============================ value formatting =========================== */

  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function str(v) { return (typeof v === 'string' || typeof v === 'number') && String(v).trim() !== '' ? String(v) : null; }
  function truthy(v) {
    if (v === true) return true;
    if (typeof v === 'number') return v !== 0;
    if (typeof v === 'string') return ['1', 'true', 'yes', 'on'].indexOf(v.trim().toLowerCase()) >= 0;
    return false;
  }
  function numOrNull(v) {
    if (isNum(v)) return v;
    if (typeof v === 'string' && v.trim() !== '' && isFinite(Number(v))) return Number(v);
    return null;
  }

  /* The single most important helper in this file: a value the backend did not
   * send renders as an explicit marker, never as a plausible default. */
  function absent(label, title) {
    return el('span', {
      class: 'absent',
      title: title || 'the backend did not send this field in its response — nothing has been substituted for it',
      text: label || 'not sent'
    });
  }

  function fmtNum(v, digits) {
    var n = numOrNull(v);
    if (n === null) return null;
    return n.toFixed(digits === undefined ? 0 : digits);
  }
  function fmtMs(ms) {
    var n = numOrNull(ms);
    if (n === null) return null;
    if (n < 1) return n.toFixed(3) + ' ms';
    if (n < 1000) return Math.round(n) + ' ms';
    if (n < 60000) return (n / 1000).toFixed(2) + ' s';
    var m = Math.floor(n / 60000);
    return m + ' m ' + Math.round((n % 60000) / 1000) + ' s';
  }
  function fmtBytesish(n) { return String(n); }

  function parseTs(ts) {
    if (typeof ts === 'number' && isFinite(ts)) return new Date(ts > 1e12 ? ts : ts * 1000);
    if (typeof ts === 'string' && ts) { var d = new Date(ts); if (!isNaN(d.getTime())) return d; }
    return null;
  }
  function pad(n, w) { var s = String(n); while (s.length < (w || 2)) s = '0' + s; return s; }
  function fmtClock(d) {
    return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds()) + '.' + pad(d.getMilliseconds(), 3);
  }
  function fmtStamp(d) { return d ? d.toISOString().replace('T', ' ').replace('Z', 'Z') : null; }
  function ago(d) {
    if (!d) return null;
    var s = (now() - d.getTime()) / 1000;
    if (s < 1) return 'just now';
    if (s < 60) return s.toFixed(1) + 's ago';
    return Math.floor(s / 60) + 'm ' + Math.floor(s % 60) + 's ago';
  }

  function attrs(ev) {
    var a = ev && ev.attributes;
    if (typeof a === 'string') { try { a = JSON.parse(a); } catch (e) { a = null; } }
    return (a && typeof a === 'object' && !Array.isArray(a)) ? a : {};
  }
  function pick(obj, names) {
    for (var i = 0; i < names.length; i++) {
      if (obj && obj[names[i]] !== undefined && obj[names[i]] !== null) return obj[names[i]];
    }
    return undefined;
  }

  /* ============================== persistence ============================= */

  function lsGet(key, fallback) {
    try {
      var v = window.localStorage.getItem(key);
      return v === null ? fallback : v;
    } catch (e) { return fallback; }
  }
  function lsSet(key, value) {
    try { window.localStorage.setItem(key, value); return true; }
    catch (e) { return false; }
  }
  function lsDel(key) { try { window.localStorage.removeItem(key); } catch (e) { /* ignore */ } }

  /* ============================ network + banner ========================== */

  function apiUrl(path) {
    var base = (state.base || '').replace(/\/+$/, '');
    if (!base) return null;
    return base + (path.charAt(0) === '/' ? path : '/' + path);
  }

  /* Every request in this app goes through here. It never throws; it returns a
   * result object that always says what kind of failure happened, so a caller
   * cannot accidentally treat "no connection" as "no data". */
  /* Every request records which API base generation it was issued under. When
   * the base changes, in-flight responses from the old host are marked
   * `superseded` and paint nothing: a failure against a host you have already
   * walked away from is not evidence about the current one, and letting it
   * through is how a recovered console keeps claiming to be broken. */
  function request(path, opts) {
    opts = opts || {};
    var gen = state.gen;
    var url = apiUrl(path);
    if (!url) {
      return Promise.resolve({ ok: false, kind: 'config', status: 0, json: null, text: '',
        url: path, error: 'No API base URL is configured', superseded: false });
    }
    var ctrl = typeof AbortController === 'function' ? new AbortController() : null;
    var timer = ctrl ? setTimeout(function () { ctrl.abort(); }, opts.timeoutMs || 25000) : null;
    var init = { method: opts.method || 'GET', cache: 'no-store', credentials: 'omit' };
    if (opts.body !== undefined) {
      init.headers = { 'Content-Type': 'application/json' };
      init.body = JSON.stringify(opts.body);
    }
    if (ctrl) init.signal = ctrl.signal;
    return fetch(url, init).then(function (res) {
      return res.text().then(function (text) {
        var json = null;
        if (text) { try { json = JSON.parse(text); } catch (e) { json = null; } }
        return { ok: res.ok, status: res.status, json: json, text: text, url: url,
          kind: res.ok ? 'ok' : 'http', superseded: gen !== state.gen };
      });
    }).catch(function (err) {
      var timedOut = err && (err.name === 'AbortError' || /abort/i.test(err.message || ''));
      return { ok: false, status: 0, json: null, text: '', url: url,
        kind: timedOut ? 'timeout' : 'network',
        error: (err && err.message) || String(err), superseded: gen !== state.gen };
    }).then(function (r) {
      if (timer) clearTimeout(timer);
      return r;
    });
  }

  function superseded(res) { return !!(res && res.superseded); }

  function hostOf(url) {
    if (!url) return '(no URL)';
    try { return new URL(url, window.location.href).host; } catch (e) { return url; }
  }

  function shortDetail(json) {
    if (json === null || json === undefined) return '';
    var d = pick(json, ['detail', 'message', 'error', 'reason', 'title']);
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      return d.map(function (item) {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object') {
          var loc = Array.isArray(item.loc) ? item.loc.join('.') : '';
          return (loc ? loc + ': ' : '') + (item.msg || JSON.stringify(item).slice(0, 120));
        }
        return String(item);
      }).join('; ');
    }
    if (typeof json === 'object') return JSON.stringify(json).slice(0, 220);
    return String(json);
  }

  function describeFailure(res, what) {
    switch (res.kind) {
      case 'network':
        return what + ' could not reach ' + hostOf(res.url) + ' — the request never completed (' +
          (res.error || 'network error') + '). This is a connection failure, not an empty result.';
      case 'timeout':
        return what + ' timed out before the backend answered. It may be blocked, still starting, or not running.';
      case 'config':
        return 'No API base URL is set, so ' + what + ' was never attempted.';
      case 'http':
        return what + ' returned HTTP ' + res.status + (res.json ? ' — ' + shortDetail(res.json) : '');
      default:
        return what + ' failed.';
    }
  }

  /* ---- issue register: one banner, many problems, highest severity wins ---- */

  var issues = Object.create(null);
  var SEV = { bad: 3, warn: 2, info: 1 };
  var bannerDismissed = false;

  function setIssue(key, level, title, detail, cmd) {
    issues[key] = { level: level, title: title, detail: detail, cmd: cmd || null };
    bannerDismissed = false;
    renderBanner();
  }
  function clearIssue(key) { if (issues[key]) { delete issues[key]; renderBanner(); } }

  function renderBanner() {
    var host = $('#global-banner');
    var keys = Object.keys(issues);
    if (!keys.length || bannerDismissed) { host.hidden = true; return; }
    keys.sort(function (a, b) { return (SEV[issues[b].level] || 0) - (SEV[issues[a].level] || 0); });
    var top = issues[keys[0]];
    host.hidden = false;
    host.className = 'global-banner' + (top.level === 'bad' ? '' : ' global-banner--' + top.level);
    $('#global-banner-title').textContent = top.title + (keys.length > 1 ? '  (' + keys.length + ' problems)' : '');
    $('#global-banner-detail').textContent = top.detail + (keys.length > 1 ? '  ·  also: ' + keys.slice(1).length + ' more' : '');
    $('#global-banner-cmd').textContent = top.cmd || '';
    $('#global-banner-cmd').parentNode.hidden = !top.cmd;
  }

  /* Raise a banner for connectivity-level failures only. A 403 gated or a 422
   * validation error is a real answer from a live backend: it is rendered where
   * the data would have been, and does not claim the service is down. */
  function reportNetFailure(res, what, issueKey) {
    if (res.kind === 'ok') return false;
    if (res.kind === 'http') return false;
    setIssue(issueKey || ('net:' + what), 'bad',
      'Backend unreachable — ' + (res.url || what),
      describeFailure(res, what) + '  Everything this page shows from that call is missing, not empty.',
      START_CMD);
    return true;
  }

  function toast(msg, kind) {
    var t = el('div', { class: 'toast' + (kind ? ' toast--' + kind : ''), text: msg });
    document.body.appendChild(t);
    setTimeout(function () { t.remove(); }, 3600);
  }

  /* ================================ config ================================ */

  function restoreConfig() {
    var saved = lsGet(LS.base, null);
    if (saved) state.base = saved;
    else if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {
      // Served from the FastAPI static mount: same origin is the right default.
      state.base = window.location.origin;
    } else state.base = DEFAULT_BASE;
    state.runId = lsGet(LS.run, '') || '';
    state.eventId = lsGet(LS.event, '') || '';
    $('#api-base').value = state.base;
    $('#run-id').value = state.runId;
  }

  function applyBase(raw) {
    var v = (raw === undefined ? $('#api-base').value : raw).trim();
    state.base = v.replace(/\/+$/, '');
    $('#api-base').value = state.base;
    lsSet(LS.base, state.base);
    /* New host: every outstanding complaint about the old one is void. */
    state.gen++;
    issues = Object.create(null);
    bannerDismissed = false;
    renderBanner();
    disconnectTrace();
    refreshHealth();
    return state.base;
  }

  /* ================================= health ================================ */

  function deepFind(root, re, maxDepth) {
    maxDepth = maxDepth === undefined ? 4 : maxDepth;
    var queue = [{ v: root, path: '', d: 0 }];
    while (queue.length) {
      var cur = queue.shift();
      if (!cur.v || typeof cur.v !== 'object') continue;
      var keys = Object.keys(cur.v);
      for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        var v = cur.v[k];
        var p = cur.path ? cur.path + '.' + k : k;
        if (re.test(k)) return { value: v, path: p };
        if (v && typeof v === 'object' && cur.d < maxDepth) queue.push({ v: v, path: p, d: cur.d + 1 });
      }
    }
    return null;
  }

  function nameFromDecisionish(v) {
    if (typeof v === 'string') return v;
    if (v && typeof v === 'object') {
      var n = pick(v, ['name', 'backend', 'active', 'source', 'kind', 'provider', 'engine', 'mode']);
      if (typeof n === 'string') return n;
    }
    return null;
  }

  function sourceTone(name, available) {
    if (available === false) return 'bad';
    if (!name) return 'unknown';
    var n = name.toLowerCase();
    if (FALLBACK_SOURCES[n]) return 'warn';
    if (MODEL_SOURCES[n]) return 'good';
    if (/live|ready|available|ok|healthy|connected/.test(n)) return 'good';
    if (/fixture|cached|offline|stub|mock|fallback|replay|degraded|unavailable/.test(n)) return 'warn';
    if (/down|failed|error|missing|absent/.test(n)) return 'bad';
    return 'unknown';
  }

  function toolTone(v) {
    if (v === true) return 'good';
    if (v === false) return 'bad';
    var s = str(v);
    if (!s) return 'unknown';
    var n = s.toLowerCase();
    if (/^(ok|live|ready|connected|available|up)$/.test(n)) return 'good';
    if (/fixture|cached|stub|mock|offline/.test(n)) return 'warn';
    if (/unavailable|failed|error|down|absent|blocked/.test(n)) return 'bad';
    return 'unknown';
  }

  function ht(key, value, tone, note) {
    return el('span', { class: 'ht ht--' + (tone || 'unknown'), title: (note || key + ': ' + value) },
      [el('span', { class: 'ht__dot' }), el('span', { class: 'ht__k', text: key }),
       el('span', { class: 'ht__v', text: value }), note ? el('span', { class: 'ht__note', text: note }) : null]);
  }

  function renderHealthStrip(h, res) {
    var host = clear($('#health-strip'));
    if (!h || !res || !res.ok) {
      host.appendChild(ht('health', res ? 'unreachable' : 'not checked', res ? 'bad' : 'unknown',
        res ? 'GET ' + (res.url || '/health') + ' did not answer' : 'press ↻ health'));
      return;
    }
    var overall = nameFromDecisionish(pick(h, ['status', 'state', 'overall']));
    var chips = [];

    if (overall) chips.push(ht('service', overall, /^(ok|up|healthy|ready|alive|degraded)$/i.test(overall)
      ? (/degraded/i.test(overall) ? 'warn' : 'good') : 'unknown', 'as reported by /health'));

    var dec = deepFind(h, /^(decision_?(backend|name)?|backend|engine)$/i, 3);
    var decName = dec ? nameFromDecisionish(dec.value) : null;
    var decAvail = deepFind(h, /decision.*(available|ready|healthy|ok)$/i, 3);
    var avail = decAvail && typeof decAvail.value === 'boolean' ? decAvail.value
      : (typeof pick(h, ['decision_available']) === 'boolean' ? h.decision_available : undefined);
    chips.push(ht('decide', decName || 'not reported', sourceTone(decName, avail),
      dec ? dec.path : '/health did not name a decision backend'));

    var mode = deepFind(h, /^(run_?mode|runmode)$/i, 2) || deepFind(h, /^mode$/i, 1);
    var modeName = mode ? nameFromDecisionish(mode.value) : null;
    chips.push(ht('run mode', modeName || 'not reported',
      sourceTone(modeName), mode ? mode.path : '/health did not report a run mode'));

    var tools = deepFind(h, /^tools?(_status|_mode)?$/i, 2);
    var tally = { good: 0, warn: 0, bad: 0, unknown: 0 };
    var toolChips = [];
    if (tools) {
      var tv = tools.value;
      var pairs = [];
      if (Array.isArray(tv)) {
        tv.forEach(function (t, i) {
          var nm = (t && typeof t === 'object') ? (str(pick(t, ['name', 'tool', 'id'])) || 'tool[' + i + ']') : ('tool[' + i + ']');
          pairs.push([nm, (t && typeof t === 'object') ? (pick(t, ['status', 'mode', 'state']) || pick(t, ['live', 'available']) || t.source) : t]);
        });
      } else if (tv && typeof tv === 'object') {
        Object.keys(tv).forEach(function (k) {
          var v = tv[k];
          if (v && typeof v === 'object') v = pick(v, ['status', 'mode', 'state', 'source']) || pick(v, ['live', 'available']);
          pairs.push([k, v]);
        });
      } else pairs.push(['tools', tv]);

      pairs.forEach(function (p) {
        var tone = toolTone(p[1]);
        tally[tone]++;
        toolChips.push(ht(String(p[0]).slice(0, 14), str(p[1]) === null ? 'not reported' : String(p[1]), tone));
      });
    }
    var agg = tally.good + tally.warn + tally.bad + tally.unknown;
    chips.push(ht('tools', agg ? (tally.good + ' live / ' + tally.warn + ' fixture / ' + tally.bad + ' down')
      : 'not reported', !agg ? 'unknown' : (tally.bad ? 'bad' : (tally.warn ? 'warn' : 'good')),
      agg ? (tools.path + ' — per-tool status below') : '/health did not report tool status'));

    var broken = [];
    toolChips.forEach(function (c) {
      var cls = c.className;
      if (/ht--bad|ht--warn/.test(cls)) broken.push(c.textContent.replace(/\s+/g, ' ').trim());
    });
    if (tally.bad || tally.warn) {
      setIssue('health-degraded', tally.bad ? 'warn' : 'warn',
        'Subsystem degraded: ' + tally.bad + ' tool(s) unavailable, ' + tally.warn + ' fixture-backed',
        '/health reports: ' + (broken.length ? broken.join(' · ') : 'no named tool') +
        '. Results from those tools are fixture-backed or absent — they are labelled as such wherever they appear, and no value is shown for them that the backend did not send.',
        null);
    } else clearIssue('health-degraded');

    chips.forEach(function (c) { host.appendChild(c); });
    toolChips.forEach(function (c) { host.appendChild(c); });
  }

  function refreshHealth() {
    return request('/health', { timeoutMs: 7000 }).then(function (res) {
      if (superseded(res)) return res;
      if (res.kind === 'ok') { clearIssue('health'); state.health = res.json || {}; }
      else { state.health = null; reportNetFailure(res, 'GET /health', 'health'); }
      renderHealthStrip(state.health, res);
      return res;
    });
  }

  /* ============================== event form ============================== */

  function fieldsFromSchema(schema) {
    var props = (schema && schema.properties) || {};
    var required = (schema && Array.isArray(schema.required)) ? schema.required : [];
    var out = [];
    Object.keys(props).forEach(function (key) {
      var p = props[key] || {};
      if (SERVER_FIELDS[key]) return;
      if (p.readOnly === true) return;
      if (p.format === 'date-time') return;      // server timestamp
      var t = p.type;
      if (Array.isArray(t)) t = t.indexOf('null') >= 0 ? t.filter(function (x) { return x !== 'null'; })[0] : t[0];
      var kind = 'text';
      if (t === 'array') kind = 'csv';
      else if (t === 'integer' || t === 'number') kind = 'number';
      else if (Array.isArray(p.enum) && p.enum.length) kind = 'enum';
      out.push({
        key: key, kind: kind, required: required.indexOf(key) >= 0,
        hint: str(p.description) || str(p.title) || null,
        min: p.minimum, max: p.maximum, maxLength: p.maxLength, minLength: p.minLength,
        enum: p.enum || null, def: p.default
      });
    });
    if (!out.length) throw new Error('schema declared no usable properties');
    return out;
  }

  function buildForm(fields, fromBackend) {
    state.formFields = fields;
    var host = clear($('#event-form-host'));
    var rowA = el('div', { class: 'form__row' });
    var rowB = el('div', { class: 'form__row' });

    fields.forEach(function (f, index) {
      var input;
      if (f.kind === 'enum') {
        input = el('select', { name: f.key });
        input.appendChild(el('option', { value: '', text: '—' }));
        f.enum.forEach(function (v) { input.appendChild(el('option', { value: String(v), text: String(v) })); });
      } else if (f.kind === 'csv') {
        input = el('input', { type: 'text', name: f.key, placeholder: 'comma separated', spellcheck: 'false' });
      } else if (f.kind === 'number') {
        input = el('input', { type: 'number', name: f.key, step: f.key === 'footfall' ? '1' : 'any',
          min: f.min !== undefined ? f.min : null, max: f.max !== undefined ? f.max : null });
      } else {
        input = el('input', { type: 'text', name: f.key,
          maxlength: f.maxLength !== undefined ? f.maxLength : null,
          minlength: f.minLength !== undefined ? f.minLength : null, spellcheck: 'false' });
      }
      if (f.required) input.required = true;
      if (f.def !== undefined && f.def !== null) input.value = String(f.def);

      var label = el('label', { class: 'f-label' },
        [el('span', { class: 'f-label__row' }, [
          el('span', { class: 'f-label__name', text: f.key }),
          f.required ? el('span', { class: 'f-label__req', text: 'required' })
                     : el('span', { class: 'f-label__opt', text: 'optional' }),
          el('span', { class: 'f-label__type', text: f.kind === 'csv' ? 'string[]' : (f.kind || 'string') })
        ]), input,
        f.hint ? el('span', { class: 'f-label__hint', text: f.hint }) : null]);
      label.querySelector('input,select').dataset.key = f.key;
      /* Schema order is preserved: the first four fields pair up, the rest
         stack in a second row rather than being interleaved by name length. */
      (index < 4 ? rowA : rowB).appendChild(label);
    });

    host.appendChild(rowA);
    if (rowB.childNodes.length) host.appendChild(rowB);

    $('#schema-note').className = 'note note--tight' + (fromBackend ? '' : ' note--warn');
    clear($('#schema-note')).appendChild(document.createTextNode(
      fromBackend
        ? 'Fields, required flags and hints above came from GET /api/schema/event.'
        : 'GET /api/schema/event did not answer, so this form uses the console’s built-in field list. Field names are the contract; descriptions are this console’s guesses.'));
  }

  function loadSchema() {
    return request('/api/schema/event', { timeoutMs: 8000 }).then(function (res) {
      if (superseded(res)) return res;
      state.schema = (res.ok && res.json) ? res.json : null;
      state.schemaFromBackend = !!(res.ok && res.json);
      if (res.ok && res.json) clear($('#schema-raw')).appendChild(document.createTextNode(JSON.stringify(res.json, null, 2)));
      else clear($('#schema-raw')).appendChild(document.createTextNode(
        'not fetched — HTTP ' + (res.status || 0) + ' from GET /api/schema/event\n' + describeFailure(res, 'GET /api/schema/event')));
      var fields = null;
      if (state.schema) { try { fields = fieldsFromSchema(state.schema); } catch (e) { fields = null; } }
      buildForm(fields || fieldsFromSchema(FALLBACK_SCHEMA), !!(fields && state.schemaFromBackend));
      return res;
    }).catch(function (err) {
      buildForm(fieldsFromSchema(FALLBACK_SCHEMA), false);
      toast('schema load threw: ' + (err && err.message ? err.message : String(err)), 'bad');
    });
  }

  function formPayload() {
    var out = {};
    state.formFields.forEach(function (f) {
      var input = $('#event-form-host [data-key="' + cssEscape(f.key) + '"]');
      if (!input) return;
      var raw = input.value;
      if (raw === null || raw === undefined) return;
      var v = String(raw).trim();
      if (v === '') return;
      if (f.kind === 'csv') {
        out[f.key] = v.split(',').map(function (s) { return s.trim(); }).filter(Boolean);
      } else if (f.kind === 'number') {
        var n = Number(v);
        if (!isFinite(n)) { input.setAttribute('aria-invalid', 'true'); return; }
        input.removeAttribute('aria-invalid');
        out[f.key] = n;
      } else if (f.kind === 'enum') {
        out[f.key] = v;
      } else {
        out[f.key] = v;
      }
    });
    return out;
  }

  function prefillForm() {
    state.formFields.forEach(function (f) {
      var input = $('#event-form-host [data-key="' + cssEscape(f.key) + '"]');
      if (!input) return;
      var v = DEMO_EVENT[f.key];
      input.value = v === undefined || v === null ? '' : String(v);
    });
    setStatus('#event-status', 'demo values filled in — nothing has been sent to the backend yet', 'busy');
  }

  function setStatus(sel, msg, kind) {
    var n = $(sel);
    if (!n) return;
    n.className = 'form__status' + (kind ? ' form__status--' + kind : '');
    n.textContent = msg || '';
  }

  function createEvent(ev) {
    if (ev) ev.preventDefault();
    var payload = formPayload();
    var missing = state.formFields.filter(function (f) {
      return f.required && payload[f.key] === undefined;
    }).map(function (f) { return f.key; });
    if (missing.length) {
      setStatus('#event-status', 'missing required: ' + missing.join(', '), 'err');
      return Promise.resolve(null);
    }
    setStatus('#event-status', 'POST /api/events …', 'busy');
    return request('/api/events', { method: 'POST', body: payload, timeoutMs: 30000 }).then(function (res) {
      if (superseded(res)) return null;
      if (reportNetFailure(res, 'POST /api/events', 'create')) { setStatus('#event-status', 'backend unreachable', 'err'); return null; }
      if (!res.ok) {
        setStatus('#event-status', 'HTTP ' + res.status + ' — ' + (shortDetail(res.json) || 'no detail in body'), 'err');
        setIssue('create', 'info', 'Create event was refused (HTTP ' + res.status + ')',
          'POST /api/events answered with a real error, so the backend is reachable: ' + (shortDetail(res.json) || 'no detail body') + ' Fix the form and try again.', null);
        return null;
      }
      clearIssue('create');
      var body = unwrap(res.json);
      var evtNested = (body && body.event && typeof body.event === 'object') ? body.event
        : ((res.json && res.json.event && typeof res.json.event === 'object') ? res.json.event : null);
      var dataNested = (res.json && res.json.data && typeof res.json.data === 'object') ? res.json.data : null;
      var id = str(pick(body, ['event_id', 'id'])) ||
        (evtNested && str(pick(evtNested, ['event_id', 'id']))) ||
        (dataNested && str(pick(dataNested, ['event_id', 'id']))) ||
        (res.json && str(pick(res.json, ['event_id', 'id'])));
      if (!id) {
        setStatus('#event-status', 'created — but the response did not include an event_id, so stages cannot be called', 'err');
        setIssue('create', 'warn', 'Event created without an id in the response',
          'POST /api/events returned HTTP ' + res.status + ' but no event_id field. This console will not guess one; paste the id once the backend includes it.', null);
      } else {
        state.eventId = id;
        lsSet(LS.event, id);
        setStatus('#event-status', 'created ' + id + ' · stages unlocked', 'ok');
        learnRunId(body);
        learnRunId(res.json);
        if (dataNested) learnRunId(dataNested);
      }
      renderStages();
      return id;
    });
  }

  /* Some endpoints answer {result: ...} or {data: ...}. Look one level down.
   * FastAPI error envelopes answer {detail: ...}: a 403 gate refusal is
   * {"detail": {gated: true, gate: {...}}}, so a dict detail must be
   * unwrapped too (422 bodies use a detail *array*, which is left alone). */
  function unwrap(json) {
    if (json && typeof json === 'object') {
      if (json.detail && typeof json.detail === 'object' && !Array.isArray(json.detail)) return json.detail;
      if (json.result && typeof json.result === 'object') return json.result;
      if (json.data && typeof json.data === 'object' && !json.event_id) return json.data;
    }
    return (json && typeof json === 'object') ? json : {};
  }

  function learnRunId(obj) {
    if (!obj || typeof obj !== 'object') return;
    var rid = str(pick(obj, ['run_id', 'runId']));
    if (!rid) return;
    if (state.runId === rid) return;
    state.runId = rid;
    state.threadId = '';
    $('#run-id').value = rid;
    lsSet(LS.run, rid);
    /* A new run id means a new stream: clear the old run's feed, watermark
     * and panels first, or the two runs render interleaved under one id. */
    resetForRun(rid);
    connectTrace({ fresh: true });
    scheduleRefresh(true);
  }

  /* The resume endpoint parks on a LangGraph thread; stages echo it back as
   * thread_id. Remember the last one seen so a resume continues the parked
   * thread instead of addressing the run id and finding nothing parked. */
  function learnThreadId(obj) {
    if (!obj || typeof obj !== 'object') return;
    var tid = str(pick(obj, ['thread_id', 'threadId']));
    if (tid) state.threadId = tid;
  }

  /* ============================== pipeline ============================== */

  function renderStages() {
    var host = clear($('#stage-list'));
    STAGES.forEach(function (s, i) {
      var st = state.stageState[s.key] || (state.stageState[s.key] = { state: 'idle', result: '' });
      var approved = s.key === 'outreach' && st.state === 'gated' && state.approvals.send;
      var btnLabel = approved ? 'Send outreach (approved)'
        : (s.key === 'outreach' && st.state === 'gated' ? 'Send outreach'
        : (s.key === 'reply' ? 'Send reply' : s.label));
      var disabled = !state.eventId;
      var wrap = el('div', { class: 'stage', 'data-agent': s.agent, 'data-state': st.state, 'data-stage': s.key }, [
        el('span', { class: 'stage__badge' }, [agentBadge(s.agent)]),
        el('span', { class: 'stage__mid' }, [
          el('span', { class: 'stage__label', text: btnLabel + '  ' }),
          el('span', { class: 'stage__blurb', text: s.blurb })
        ]),
        el('span', { class: 'stage__state', text: st.state }),
        el('span', {}, [el('button', {
          type: 'button', class: 'btn btn--xs' + (s.key === 'outreach' && st.state === 'gated' ? ' btn--primary' : ''),
          disabled: disabled,
          title: disabled ? 'create an event first' : 'press ' + (i + 1) + ' to run ' + s.key,
          onclick: function () { runStage(s); }
        }, [st.state === 'running' ? el('span', { class: 'spinner' }) : null,
            el('span', { text: st.state === 'running' ? 'running' : 'run ' })])]),
        el('span', { class: 'stage__result', text: st.result })
      ]);
      host.appendChild(wrap);
      st.node = wrap;
    });
  }

  function stageDef(key) {
    for (var i = 0; i < STAGES.length; i++) if (STAGES[i].key === key) return STAGES[i];
    return { key: key, label: key, blurb: '' };
  }

  function setStage(key, st, result) {
    var s = state.stageState[key];
    if (!s) return;
    var def = stageDef(key);
    s.state = st;
    if (result !== undefined) s.result = result;
    var wrap = s.node;
    if (!wrap) { renderStages(); return; }
    wrap.setAttribute('data-state', st);
    var label = wrap.querySelector('.stage__state');
    var btn = wrap.querySelector('button');
    if (label) label.textContent = st;
    if (btn) {
      clear(btn);
      if (st === 'running') btn.appendChild(el('span', { class: 'spinner' }));
      btn.appendChild(el('span', { text: st === 'running' ? 'running' : 'run ' }));
      btn.className = 'btn btn--xs' + (key === 'outreach' && st === 'gated' ? ' btn--primary' : '');
    }
    var res = wrap.querySelector('.stage__result');
    if (res) res.textContent = s.result || '';
    if (key === 'outreach') {
      var mid = wrap.querySelector('.stage__label');
      if (mid) mid.textContent =
        (state.approvals.send && st === 'gated' ? 'Send outreach (approved)  '
          : (st === 'gated' ? 'Send outreach  ' : def.label + '  '));
    }
  }

  function summarizePayload(json) {
    if (json === null || json === undefined) return 'backend returned an empty body';
    if (typeof json === 'string') return json.slice(0, 160);
    if (typeof json !== 'object') return String(json);
    var body = unwrap(json);
    var bits = [];
    var seen = 0;
    for (var k in body) {
      if (!Object.prototype.hasOwnProperty.call(body, k)) continue;
      if (seen >= 7) { bits.push('…'); break; }
      var v = body[k];
      if (Array.isArray(v)) { bits.push(k + ': ' + v.length); seen++; }
      else if (v && typeof v === 'object') {
        var inner = Object.keys(v);
        bits.push(k + ': {' + inner.length + ' keys}');
        seen++;
      } else if (typeof v === 'boolean') { bits.push(k + '=' + v); seen++; }
      else if (typeof v === 'number') { bits.push(k + '=' + v); seen++; }
      else if (typeof v === 'string') { bits.push(k + '="' + v.slice(0, 40) + '"'); seen++; }
    }
    if (!bits.length) return 'response body had no reportable fields: ' + JSON.stringify(json).slice(0, 120);
    return bits.join(' · ');
  }

  function latestGateIdFor(kindWanted) {
    var want = (kindWanted || '').toLowerCase();
    var fallback = null;
    for (var i = state.gates.length - 1; i >= 0; i--) {
      var g = state.gates[i];
      if (!g || g.synthetic) continue;
      /* Graph-interrupt ids are not API-ledger ids: ?gate_id= must name a
       * ledger row, so parked-run rows are never eligible here. */
      if (g.origin === 'graph') continue;
      var gk = (g.kind || '').toLowerCase();
      var match = !want || gk === want || (want === 'send' && gk === '');
      if (!match) continue;
      fallback = fallback || g.gate_id;
      if (g.outcome && /approve/i.test(g.outcome)) return g.gate_id;
    }
    return fallback;
  }

  function runStage(s) {
    if (!state.eventId) {
      toast('create an event first — the stage endpoints are keyed by event_id', 'bad');
      return Promise.resolve(null);
    }
    var body;
    var path;
    if (s.key === 'reply') {
      var text = $('#reply-text').value.trim();
      if (!text) { toast('type or pick a sponsor reply first', 'bad'); return Promise.resolve(null); }
      body = { text: text };
      var brand = $('#reply-brand').value.trim();
      if (brand) body.brand = brand;
      /* No event_id in the body: StageRequest is extra=forbid, so it would
       * be a 422. The event travels in the URL path. */
      if (state.runId) body.thread_id = state.runId;
      path = s.path(state.eventId);
    } else {
      body = {};
      if (state.runId) body.thread_id = state.runId;
      if (s.key === 'contract') {
        /* The contract stage has no brand input of its own: reuse the reply
         * brand when the user typed one, otherwise omit it and let the
         * backend fall back to the last brand. Never send an empty brand. */
        var cbrandEl = $('#reply-brand');
        var cbrand = cbrandEl ? cbrandEl.value.trim() : '';
        if (cbrand) body.brand = cbrand;
      }
      path = s.path(state.eventId);
      if (s.key === 'outreach') {
        /* Retry must resume the same thread with the same gate: the
         * event-scoped form reads gate_id from the query string (a body
         * gate_id would be a 422 on StageRequest), so it travels in the URL
         * while thread_id travels in the body. */
        var gid = latestGateIdFor('send');
        if (gid) path += (path.indexOf('?') >= 0 ? '&' : '?') + 'gate_id=' + enc(gid);
      }
    }

    /* 300 s, not 120 s: a full stage run was measured at 133 s wall clock,
     * so the old budget reported live backends as unreachable. A genuine
     * timeout still lands in the timeout branch below (banner + stage note),
     * never as silence. */
    var t0 = Date.now();
    setStage(s.key, 'running', '…');
    return request(path, { method: 'POST', body: body, timeoutMs: 300000 }).then(function (res) {
      if (superseded(res)) return null;
      var took = ' · took ' + (fmtMs(Date.now() - t0) || ((Date.now() - t0) + ' ms'));
      if (reportNetFailure(res, 'POST ' + path, 'stage:' + s.key)) {
        setStage(s.key, 'error', 'backend unreachable — ' + (res.error || '') + ' (nothing ran)' + took);
        return null;
      }
      learnRunId(res.json || {});
      var bodyObj = unwrap(res.json);
      learnRunId(bodyObj);
      learnThreadId(res.json || {});
      learnThreadId(bodyObj);

      if (!res.ok) {
        /* FastAPI wraps the refusal as {"detail": {gated:true, gate:{...}}},
         * so the flag lives one level down. unwrap() already descends, but
         * the envelope is checked explicitly too — a missed gate here means
         * no gate card ever appears. */
        var detailObj = (res.json && res.json.detail && typeof res.json.detail === 'object' && !Array.isArray(res.json.detail))
          ? res.json.detail : {};
        var gated = truthy(pick(res.json, ['gated']))
          || (res.status === 403 && truthy(pick(bodyObj, ['gated'])))
          || truthy(pick(detailObj, ['gated']));
        if (gated) {
          setStage(s.key, 'gated', 'refused HTTP ' + res.status + ' {gated:true} — a human gate is holding this side effect' + took);
          addGate(pick(res.json, ['gate']) || pick(bodyObj, ['gate']) || pick(detailObj, ['gate']) || null, res.json, path);
          clearIssue('stage:' + s.key);
          refreshGates();
          return res;
        }
        setStage(s.key, 'error', 'HTTP ' + res.status + ' — ' + (shortDetail(res.json) || 'no detail in body') + took);
        setIssue('stage:' + s.key, 'info', s.label + ' refused (HTTP ' + res.status + ')',
          'POST ' + path + ' answered with a real error: ' + (shortDetail(res.json) || 'no detail body'), null);
        return res;
      }

      clearIssue('stage:' + s.key);
      var note = summarizePayload(res.json);
      /* Never claim sent while a gate is pending: a parked run answers 200
       * with {parked:true, pending_gates:[...]}, which reads as gated. */
      var parked = bodyObj && (bodyObj.parked === true || (bodyObj.pending_gates && bodyObj.pending_gates.length));
      if (parked) {
        var pend = bodyObj.pending_gates || [];
        state.parked = { at: Date.now(), pending: pend, stage: s.key };
        setStage(s.key, 'gated', 'parked on a human gate — nothing was sent' +
          (pend.length && pend[0].gate_id ? ' (gate ' + pend[0].gate_id + ')' : '') + took);
        /* pending_gates are graph interrupts (a different id space from the
         * API ledger): each renders with approve/reject/revise wired to
         * POST /api/runs/{run_id}/resume. */
        if (pend.length) addGraphInterrupts(pend, res.json, path);
        refreshGates();
        refreshSummary();
        scheduleRefresh(true);
        return res;
      }
      state.parked = null;
      setStage(s.key, 'done', note + took);
      if (s.key === 'reply') renderReplyResult(res.json);
      refreshGates();
      scheduleRefresh(true);
      return res;
    }).catch(function (err) {
      setStage(s.key, 'error', 'request threw: ' + (err && err.message ? err.message : String(err)));
      return null;
    });
  }

  /* ================================ gates ================================ */

  /* One gate is one card. A gate reaches this console twice — in the 403 body and
     * again from GET /api/gates/{run_id} — so every path funnels through here
     * and merges on gate_id rather than appending. */
  function upsertGate(raw, path, synthetic, origin) {
    if (!raw || typeof raw !== 'object') return null;
    var id = str(pick(raw, ['gate_id', 'id']));
    var existing = id ? findGate(id) : null;
    if (existing) {
      if (raw && Object.keys(raw).length) existing.raw = raw;
      existing.kind = str(pick(raw, ['kind'])) || existing.kind;
      existing.question = str(pick(raw, ['question'])) || existing.question;
      existing.payload_preview = str(pick(raw, ['payload_preview'])) || existing.payload_preview;
      existing.raised_at = str(pick(raw, ['raised_at', 'at'])) || existing.raised_at;
      if (path) existing.path = path;
      if (!existing.synthetic && synthetic) existing.synthetic = false;
      return existing;
    }
    var nid = id || ('unidentified-' + now() + '-' + Math.floor(Math.random() * 1e4));
    var rec = {
      gate_id: nid, synthetic: synthetic || !id, raw: raw, path: path || null,
      origin: origin || 'api',
      kind: str(pick(raw, ['kind'])),
      question: str(pick(raw, ['question'])),
      payload_preview: str(pick(raw, ['payload_preview'])),
      raised_at: str(pick(raw, ['raised_at', 'at'])),
      outcome: null
    };
    state.gates.push(rec);
    return rec;
  }

  function addGate(gate, envelope, path) {
    var raw = (gate && typeof gate === 'object') ? gate : (envelope && typeof envelope === 'object' ? envelope : {});
    var rec = upsertGate(raw, path, !(gate && str(pick(gate, ['gate_id', 'id']))), 'api');
    if (rec && rec.synthetic && !rec.path) {
      setIssue('gate-noid', 'warn', 'Gate raised but no gate_id came back',
        'POST ' + path + ' answered ' + JSON.stringify(envelope).slice(0, 220) +
        '. Without a gate_id this console cannot submit an outcome to POST /api/gates/{gate_id}; press Refresh to see whether the backend lists it under /api/gates/' + (state.runId || '{run_id}') + '.', null);
    }
    renderGates();
  }

  /* Graph-interrupt rows from a 200 {parked:true} stage response (or a resume
   * response) live in the graph's id space, not the API GateLedger's:
   * POST /api/gates/{graph gate_id} would 404, and GET /api/gates/{run_id}
   * never lists them. They are stored with origin 'graph' and answered via
   * POST /api/runs/{run_id}/resume with NO gate_id (the resume bridge lookup
   * would 404 a graph-local id). Rows without a gate_id ({interrupt: "..."})
   * become synthetic cards that resume the run the same way. */
  function addGraphInterrupts(pend, envelope, path) {
    void envelope;
    (pend || []).forEach(function (row) {
      if (row && typeof row === 'object' && str(pick(row, ['gate_id', 'id']))) {
        upsertGate(row, path, false, 'graph');
      } else if (row && typeof row === 'object') {
        var q = str(pick(row, ['interrupt', 'question', 'message']))
          || JSON.stringify(row).slice(0, 300);
        /* Synthetic ids are random, so de-duplicate on the question text:
         * every parked response would otherwise append another card. */
        var dup = null;
        for (var i = 0; i < state.gates.length; i++) {
          var g0 = state.gates[i];
          if (!g0.outcome && g0.synthetic && g0.origin === 'graph' && g0.question === q) { dup = g0; break; }
        }
        if (dup) { if (path) dup.path = path; }
        else upsertGate({ question: q, kind: str(pick(row, ['kind'])) || 'interrupt' }, path, true, 'graph');
      }
    });
    renderGates();
  }

  function gateList(json) {
    if (Array.isArray(json)) return json;
    if (json && typeof json === 'object') {
      if (Array.isArray(json.gates)) return json.gates;
      if (Array.isArray(json.items)) return json.items;
      /* GET /api/gates/{run_id} answers {outstanding:[{gate, decision, ...}],
       * answered:[...]} — flatten each row to its gate, carrying the recorded
       * outcome (if any) so mergeGates can mark it decided. */
      if (Array.isArray(json.outstanding) || Array.isArray(json.answered)) {
        var rows = (json.outstanding || []).concat(json.answered || []);
        return rows.map(function (row) {
          if (row && typeof row === 'object' && row.gate && typeof row.gate === 'object') {
            var flat = {};
            for (var k in row.gate) {
              if (Object.prototype.hasOwnProperty.call(row.gate, k)) flat[k] = row.gate[k];
            }
            var dec = row.decision && typeof row.decision === 'object' ? row.decision : null;
            var outc = dec && str(pick(dec, ['outcome']));
            if (outc) flat.outcome = outc;
            return flat;
          }
          return row;
        });
      }
      if (json.gate) return [json.gate];
    }
    return null;
  }

  function mergeGates(list, origin) {
    list.forEach(function (g) {
      var rec = upsertGate(g, null, !str(pick(g, ['gate_id', 'id'])), origin || 'api');
      if (rec) {
        var outc = pick(g, ['outcome', 'decision_outcome']);
        if (outc && !rec.outcome) rec.outcome = String(outc);
      }
    });
    renderGates();
  }

  function findGate(id) {
    for (var i = 0; i < state.gates.length; i++) if (state.gates[i].gate_id === id) return state.gates[i];
    return null;
  }

  function refreshGates() {
    if (!state.runId) {
      clear($('#gates-host')).appendChild(el('p', { class: 'empty',
        text: 'No run_id yet, so GET /api/gates/{run_id} cannot be called. Gates appear here the moment a stage returns 403 {gated:true}.' }));
      return Promise.resolve(null);
    }
    var path = '/api/gates/' + enc(state.runId);
    return request(path, { timeoutMs: 10000 }).then(function (res) {
      if (superseded(res)) return res;
      if (reportNetFailure(res, 'GET ' + path, 'gates')) {
        clear($('#gates-host')).appendChild(el('p', { class: 'note note--bad',
          text: describeFailure(res, 'GET ' + path) }));
        return res;
      }
      if (!res.ok) {
        clear($('#gates-host')).appendChild(el('p', { class: 'note note--bad',
          text: 'GET ' + path + ' → HTTP ' + res.status + ' — ' + (shortDetail(res.json) || 'no detail in body') }));
        return res;
      }
      var list = gateList(res.json);
      if (list === null) {
        clear($('#gates-host')).appendChild(el('p', { class: 'empty',
          text: 'GET ' + path + ' answered with a shape this console does not recognise: ' + JSON.stringify(res.json).slice(0, 160) }));
        return res;
      }
      mergeGates(list);
      return res;
    });
  }

  function renderGates() {
    var host = clear($('#gates-host'));
    var open = state.gates.filter(function (g) { return !g.outcome; });
    var done = state.gates.filter(function (g) { return g.outcome; });

    if (!state.gates.length) {
      host.appendChild(el('p', { class: 'empty',
        text: 'No gate requested yet. Press Outreach — it is expected to be refused with 403 {gated:true} until a human approves the send.' }));
      return;
    }
    host.appendChild(el('p', { class: 'note note--tight',
      text: open.length + ' outstanding · ' + done.length + ' decided · api gates from POST responses + GET /api/gates/' + (state.runId || '{run_id}') + ' · graph interrupts from parked/resumed stage responses' }));

    open.concat(done).forEach(function (g) { host.appendChild(gateCard(g)); });
  }

  function gateCard(g) {
    var decided = !!g.outcome;
    var isGraph = g.origin === 'graph';
    var card = el('div', { class: 'gate' + (decided ? ' gate--resolved' : '') }, [
      el('div', { class: 'gate__head' }, [
        g.kind ? el('span', { class: 'chip chip--warn', text: g.kind }) : null,
        isGraph
          ? el('span', { class: 'chip chip--accent', title: 'this gate comes from a parked graph run (pending_gates), not the API gate ledger — it is answered via POST /api/runs/{run_id}/resume', text: 'graph interrupt' })
          : el('span', { class: 'chip chip--muted', title: 'this gate comes from the API gate ledger — answered via POST /api/gates/{gate_id}', text: 'api gate' }),
        el('span', { class: 'gate__meta', text: 'gate_id ' + g.gate_id }),
        g.synthetic ? el('span', { class: 'chip chip--bad', text: 'id not sent' }) : null,
        g.raised_at ? el('span', { class: 'gate__meta', title: g.raised_at, text: ago(parseTs(g.raised_at)) || g.raised_at }) : null
      ]),
      el('div', { class: 'gate__q', text: g.question || '(the backend sent no question text)' })
    ]);

    if (g.payload_preview) {
      card.appendChild(el('div', { class: 'gate__preview', text: g.payload_preview }));
    } else if (g.raw && Object.keys(g.raw).length) {
      card.appendChild(el('div', { class: 'gate__preview', text: JSON.stringify(g.raw, null, 2).slice(0, 900) }));
    }

    if (decided) {
      card.appendChild(el('div', { class: 'gate__resolved', text: 'decided: ' + g.outcome }));
      return card;
    }

    var ta = el('textarea', { rows: 2, placeholder: 'instruction for the agents (sent with the outcome)', spellcheck: 'false' });
    var status = el('span', { class: 'form__status' });
    var mk = function (outcome, cls) {
      return el('button', {
        type: 'button', class: 'btn btn--xs ' + cls,
        onclick: function () { submitGate(g, outcome, ta.value, status); }
      }, [el('span', { text: outcome })]);
    };
    card.appendChild(el('div', { class: 'gate__row' }, [
      mk('approve', 'btn--approve'), mk('reject', 'btn--danger'), mk('revise', '')
    ]));
    card.appendChild(el('div', { class: 'gate__instruction' }, [ta]));
    card.appendChild(el('div', { class: 'gate__row' }, [status,
      el('span', { class: 'gate__missing', text: isGraph
        ? (g.synthetic
          ? 'graph interrupt with no gate_id — answer it below (POST /api/runs/' + (state.runId || '{run_id}') + '/resume, no gate_id)'
          : 'graph interrupt — approve/reject/revise below resumes the parked run (POST /api/runs/' + (state.runId || '{run_id}') + '/resume, no gate_id)')
        : (g.synthetic
          ? 'the backend did not send a gate_id, so POST /api/gates/{gate_id} cannot be formed — refresh gates'
          : ('POST /api/gates/' + g.gate_id +
            (state.parked && state.parked.pending && state.parked.pending.length ? ' · then auto-resumes the parked run' : ''))) })]));
    /* Synthetic graph cards keep their buttons: resuming needs no gate_id. */
    if (g.synthetic && !isGraph) {
      var btn = card.querySelectorAll('.gate__row')[0].children[0];
      btn.disabled = true;
    }
    return card;
  }

  function submitGate(g, outcome, instruction, statusEl) {
    var instructionText = (instruction && instruction.trim()) ? instruction.trim() : '';
    /* Graph interrupts are answered by resuming the parked run, not by the
     * API ledger: their ids live in the graph's id space and would 404 both
     * POST /api/gates/{gate_id} and the resume bridge lookup. */
    if (g.origin === 'graph') {
      resumeRun({ outcome: outcome, instruction: instructionText, gate_id: null, gate: g, statusEl: statusEl });
      return;
    }
    if (g.synthetic) { statusEl.className = 'form__status form__status--err'; statusEl.textContent = 'no gate_id to submit to'; return; }
    statusEl.className = 'form__status form__status--busy';
    statusEl.textContent = 'POST /api/gates/' + g.gate_id + ' …';
    var body = { outcome: outcome, decided_by: 'reviewer' };
    if (instructionText) body.instruction = instructionText;
    request('/api/gates/' + enc(g.gate_id), { method: 'POST', body: body, timeoutMs: 20000 }).then(function (res) {
      if (superseded(res)) return;
      if (reportNetFailure(res, 'POST /api/gates/' + g.gate_id, 'gate')) {
        statusEl.className = 'form__status form__status--err';
        statusEl.textContent = 'backend unreachable — outcome not submitted';
        return;
      }
      if (!res.ok) {
        statusEl.className = 'form__status form__status--err';
        statusEl.textContent = 'HTTP ' + res.status + ' — ' + (shortDetail(res.json) || 'no detail');
        return;
      }
      clearIssue('gate');
      g.outcome = outcome + (instructionText ? ' · "' + instructionText.slice(0, 60) + '"' : '');
      if (outcome === 'approve') state.approvals[g.kind || 'send'] = true;
      renderGates();
      if (state.stageState.outreach && state.stageState.outreach.state === 'gated') {
        setStage('outreach', 'gated', 'gate ' + g.gate_id + ' approved — resuming the parked run; if it was never parked, press Send outreach to re-run it with the approval on file');
      }
      /* An approved API gate unblocks the side effect, but a parked graph
       * run stays parked until it is resumed: chain the resume on the same
       * thread so one click finishes both ledgers. A 409 from the resume
       * just means the run was never parked — the approval still stands. */
      if (shouldChainResume(g)) {
        resumeRun({ outcome: outcome, instruction: instructionText, gate_id: g.gate_id, gate: g, statusEl: statusEl, chained: true });
      } else {
        scheduleRefresh(true);
      }
    });
  }

  /* True when answering this API gate should be followed by a resume: the
   * last stage response parked the run, or an outreach/contract stage is
   * sitting gated (those are the stages whose re-runs park). */
  function shouldChainResume(g) {
    if (!state.runId) return false;
    if (state.parked && state.parked.pending && state.parked.pending.length) return true;
    var k = (g.kind || '').toLowerCase();
    if (k === 'send' || k === 'mou' || k === 'counter') return true;
    var o = state.stageState.outreach, c = state.stageState.contract;
    if ((o && o.state === 'gated') || (c && c.state === 'gated')) return true;
    return false;
  }

  /* Resume a parked run on its thread. gate_id is sent ONLY for API-ledger
   * gates (so the resume bridges that ledger row too); graph-local ids are
   * passed as null because the bridge lookup would 404 them. */
  function resumeRun(opts) {
    var outcome = opts.outcome;
    var instructionText = opts.instruction || '';
    var gateId = opts.gate_id || null;
    var gate = opts.gate || null;
    var statusEl = opts.statusEl || null;
    if (!state.runId) {
      var msg = 'no run_id — resume needs POST /api/runs/{run_id}/resume';
      if (statusEl) { statusEl.className = 'form__status form__status--err'; statusEl.textContent = msg; }
      else toast(msg, 'bad');
      return Promise.resolve(null);
    }
    var body = { outcome: outcome, decided_by: 'reviewer' };
    if (instructionText) body.instruction = instructionText;
    if (gateId) body.gate_id = gateId;
    body.thread_id = state.threadId || state.runId;
    var path = '/api/runs/' + enc(state.runId) + '/resume';
    if (statusEl) { statusEl.className = 'form__status form__status--busy'; statusEl.textContent = 'POST ' + path + ' …'; }
    /* Stage-like budget: a resume continues the parked run, which can run
     * agents before answering. This is a new call, not a changed timeout. */
    return request(path, { method: 'POST', body: body, timeoutMs: 300000 }).then(function (res) {
      if (superseded(res)) return res;
      if (reportNetFailure(res, 'POST ' + path, 'resume')) {
        if (statusEl) { statusEl.className = 'form__status form__status--err'; statusEl.textContent = 'backend unreachable — resume not sent'; }
        return res;
      }
      if (res.status === 409) {
        /* Not parked: the ledger approval (if any) still stands; the stage
         * must simply be re-run. A real answer, not a failure. */
        state.parked = null;
        if (statusEl) { statusEl.className = 'form__status form__status--ok'; statusEl.textContent = 'run is not parked (' + (shortDetail(res.json) || 'no pending gate') + ') — approval stands; re-run the stage'; }
        else toast('run is not parked — approval stands; re-run the stage', 'good');
        refreshGates();
        scheduleRefresh(true);
        return res;
      }
      if (!res.ok) {
        if (statusEl) { statusEl.className = 'form__status form__status--err'; statusEl.textContent = 'HTTP ' + res.status + ' — ' + (shortDetail(res.json) || 'no detail'); }
        return res;
      }
      clearIssue('resume');
      var rbody = unwrap(res.json);
      learnRunId(res.json || {});
      learnRunId(rbody);
      learnThreadId(res.json || {});
      learnThreadId(rbody);
      var pend = (rbody && rbody.pending_gates) || [];
      var stillParked = !!(rbody && (rbody.parked === true || (pend && pend.length)));
      var suffix = instructionText ? ' · "' + instructionText.slice(0, 60) + '"' : '';
      if (stillParked) {
        state.parked = { at: Date.now(), pending: pend, stage: (state.parked && state.parked.stage) || '' };
        /* The answered interrupt is consumed; whatever is pending now is a
         * different gate, so close the answered card and list the rest. */
        if (gate) gate.outcome = outcome + suffix + ' · resumed, run still parked';
        if (pend.length) addGraphInterrupts(pend, res.json, path);
        renderGates();
        if (statusEl) { statusEl.className = 'form__status form__status--ok'; statusEl.textContent = 'resumed — run is parked on ' + pend.length + ' more gate(s)'; }
        var pst = state.parked.stage && state.stageState[state.parked.stage];
        if (pst) setStage(state.parked.stage, 'gated', 'resumed — still parked on ' + pend.length + ' gate(s), nothing was sent');
      } else {
        state.parked = null;
        markGraphGatesDecided(outcome, suffix, gate);
        if (gate) gate.outcome = outcome + suffix + ' · resumed';
        renderGates();
        if (state.stageState.outreach && state.stageState.outreach.state === 'gated' && outcome === 'approve') {
          setStage('outreach', 'done', 'resumed — parked:false, run continued');
        }
        if (statusEl) { statusEl.className = 'form__status form__status--ok'; statusEl.textContent = 'resumed — parked:false, run continued'; }
        toast('run resumed — parked:false, run continued', 'good');
      }
      refreshGates();
      refreshSummary();
      scheduleRefresh(true);
      return res;
    });
  }

  /* A resume that clears the park consumes every pending interrupt, so any
   * other open graph card from the same park is answered too. API-ledger
   * cards are left alone: GET /api/gates/{run_id} is the truth for those. */
  function markGraphGatesDecided(outcome, suffix, except) {
    state.gates.forEach(function (r) {
      if (r === except || r.outcome || r.origin !== 'graph') return;
      r.outcome = outcome + suffix + ' · cleared by resume';
    });
  }

  /* ============================= reply presets ============================ */

  function buildPresets() {
    var row = clear($('#preset-row'));
    var note = clear($('#preset-note'));
    note.className = 'note note--tight';
    PRESETS.forEach(function (p) {
      var b = el('button', {
        type: 'button', class: 'preset' + (p.trap ? ' preset--trap' : ''),
        'aria-pressed': 'false', title: p.note || 'send this exact text as the simulated sponsor',
        onclick: function () { selectPreset(p.key); }
      }, [el('span', { text: p.label })]);
      p.node = b;
      row.appendChild(b);
    });
    selectPreset('pushback');
  }

  function selectPreset(key) {
    var p = null;
    for (var i = 0; i < PRESETS.length; i++) if (PRESETS[i].key === key) p = PRESETS[i];
    if (!p) return;
    PRESETS.forEach(function (q) { if (q.node) q.node.setAttribute('aria-pressed', q === p ? 'true' : 'false'); });
    $('#reply-text').value = p.text;
    var note = clear($('#preset-note'));
    note.className = 'note note--tight' + (p.trap ? ' note--warn' : '');
    note.appendChild(el('span', { text: p.note ? p.note : 'expected intent: ' + p.key + '. The backend classifies it; this console only shows what came back.' }));
  }

  function renderReplyResult(json) {
    var host = clear($('#reply-result'));
    var body = unwrap(json);
    var intent = str(pick(body, ['intent', 'classified_intent', 'reply_intent']));
    var route = pick(body, ['route', 'routed_to', 'next_agent', 'next', 'next_stage']);
    var reason = str(pick(body, ['reason', 'rationale', 'explanation', 'why']));
    var thread = str(pick(body, ['thread_id', 'brand']));
    var decision = findIntentDecision();

    host.appendChild(el('div', { class: 'intent-card', 'data-agent': 'A3' }, [
      el('div', { class: 'intent-card__top' }, [
        agentBadge('ENV'),
        el('span', { class: 'chip chip--muted', text: 'counterparty replied' }),
        el('span', { class: 'chip chip--accent', text: 'A3 classified' }),
        intent ? el('span', { class: 'intent-card__intent', text: intent }) : absent('no intent in response'),
        thread ? el('span', { class: 'chip', text: thread }) : null
      ]),
      el('div', { class: 'intent-card__route' },
        ['route: ', route ? el('span', { class: 'mono', text: String(route) }) : absent('no route in response')]),
      reason ? el('div', { class: 'intent-card__why', text: 'backend says: ' + reason }) : null,
      decision ? el('div', { class: 'intent-card__why' }, [
        'classified by the decision at seq ', el('span', { class: 'mono', text: String(decision.seq) }),
        decision.request_id ? el('span', {}, [' (', el('span', { class: 'mono', text: decision.request_id }), ')']) : null,
        ' from ', el('span', { class: 'mono', text: decision.source || 'source not sent' }),
        decision.degraded ? el('span', { class: 'chip chip--warn', text: 'degraded fallback' }) : null,
        el('button', { type: 'button', class: 'dec__jump', text: 'show in inspector', onclick: function () { scrollToEvent(decision.seq); } })
      ]) : el('div', { class: 'intent-card__why' },
        'no decision event in the trace mentions this reply yet — the classifier may not have run, or its event has not streamed in'),
      el('details', { class: 'disclosure' }, [
        el('summary', { text: 'raw response body' }),
        el('pre', { class: 'pre pre--scroll', text: JSON.stringify(json, null, 2).slice(0, 4000) })
      ])
    ]));
  }

  function findIntentDecision() {
    for (var i = state.decisions.length - 1; i >= 0; i--) {
      var d = state.decisions[i];
      var hay = (d.question + ' ' + d.name).toLowerCase();
      if (/intent|reply|classif|negotiat/.test(hay)) return d;
    }
    return null;
  }

  /* ============================ SSE + trace feed ========================== */

  function agentBadge(agent) {
    var id = agent === null || agent === undefined ? '' : String(agent).trim().toUpperCase();
    if (id === 'ENVIRONMENT') id = 'ENV';
    if (!id) {
      return el('span', { class: 'chip chip--muted', title: 'this event names no agent — the backend did not send one', text: '—' });
    }
    var meta = AGENTS[id];
    var title = meta ? meta.label : 'agent id "' + id + '" is not one of A1–A7 or ENV';
    return el('span', {
      class: 'badge' + (id === 'ENV' ? ' badge--env' : ''), 'data-agent': id, title: title
    }, [el('span', { text: id }), el('span', { class: 'badge__name', text: meta ? meta.short : '?' })]);
  }

  function kindChip(kind) {
    return el('span', { class: 'chip chip--kind', style: '--kc:' + (KIND_COLOR[kind] || KIND_COLOR.unknown), text: kind });
  }

  function normKind(k) {
    var s = k === null || k === undefined ? '' : String(k).toLowerCase();
    return KINDS.indexOf(s) >= 0 ? s : 'unknown';
  }

  function eventKey(ev) {
    if (ev.seq !== undefined && ev.seq !== null && ev.run_id) return 'k:' + ev.run_id + ':' + ev.seq;
    if (ev.seq !== undefined && ev.seq !== null && ev.trace_id) return 'k:' + ev.trace_id + ':' + ev.seq;
    if (ev.seq !== undefined && ev.seq !== null) return 's:' + ev.seq + ':' + String(ev.kind) + ':' + String(ev.name);
    try { return 'j:' + JSON.stringify(ev).slice(0, 4000); } catch (e) { return 'r:' + Math.random(); }
  }

  function connectTrace(opts) {
    if (!state.runId) {
      setIssue('nrun', 'info', 'No run_id yet',
        'The stream endpoint is keyed by run_id. Create an event and run a stage, or paste a run_id above and press Connect.', null);
      return;
    }
    clearIssue('nrun');
    disconnectTrace();
    var url = apiUrl('/api/runs/' + enc(state.runId) + '/stream');
    /* A fresh EventSource has no memory of the previous one, so the browser's
     * automatic Last-Event-ID only covers retries of the same object. On a
     * manual (re)connect to the SAME run, pass the resume position explicitly
     * so the server skips what is already rendered. A run switch always starts
     * from the beginning: resetForRun cleared lastSeq before calling here. */
    if (state.lastSeq !== null && !(opts && opts.fresh)) {
      url += (url.indexOf('?') >= 0 ? '&' : '?') + 'last_event_id=' + enc(state.lastSeq);
    }
    setConn('connecting');
    try {
      es = new EventSource(url);
    } catch (e) {
      setIssue('sse', 'bad', 'Could not open the event stream',
        'EventSource(' + url + ') threw: ' + (e && e.message ? e.message : String(e)), START_CMD);
      setConn('down');
      return;
    }
    es.onopen = function () {
      clearIssue('sse');
      setConn('live');
    };
    es.onmessage = function (e) { handleMessage(e.data); };
    /* Named `trace` frames carry live events; onmessage alone never sees them,
     * so listen explicitly while keeping onmessage as the fallback for any
     * unnamed (default-event) frames the backend may emit. */
    es.addEventListener('trace', function (e) { handleMessage(e.data); });
    es.addEventListener('error', function (e) { if (e && e.data) handleStreamError(safeParse(e.data)); });
    /* The contract promises a final `summary` event; named events do not reach
     * onmessage, so listen for it explicitly (plus common spellings). `done`
     * is terminal, not data: it closes the stream and must never reach the
     * feed or the summary panel (see handleDone). */
    ['summary', 'complete', 'done', 'end'].forEach(function (name) {
      es.addEventListener(name, function (e) { handleNamed(name, e.data); });
    });
    es.onerror = function () {
      var rs = es ? es.readyState : -1;
      var closed = !es || rs === 2;
      if (closed) {
        setConn('down');
        setIssue('sse', 'bad', 'Event stream closed',
          'The SSE connection to ' + hostOf(url) + ' closed (EventSource readyState ' + rs + ' = CLOSED). ' +
          'Usual causes, in order: the backend is not running; the run_id is unknown, so the endpoint answered 404 and there is nothing to stream; ' +
          'or, when this page was opened from file://, the page origin was rejected by the backend CORS policy (the backend must allow the page origin — the default wildcard origin allows file:// pages with no credentials). ' +
          'Events already rendered are kept; a reconnect resumes from the last received event id, so the feed will not double up.', START_CMD);
      } else {
        setConn('reconnecting');
        setIssue('sse', 'warn', 'Event stream reconnecting',
          'Lost the SSE connection to ' + hostOf(url) + '. EventSource is retrying on its own with Last-Event-ID, so the server resumes after the last rendered event; anything re-sent is de-duplicated by seq so the feed will not double up.', null);
      }
    };
  }

  function disconnectTrace() {
    if (es) { try { es.close(); } catch (e) { /* ignore */ } es = null; }
    if (state.conn !== 'idle') setConn('idle');
  }

  function handleNamed(name, data) {
    if (name === 'summary') {
      var json = safeParse(data);
      if (json && typeof json === 'object') applySummary(json);
      return;
    }
    /* The terminal frame is bookkeeping, not data. It carries run_id,
     * terminated_by and event_count but no summary content: routing it into
     * handleMessage would either clobber the summary panel (it has
     * event_count, which applySummary accepts) or paint a bogus error row in
     * the feed. Close the stream instead — the server sent everything — and
     * say so where the trace status lives. */
    if (name === 'done' || name === 'complete' || name === 'end') {
      handleDone(safeParse(data));
      return;
    }
    if (name === 'error') {
      handleStreamError(safeParse(data));
      return;
    }
    handleMessage(data);
  }

  function handleDone(json) {
    var body = (json && typeof json === 'object') ? json : {};
    var rid = str(pick(body, ['run_id'])) || state.runId;
    var how = str(pick(body, ['terminated_by'])) || 'done';
    var count = numOrNull(pick(body, ['event_count']));
    /* The server closes the connection after `done`; without an explicit
     * close the browser would reconnect on its own and replay the run. */
    disconnectTrace();
    setStatus('#trace-status-note',
      'stream finished — run ' + (rid || '(unknown)') + ' · ' + how +
      (count !== null ? ' · ' + count + ' event(s)' : '') +
      '. The summary panel holds the final numbers.',
      'ok');
  }

  /* A named `error` frame from the server (no tracer, startup timeout,
   * deadline): a stated reason, rendered where the trace status lives — never
   * as a feed row, which would read as an agent failure. */
  function handleStreamError(json) {
    var body = (json && typeof json === 'object') ? json : null;
    var msg = body ? (str(pick(body, ['reason'])) || str(pick(body, ['error'])) || JSON.stringify(body).slice(0, 220)) : 'empty error frame';
    var rid = body ? (str(pick(body, ['run_id'])) || '') : '';
    setStatus('#trace-status-note', 'stream error' + (rid ? ' on run ' + rid : '') + ' — ' + msg, 'err');
    if (body && body.unavailable) {
      setIssue('sse-error', 'warn', 'The stream reported an unavailable subsystem',
        msg + ' Nothing was fabricated in its place; press load /trace or re-run the stage once the subsystem is back.', null);
    }
  }

  function safeParse(data) {
    if (data === null || data === undefined) return null;
    if (typeof data === 'object') return data;
    try { return JSON.parse(data); } catch (e) { return null; }
  }

  function handleMessage(data) {
    var json = safeParse(data);
    if (json === null) {
      state.malformed++;
      pushSyntheticRow('sse', null, 'unparseable SSE message', String(data).slice(0, 300));
      renderConnStats();
      return;
    }
    if (Array.isArray(json)) { json.forEach(handleMessage); return; }
    if (typeof json !== 'object') {
      pushSyntheticRow('sse', null, 'unexpected SSE payload type: ' + typeof json, String(json).slice(0, 200));
      return;
    }
    /* tolerate envelope shapes */
    var inner = json.event || json.payload || json.data || null;
    if (inner && typeof inner === 'object' && !Array.isArray(inner) &&
        (inner.kind !== undefined || inner.name !== undefined) && json.kind === undefined) {
      return handleMessage(inner);
    }
    if (json.kind !== undefined || (json.seq !== undefined && json.name !== undefined)) return ingestEvent(json);
    if (json.summary && typeof json.summary === 'object') return applySummary(json.summary);
    /* A terminal `done` frame seen without its event name (an unnamed-frame
     * fallback, a proxy that strips event names): it is bookkeeping, not a
     * summary and not a feed row. handleDone owns these; here they are noise. */
    if (json.terminated_by !== undefined && json.kind === undefined && json.seq === undefined && !json.summary) return;
    if (json.distinct_gap_values !== undefined || json.event_count !== undefined || json.decision_counts !== undefined) {
      return applySummary(json);
    }
    if (json.gate_id !== undefined || json.gates !== undefined) { mergeGates(gateList(json) || []); return; }
    pushSyntheticRow('sse', null, 'SSE message in an unrecognised shape', JSON.stringify(json).slice(0, 300));
  }

  function pushSyntheticRow(agent, kind, name, detail) {
    var ev = { seq: null, kind: kind || 'error', name: name, agent: agent || null, ts: new Date().toISOString(),
      duration_ms: null, status: 'error', attributes: { error: detail, origin: 'frontend' } };
    state.events.push({ ev: ev, node: null, synthetic: true });
    appendRow(ev, true);
    bumpCounters();
    return ev;
  }

  function ingestEvent(ev) {
    var key = eventKey(ev);
    if (state.seen[key]) {
      state.dupSuppressed++;
      renderConnStats();
      return null;
    }
    state.seen[key] = true;
    /* Advance the resume watermark: a manual reconnect sends this back as
     * ?last_event_id= so the server skips everything already rendered. */
    var seq = numOrNull(ev.seq);
    if (seq !== null && (state.lastSeq === null || seq > state.lastSeq)) state.lastSeq = seq;
    var rec = { ev: ev, node: null, kind: normKind(ev.kind), at: parseTs(ev.ts) || new Date() };
    state.events.push(rec);
    learnRunId(ev);
    learnRunId(attrs(ev));
    if (rec.kind === 'decision') {
      state.decisions.push(decRec(ev));
      scheduleDecisions();
    }
    if (rec.kind === 'handoff') scheduleChain();
    if (state.paused) { state.buffer.push(rec); bumpCounters(); }
    else appendRow(ev, true);
    bumpCounters();
    scheduleRefresh(false);
    return rec;
  }

  function decRec(ev) {
    var a = attrs(ev);
    return {
      seq: ev.seq, name: str(ev.name) || '', agent: ev.agent || null,
      question: str(pick(a, ['question'])) || str(ev.name) || '',
      choice: str(pick(a, ['choice'])),
      source: str(pick(a, ['source', 'decision.source'])),
      model: str(pick(a, ['model', 'decision.model'])),
      confidence: numOrNull(pick(a, ['confidence', 'decision.confidence'])),
      degraded: truthy(pick(a, ['degraded', 'decision.degraded'])),
      request_id: str(pick(a, ['request_id'])),
      probs: probsFrom(a),
      ev: ev
    };
  }

  function probsFrom(a) {
    var raw = pick(a, ['probabilities', 'distribution', 'probs', 'choice_probabilities']);
    if ((!raw || typeof raw !== 'object') && a.raw && typeof a.raw === 'object') raw = pick(a.raw, ['probabilities', 'distribution']);
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
    var out = [];
    for (var k in raw) {
      if (!Object.prototype.hasOwnProperty.call(raw, k)) continue;
      var v = numOrNull(raw[k]);
      if (v !== null) out.push([k, v]);
    }
    return out.length ? out : null;
  }

  function scheduleDecisions() {
    if (scheduleDecisions._t) return;
    scheduleDecisions._t = setTimeout(function () { scheduleDecisions._t = null; renderDecisions(); }, 350);
  }
  function scheduleChain() {
    if (scheduleChain._t) return;
    scheduleChain._t = setTimeout(function () { scheduleChain._t = null; renderChain(); }, 250);
  }

  function appendRow(ev, isNew) {
    var rec = state.events[state.events.length - 1];
    var feed = $('#trace-feed');
    var empty = feed.querySelector('.empty--tall');
    if (empty) empty.remove();
    var row = renderRow(ev, isNew);
    if (rec && rec.ev === ev) rec.node = row;
    feed.appendChild(row);
    applyFilterTo(row, normKind(ev.kind));
    if (state.follow) feed.scrollTop = feed.scrollHeight;
    return row;
  }

  function renderRow(ev, isNew) {
    var kind = normKind(ev.kind);
    var a = attrs(ev);
    var seq = (ev.seq === null || ev.seq === undefined) ? '·' : String(ev.seq);
    var d = parseTs(ev.ts);
    var cls = 'row' + (isNew && !reduceMotion ? ' row--new' : '') + (kind === 'handoff' ? ' row--handoff-card' : '');

    var cells = [
      el('span', { class: 'row__seq', text: seq }),
      el('span', { class: 'row__time', text: d ? fmtClock(d) : '—', title: d ? fmtStamp(d) : 'no timestamp sent' }),
      agentBadge(ev.agent)
    ];

    var row;
    if (kind === 'handoff') {
      /* The card already names both ends of the transfer, so the agent column
       * carries the kind chip instead of repeating the sender. */
      cells.push(kindChip(kind));
      var from = normAgent(str(pick(a, ['from_agent', 'handoff.from'])) || ev.agent || '?');
      var to = normAgent(str(pick(a, ['to_agent', 'handoff.to'])) || '?');
      var reason = str(pick(a, ['reason', 'handoff.reason']));
      var src = str(pick(a, ['decision_source', 'handoff.decision_source']));
      var conf = numOrNull(pick(a, ['confidence', 'handoff.confidence']));
      var degraded = src ? !!FALLBACK_SOURCES[src.toLowerCase()] : false;
      var status = str(ev.status) || 'unknown';
      cells.push(el('span', { class: 'handoff-card' + (degraded ? ' handoff-card--degraded' : '') }, [
        el('span', { class: 'badge', 'data-agent': from, title: 'control leaves ' + from, text: from }),
        el('span', { class: 'handoff-card__arrow', text: '──▶' }),
        el('span', { class: 'badge', 'data-agent': to, title: 'control arrives at ' + to, text: to }),
        el('span', { class: 'handoff-card__reason', title: reason || 'no reason field in the event attributes',
          text: reason || '(no reason sent)' }),
        el('span', { class: 'handoff-card__why' }, [
          src ? el('span', { class: 'chip chip--' + (degraded ? 'warn' : 'good'),
            title: 'who decided the routing', text: 'routed by ' + src }) : absent('routing source not sent'),
          conf === null ? absent('confidence') : el('span', { class: 'chip', text: 'conf ' + conf.toFixed(2) }),
          el('span', { class: 'chip chip--muted', title: 'duration_ms', text: fmtMs(ev.duration_ms) || 'no duration sent' }),
          el('span', { class: 'chip chip--muted', style: 'color:' + statusColor(status), text: status })
        ])
      ]));
      row = el('div', { class: cls, 'data-kind': kind, 'data-seq': seq }, cells);
      var refs = pick(a, ['payload_refs']);
      if (Array.isArray(refs) && refs.length) {
        row.appendChild(el('div', { class: 'row__detail' }, [
          el('span', { class: 'chip', text: refs.length + ' blackboard refs the receiving agent should read' }),
          el('span', { class: 'chip chip--muted', title: refs.join(', '), text: refs.slice(0, 3).join(', ') + (refs.length > 3 ? ' …' : '') })
        ]));
      }
      return row;
    }

    cells.push(kindChip(kind));
    var nameCell = el('span', { class: 'row__name' }, [el('span', { text: str(ev.name) || '(no name sent)' })]);
    var extra = rowExtra(ev, kind, a);
    if (extra) nameCell.appendChild(extra);
    cells.push(nameCell);
    cells.push(el('span', { class: 'row__dur', text: fmtMs(ev.duration_ms) || '—',
      title: ev.duration_ms === null || ev.duration_ms === undefined ? 'no duration sent' : String(ev.duration_ms) + ' ms' }));
    var st = str(ev.status) || 'unknown';
    cells.push(el('span', { class: 'row__status', style: 'color:' + statusColor(st), text: st }));

    row = el('div', { class: cls, 'data-kind': kind, 'data-seq': seq }, cells);

    var detail = rowDetail(ev, kind, a);
    if (detail) row.appendChild(detail);

    if (kind === 'error') {
      var msg = str(pick(a, ['error', 'message', 'exception', 'detail', 'reason'])) ||
        str(pick(a, ['error.type'])) || 'event of kind "error" with no message in its attributes';
      row.appendChild(el('div', { class: 'row__err', text: msg }));
    }
    return row;
  }

  function normAgent(a) {
    var id = a === null || a === undefined ? '' : String(a).trim().toUpperCase();
    return id === 'ENVIRONMENT' ? 'ENV' : id;
  }

  function rowExtra(ev, kind, a) {
    if (kind === 'decision') {
      var bits = [];
      var choice = str(pick(a, ['choice']));
      if (choice) bits.push('→ ' + choice);
      var src = str(pick(a, ['source', 'decision.source']));
      if (src) bits.push('[' + src + (truthy(pick(a, ['degraded', 'decision.degraded'])) ? ' degraded' : '') + ']');
      var conf = numOrNull(pick(a, ['confidence']));
      if (conf !== null) bits.push('conf ' + conf.toFixed(2));
      if (!bits.length) return null;
      return el('span', { class: 'row__extra', text: '  ' + bits.join('  ') });
    }
    if (kind === 'tool') {
      var bits2 = [];
      var tool = str(pick(a, ['tool', 'tool_name']));
      var ts = str(pick(a, ['tool_status', 'status']));
      var src2 = str(pick(a, ['source']));
      var target = str(pick(a, ['target', 'url', 'query']));
      /* ev.name is usually already the tool name; repeating it wastes the row. */
      if (tool && tool !== ev.name) bits2.push(tool);
      if (target) bits2.push(target);
      if (ts) bits2.push('status=' + ts);
      if (src2) bits2.push('source=' + src2);
      if (!bits2.length) return null;
      return el('span', { class: 'row__extra', text: '  ' + bits2.join('  ') });
    }
    if (kind === 'human') {
      var outc = str(pick(a, ['outcome']));
      var by = str(pick(a, ['decided_by', 'operator']));
      var instr = str(pick(a, ['instruction']));
      var bits3 = [];
      if (outc) bits3.push('outcome=' + outc);
      if (by) bits3.push('by ' + by);
      if (instr) bits3.push('“' + instr.slice(0, 70) + (instr.length > 70 ? '…' : '') + '”');
      return bits3.length ? el('span', { class: 'row__extra', text: '  ' + bits3.join('  ') }) : null;
    }
    if (kind === 'llm') {
      var m = str(pick(a, ['model']));
      var p = str(pick(a, ['provider']));
      var op = str(pick(a, ['operation', 'gen_ai.operation.name']));
      var bits4 = [];
      if (m) bits4.push(m);
      if (p) bits4.push(p);
      if (op) bits4.push(op);
      var toks = pick(a, ['prompt_tokens', 'completion_tokens', 'total_tokens']);
      if (toks !== undefined) bits4.push('tokens=' + JSON.stringify(toks));
      return bits4.length ? el('span', { class: 'row__extra', text: '  ' + bits4.join('  ') }) : null;
    }
    if (kind === 'agent') {
      var s = str(pick(a, ['summary', 'note', 'result']));
      return s ? el('span', { class: 'row__extra', text: '  ' + s.slice(0, 90) }) : null;
    }
    return null;
  }

  function rowDetail(ev, kind, a) {
    if (kind === 'error') return null;
    var chips = [];
    if (kind === 'tool') {
      var src = str(pick(a, ['source']));
      if (src && /fixture|seed|stub|mock/i.test(src)) chips.push(el('span', { class: 'chip chip--fixture', text: 'fixture-backed, not a live lookup' }));
      var ts = str(pick(a, ['tool_status']));
      if (ts && /unavailable|failed|error/.test(ts)) chips.push(el('span', { class: 'chip chip--bad', text: 'tool ' + ts }));
    }
    if (kind === 'handoff') {
      var refs = pick(a, ['payload_refs']);
      if (Array.isArray(refs) && refs.length) chips.push(el('span', { class: 'chip', text: refs.length + ' blackboard refs' }));
    }
    if (!chips.length) return null;
    return el('div', { class: 'row__detail' }, chips);
  }

  function statusColor(s) {
    var n = String(s).toLowerCase();
    if (n === 'ok') return 'var(--ink-faint)';
    if (n === 'fallback') return 'var(--warn)';
    if (n === 'interrupted') return 'var(--violet)';
    if (n === 'error') return 'var(--bad)';
    return 'var(--ink-faint)';
  }

  function bumpCounters() {
    $('#event-count').textContent = fmtBytesish(state.events.length);
    var sub = [];
    if (state.paused && state.buffer.length) sub.push(state.buffer.length + ' buffered (paused)');
    if (state.dupSuppressed) sub.push(state.dupSuppressed + ' duplicate(s) suppressed');
    if (state.malformed) sub.push(state.malformed + ' unparseable');
    var errs = state.events.filter(function (r) { return r.kind === 'error'; }).length;
    if (errs) sub.push(errs + ' error event(s)');
    $('#event-count-sub').textContent = sub.length ? sub.join(' · ')
      : (state.conn === 'live' ? 'streaming' : (state.conn === 'idle' ? 'stream not connected' : state.conn));
    renderConnStats();
    renderKindFilters();
  }

  function renderConnStats() {
    var bits = [];
    if (state.events.length) {
      var t0 = state.events[0].at, t1 = state.events[state.events.length - 1].at;
      bits.push('span ' + fmtMs(t1.getTime() - t0.getTime()));
    }
    if (state.dupSuppressed) bits.push('dedup: ' + state.dupSuppressed);
    $('#conn-stats').textContent = bits.join('  ·  ');
  }

  function renderKindFilters() {
    var host = clear($('#kind-filter'));
    var counts = {};
    state.events.forEach(function (r) { counts[r.kind] = (counts[r.kind] || 0) + 1; });
    DISPLAY_KINDS.forEach(function (k) {
      var on = state.filters.indexOf(k) >= 0;
      host.appendChild(el('button', {
        type: 'button', class: 'filter', 'aria-pressed': on ? 'true' : 'false',
        style: '--kc:' + (KIND_COLOR[k] || KIND_COLOR.unknown),
        title: 'show/hide ' + k + ' events',
        onclick: function () {
          var i = state.filters.indexOf(k);
          if (i >= 0) state.filters.splice(i, 1); else state.filters.push(k);
          renderKindFilters();
          state.events.forEach(function (r) { applyFilterTo(r.node, r.kind); });
        }
      }, [el('span', { text: k }), el('span', { class: 'filter__n', text: String(counts[k] || 0) })]));
    });
  }

  function applyFilterTo(node, kind) {
    if (!node) return;
    node.hidden = state.filters.indexOf(kind === 'unknown' ? 'unknown' : kind) < 0;
  }

  function renderChain() {
    var host = clear($('#chain-rail'));
    var hops = [];
    state.events.forEach(function (r) {
      if (r.kind !== 'handoff') return;
      var a = attrs(r.ev);
      hops.push({
        from: normAgent(str(pick(a, ['from_agent', 'handoff.from'])) || r.ev.agent || '?'),
        to: normAgent(str(pick(a, ['to_agent', 'handoff.to'])) || '?'),
        reason: str(pick(a, ['reason', 'handoff.reason'])) || '(no reason sent)',
        src: str(pick(a, ['decision_source', 'handoff.decision_source'])),
        conf: numOrNull(pick(a, ['confidence'])),
        seq: r.ev.seq
      });
    });
    var note = $('#chain-note');
    if (!hops.length) {
      host.appendChild(el('span', { class: 'chain__empty', text: 'A1 → A2 → … appears here, one link per recorded handoff' }));
      note.textContent = 'control has not moved yet — no handoff event received';
      return;
    }
    var degraded = hops.filter(function (h) { return h.src && FALLBACK_SOURCES[h.src.toLowerCase()]; }).length;
    note.textContent = hops.length + ' handoff' + (hops.length === 1 ? '' : 's') + ' — control is passing between agents, not following a script'
      + (degraded ? '  ·  ' + degraded + ' routed by a deterministic fallback' : '')
      + (hops[hops.length - 1].conf !== null ? '  ·  last confidence ' + hops[hops.length - 1].conf.toFixed(2) : '');

    /* Both ends of every hop are drawn. Showing only the receiving agent would
     * read as "A2 handed to A2" whenever two hops meet at the same agent, which
     * is exactly the mesh behaviour the chain exists to show. When a hop starts
     * somewhere other than where the last one ended (A2 -> A3 -> A2), the extra
     * sender is inserted rather than letting the rail imply a direct transfer. */
    var frag = document.createDocumentFragment();
    frag.appendChild(chainNode(hops[0].from, false));
    hops.forEach(function (h, i) {
      var cont = i > 0 && hops[i - 1].to === h.from;
      if (i > 0 && !cont) frag.appendChild(chainNode(h.from, false));
      frag.appendChild(hopArrow(h, cont));
      frag.appendChild(chainNode(h.to, i === hops.length - 1));
    });
    host.appendChild(frag);
  }

  function hopArrow(h, cont) {
    var degraded = h.src && FALLBACK_SOURCES[h.src.toLowerCase()];
    return el('span', { class: 'chain-hop', style: '--kc:' + (degraded ? 'var(--warn)' : 'var(--good)') }, [
      el('span', { class: 'chain-line', text: cont ? '──┤' : '──▶' }),
      el('span', { class: 'chain-reason' + (degraded ? ' chain-reason--degraded' : ''),
        title: h.reason + (h.src ? '  (routed by ' + h.src + (h.conf !== null ? ', confidence ' + h.conf.toFixed(2) : '') + ')' : '  (routing source not sent)'),
        text: h.reason })
    ]);
  }

  function chainNode(id, active) {
    var meta = AGENTS[id];
    return el('button', {
      type: 'button', class: 'chain-node' + (active ? ' chain-node--active' : ''), 'data-agent': id,
      title: (meta ? meta.label : 'unknown agent id') + ' — click to jump to its last event',
      onclick: function () { jumpToAgent(id); }
    }, [el('span', { text: id }), el('small', { text: meta ? meta.short : '?' })]);
  }

  function jumpToAgent(id) {
    for (var i = state.events.length - 1; i >= 0; i--) {
      var r = state.events[i];
      if (r.node && normAgent(r.ev.agent) === id) { flash(r); return; }
      if (r.kind === 'handoff' && r.node) { flash(r); return; }
    }
    toast('no rendered event for ' + id + ' yet', 'bad');
  }

  function scrollToEvent(seq) {
    if (seq === null || seq === undefined) return;
    for (var i = state.events.length - 1; i >= 0; i--) {
      var r = state.events[i];
      if (String(r.ev.seq) === String(seq) && r.node) { flash(r); return; }
    }
    toast('event seq ' + seq + ' has not been rendered (not in this stream)', 'bad');
  }

  function flash(rec) {
    if (!rec || !rec.node) return;
    if (rec.node.hidden) rec.node.hidden = false;
    rec.node.scrollIntoView({ block: 'center', behavior: reduceMotion ? 'auto' : 'smooth' });
    rec.node.classList.remove('row--flash');
    void rec.node.offsetWidth;
    rec.node.classList.add('row--flash');
    setTimeout(function () { rec.node.classList.remove('row--flash'); }, 1500);
  }

  function setConn(c) {
    state.conn = c;
    var dot = $('#live-dot');
    dot.setAttribute('data-state', state.paused ? 'paused' : c);
    var cdot = $('#conn-dot');
    cdot.className = 'dot dot--' + (state.paused ? 'paused' : (c === 'idle' ? 'idle' : c));
    $('#conn-text').textContent = state.paused
      ? 'paused — the feed is frozen, panels keep updating'
      : ({ idle: 'not connected', connecting: 'connecting…', live: 'streaming ' + (state.runId || ''), reconnecting: 'reconnecting (EventSource retrying)', down: 'disconnected' }[c] || c);
    $('#trace-connect').disabled = !state.runId;
  }

  function flushBuffer() {
    var n = state.buffer.length;
    while (state.buffer.length) {
      var rec = state.buffer.shift();
      appendRow(rec.ev, true);
    }
    if (n) toast('rendered ' + n + ' buffered event(s)', 'good');
    bumpCounters();
  }

  function togglePause() {
    state.paused = !state.paused;
    var btn = $('#trace-pause');
    btn.textContent = state.paused ? 'Resume' : 'Pause';
    btn.setAttribute('aria-pressed', state.paused ? 'true' : 'false');
    if (!state.paused) flushBuffer();
    setConn(state.conn);
    bumpCounters();
  }

  function clearFeed(note) {
    clear($('#trace-feed')).appendChild(el('p', { class: 'empty empty--tall',
      text: note || 'Feed cleared. The stream stays connected; only the rendered rows were removed. The next (re)connect replays the run from the start.' }));
    state.events = [];
    state.seen = Object.create(null);
    state.decisions = [];
    state.dupSuppressed = 0;
    state.malformed = 0;
    state.buffer = [];
    /* Forgetting the rows means forgetting the resume position too: a later
     * reconnect must replay from the start, otherwise cleared events would
     * stay skipped and never be seen again. */
    state.lastSeq = null;
    renderChain();
    renderDecisions();
    bumpCounters();
  }

  /* Switching runs: the old run's feed, resume watermark and panels must not
   * survive under the new run's id. Called before connectTrace({fresh:true})
   * so the new stream starts at the beginning of the new run. */
  function resetForRun(runId) {
    clearFeed('Switched to run ' + runId + ' — the previous run’s events were cleared. Streaming the new run from its first event.');
    state.summary = null;
    renderSummary();
    state.board = null;
    state.tree = '';
    renderBoard();
    state.disputes = [];
    state.lessons = [];
    state.handoffs = [];
    state.coordLoadedFor = '';
    renderCoordination();
  }

  function scheduleRefresh(force) {
    var t = now();
    if (!force && t - (scheduleRefresh._last || 0) < 2500) return;
    scheduleRefresh._last = t;
    if (refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(function () {
      refreshGates();
      if (state.runId) { refreshSummary(); refreshBoard(); refreshCoordination(); }
    }, 400);
  }

  /* Fallback for when the event stream cannot be used at all: a proxy that
     * buffers, a CORS policy that blocks EventSource, a demo on hotel wifi.
     * The same ingest path runs, so the de-dup memory means mixing a fetched
     * trace with a live stream cannot double-render a seq. */
  function fetchTrace() {
    if (!state.runId) { toast('no run_id yet', 'bad'); return Promise.resolve(null); }
    var p = '/api/runs/' + enc(state.runId) + '/trace';
    setStatus('#trace-status-note', 'GET ' + p + ' …', 'busy');
    return request(p, { timeoutMs: 20000 }).then(function (res) {
      if (superseded(res)) return res;
      if (reportNetFailure(res, 'GET ' + p, 'trace')) {
        setStatus('#trace-status-note', 'unreachable', 'err');
        return res;
      }
      if (!res.ok) {
        setStatus('#trace-status-note', 'HTTP ' + res.status + ' — ' + (shortDetail(res.json) || 'no detail'), 'err');
        return res;
      }
      var list = Array.isArray(res.json) ? res.json
        : listOf(res.json, ['trace', 'events', 'items', 'records']);
      if (!Array.isArray(list)) {
        setStatus('#trace-status-note', 'unrecognised body shape', 'err');
        return res;
      }
      var added = 0;
      list.forEach(function (ev) { if (ingestEvent(ev)) added++; });
      state.paused = false;
      $('#trace-pause').textContent = 'Pause';
      $('#trace-pause').setAttribute('aria-pressed', 'false');
      setConn(state.conn);
      flushBuffer();
      setStatus('#trace-status-note', 'loaded ' + added + ' new of ' + list.length + ' from ' + p, 'ok');
      renderChain();
      renderDecisions();
      return res;
    });
  }

  /* ========================= decision inspector ========================= */

  function renderDecisions() {
    var host = clear($('#decisions-host'));
    var list = state.decisions;
    var tally = { model: 0, fallback: 0, degraded: 0, unnamed: 0 };
    list.forEach(function (d) {
      var s = (d.source || '').toLowerCase();
      if (!d.source) tally.unnamed++;
      else if (FALLBACK_SOURCES[s]) tally.fallback++;
      else if (MODEL_SOURCES[s]) tally.model++;
      else tally.unnamed++;
      if (d.degraded) tally.degraded++;
    });
    $('#dec-tally').replaceChildren(
      el('span', { class: 'chip chip--good', text: tally.model + ' model' }),
      el('span', { class: 'chip chip--warn', text: tally.fallback + ' fallback' }),
      el('span', { class: 'chip chip--' + (tally.degraded ? 'bad' : 'muted'), text: tally.degraded + ' degraded' })
    );

    if (!list.length) {
      host.appendChild(el('p', { class: 'empty',
        text: 'No decision events yet. Every decision carries the subsystem that produced it: model-driven (clef/gemini) or deterministic fallback (rules/replay).' }));
      return;
    }
    var shown = list.slice(-40).reverse();
    if (list.length > 40) {
      host.appendChild(el('p', { class: 'note note--tight',
        text: 'Showing the 40 most recent of ' + list.length + ' decisions.' }));
    }
    shown.forEach(function (d) { host.appendChild(decCard(d)); });
  }

  function decCard(d) {
    var src = (d.source || '').toLowerCase();
    var isFallback = !!FALLBACK_SOURCES[src];
    var isModel = !!MODEL_SOURCES[src];
    var tone = !d.source ? 'unknown' : isFallback ? 'warn' : isModel ? 'good' : 'accent';
    var card = el('div', { class: 'dec', 'data-source': src || 'none' }, [
      el('div', { class: 'dec__top' }, [
        agentBadge(d.agent),
        el('span', { class: 'chip chip--' + tone, title: d.source ? 'DecisionSource: ' + d.source : 'no source field in the event',
          text: d.source ? (isModel ? 'MODEL · ' + d.source : isFallback ? 'FALLBACK · ' + d.source : d.source) : 'source not sent' }),
        d.degraded ? el('span', { class: 'chip chip--bad', title: 'degraded=true: a backend failed and this answered instead', text: 'DEGRADED' }) : null,
        d.model ? el('span', { class: 'chip chip--muted', text: d.model }) : null,
        el('span', { class: 'chip chip--muted', text: 'seq ' + (d.seq === null || d.seq === undefined ? '·' : d.seq) })
      ]),
      el('p', { class: 'dec__q', text: d.question }),
      d.choice ? el('div', { class: 'dec__choice', text: 'chose: ' + d.choice }) : absent('no choice sent'),
      distribution(d),
      el('div', { class: 'dec__foot' }, [
        el('span', { class: 'chip', text: 'confidence ' + (d.confidence === null ? '—' : d.confidence.toFixed(3)) }),
        d.confidence === null ? absent('confidence not sent') : null,
        d.request_id ? el('span', { class: 'chip chip--muted', text: d.request_id }) : null,
        el('button', { type: 'button', class: 'dec__jump', text: 'jump to trace row',
          onclick: function () { scrollToEvent(d.seq); } })
      ])
    ]);
    return card;
  }

  function distribution(d) {
    var wrap = el('div', { class: 'dist' });
    if (d.probs && d.probs.length) {
      d.probs.forEach(function (p) {
        var chosen = d.choice && p[0] === d.choice;
        var pct = Math.max(0, Math.min(100, p[1] * 100));
        wrap.appendChild(el('div', { class: 'dist__row' + (chosen ? ' dist__row--chosen' : '') }, [
          el('span', { class: 'dist__label', title: p[0], text: p[0] }),
          el('span', { class: 'dist__track' }, [el('span', { class: 'dist__fill', style: 'width:' + pct.toFixed(2) + '%' })]),
          el('span', { class: 'dist__val', text: (p[1] * 100).toFixed(1) + '%' })
        ]));
      });
      var sum = d.probs.reduce(function (a, p) { return a + p[1]; }, 0);
      wrap.appendChild(el('div', { class: 'dec__absent',
        text: 'distribution as sent by the backend (sums to ' + sum.toFixed(3) + ')' }));
      return wrap;
    }
    /* No distribution sent. Show the confidence we DO have, labelled exactly for
     * what it is, and say plainly that the full distribution was not transmitted. */
    wrap.appendChild(el('div', { class: 'dec__absent',
      text: 'the backend did not send a probability distribution with this event — only the confidence below is available, so nothing else is drawn' }));
    if (d.confidence !== null) {
      wrap.appendChild(el('div', { class: 'dist__row dist__row--chosen' }, [
        el('span', { class: 'dist__label', title: 'confidence', text: 'confidence (probability of the choice taken)' }),
        el('span', { class: 'dist__track' }, [el('span', { class: 'dist__fill', style: 'width:' + (d.confidence * 100).toFixed(2) + '%' })]),
        el('span', { class: 'dist__val', text: (d.confidence * 100).toFixed(1) + '%' })
      ]));
    }
    return wrap;
  }

  /* ============================== blackboard ============================== */

  function renderBoard() {
    var host = clear($('#board-host'));
    var sn = staleNote('board', 'this snapshot');
    if (sn) host.appendChild(sn);
    if (!state.board) {
      host.appendChild(el('p', { class: 'empty',
        text: state.runId ? 'Not loaded — press Refresh (GET /api/runs/' + state.runId + '/board).' : 'No run_id yet, so the board cannot be fetched.' }));
      return;
    }
    var snap = state.board;
    var entries = Array.isArray(snap.entries) ? snap.entries : [];
    var zones = Array.isArray(snap.zones) ? snap.zones : [];

    var stats = snap.stats && typeof snap.stats === 'object' ? snap.stats : null;
    if (stats) {
      var kv = el('div', { class: 'kv' });
      Object.keys(stats).forEach(function (k) {
        kv.appendChild(el('span', { class: 'kv__k', text: k }));
        kv.appendChild(el('span', { class: 'kv__v', text: JSON.stringify(stats[k]) }));
      });
      host.appendChild(kv);
    }
    host.appendChild(el('p', { class: 'note note--tight',
      text: (str(snap.captured_at) ? 'snapshot ' + snap.captured_at + ' · ' : '') +
        entries.length + ' entries · ' + zones.length + ' zones' + (snap.kind ? ' · ' + snap.kind : '') }));

    if (!entries.length) {
      host.appendChild(el('p', { class: 'empty', text: 'The snapshot contains no entries — either the board is empty or the run posted nothing.' }));
    }

    var zoneOrder = zones.slice();
    entries.forEach(function (e) {
      if (e && zoneOrder.indexOf(e.zone) < 0) zoneOrder.push(e.zone);
    });

    zoneOrder.forEach(function (z) {
      var list = entries.filter(function (e) { return e && e.zone === z; });
      var zone = el('div', { class: 'bb-zone' }, [
        el('div', { class: 'bb-zone__head' }, [
          el('span', { class: 'bb-zone__name', text: z }),
          el('span', { class: 'bb-zone__owner', text: 'zone' }),
          el('span', { class: 'bb-zone__n', text: list.length + ' entr' + (list.length === 1 ? 'y' : 'ies') })
        ])
      ]);
      list.forEach(function (e) { zone.appendChild(bbEntry(e)); });
      host.appendChild(zone);
    });

    renderTree();
  }

  function bbEntry(e) {
    var a = e || {};
    var src = str(a.source);
    return el('div', { class: 'bb-entry' }, [
      el('div', { class: 'bb-entry__top' }, [
        el('span', { class: 'bb-entry__seq', text: '#' + (a.seq === undefined ? '?' : a.seq) }),
        agentBadge(a.author),
        el('span', { class: 'bb-entry__kind', text: str(a.kind) || '(no kind sent)' }),
        el('span', { class: 'bb-entry__meta' }, [
          'c=' + (numOrNull(a.confidence) === null ? '—' : numOrNull(a.confidence).toFixed(2)),
          src ? '  src=' + src : ''
        ])
      ]),
      Array.isArray(a.refs) && a.refs.length
        ? el('div', { class: 'bb-entry__refs', text: 'cites: ' + a.refs.join(', ') })
        : null,
      a.payload && typeof a.payload === 'object' && Object.keys(a.payload).length
        ? el('details', { class: 'bb-entry__payload' }, [
            el('summary', { text: 'payload (' + Object.keys(a.payload).length + ' fields)' }),
            el('pre', { class: 'pre', text: JSON.stringify(a.payload, null, 2).slice(0, 2500) })
          ])
        : el('div', { class: 'bb-entry__refs', text: 'payload: not sent' })
    ]);
  }

  function renderTree() {
    var host = $('#board-host');
    var existing = host.querySelector('.tree-block');
    if (existing) existing.remove();
    if ($('#board-tree-toggle').getAttribute('aria-pressed') !== 'true') return;
    var block = el('div', { class: 'tree-block' }, [
      el('p', { class: 'section-label', text: 'derivation view (collapsible)' })
    ]);
    var text = state.tree;
    if (typeof text !== 'string' || !text.trim()) {
      block.appendChild(el('p', { class: 'empty', text: 'GET /api/runs/{run_id}/board/tree returned no text.' }));
      host.appendChild(block);
      return;
    }
    var lines = text.split(/\r?\n/);
    var header = [];
    var nodes = [];
    lines.forEach(function (raw) {
      var lead = raw.match(/^ */)[0].length;
      var t = raw.trim();
      if (!t) return;
      if (t.charAt(0) === '#') nodes.push({ depth: Math.floor(lead / 2), text: t });
      else if (header.length < 3) header.push(t);
    });
    header.forEach(function (h) { block.appendChild(el('div', { class: 'note note--tight', text: h })); });
    if (!nodes.length) {
      block.appendChild(el('pre', { class: 'pre', text: text }));
      host.appendChild(block);
      return;
    }
    var treeHost = el('div', { class: 'tree' });
    var stack = [];
    nodes.forEach(function (n) {
      var details = el('details', { class: 'tree-node' });
      var summary = el('summary', null, [el('span', { class: 'tree-node__line', text: n.text })]);
      var kids = el('div', { class: 'tree-kids' });
      details.appendChild(summary);
      details.appendChild(kids);
      while (stack.length && stack[stack.length - 1].depth >= n.depth) stack.pop();
      var parent = stack.length ? stack[stack.length - 1].kids : treeHost;
      parent.appendChild(details);
      stack.push({ depth: n.depth, kids: kids });
    });
    block.appendChild(treeHost);
    block.appendChild(el('p', { class: 'note note--tight',
      text: 'Collapsed by default: ' + nodes.length + ' entries in the citation ladder. Open a row to see the board entries it rests on.' }));
    host.appendChild(block);
  }

  function refreshBoard() {
    if (!state.runId) return Promise.resolve(null);
    var p = '/api/runs/' + enc(state.runId) + '/board';
    return request(p, { timeoutMs: 15000 }).then(function (res) {
      if (superseded(res)) return res;
      if (reportNetFailure(res, 'GET ' + p, 'board')) { markStale('board'); renderBoard(); return res; }
      if (!res.ok) {
        clear($('#board-host')).appendChild(el('p', { class: 'note note--bad',
          text: 'GET ' + p + ' → HTTP ' + res.status + ' — ' + (shortDetail(res.json) || 'no detail in body') }));
        return res;
      }
      clearIssue('board');
      var snap = res.json;
      if (snap && (snap.snapshot || snap.board)) snap = snap.snapshot || snap.board;
      state.board = (snap && typeof snap === 'object') ? snap : null;
      markFresh('board');
      renderBoard();
      var tp = '/api/runs/' + enc(state.runId) + '/board/tree';
      return request(tp, { timeoutMs: 15000 }).then(function (tr) {
        /* The derivation view is served as text/plain, so the text body is the
         * payload; a JSON envelope is accepted too but never assumed. */
        if (!tr.ok) state.tree = '';
        else if (tr.json && typeof tr.json === 'object') {
          state.tree = str(pick(tr.json, ['tree', 'text', 'derivation', 'rendered']))
            || JSON.stringify(tr.json, null, 2);
        } else state.tree = tr.text || '';
        renderTree();
        return tr;
      });
    });
  }

  /* ========================= disputes, lessons, delta ===================== */

  function listOf(json, keys) {
    if (Array.isArray(json)) return json;
    if (json && typeof json === 'object') {
      for (var i = 0; i < keys.length; i++) {
        var v = pick(json, [keys[i]]);
        if (Array.isArray(v)) return v;
      }
    }
    return [];
  }

  function refreshCoordination() {
    if (!state.runId) return Promise.resolve(null);
    var base = '/api/runs/' + enc(state.runId);
    return Promise.all([
      request(base + '/disputes', { timeoutMs: 12000 }),
      request(base + '/lessons', { timeoutMs: 12000 }),
      request(base + '/handoffs', { timeoutMs: 12000 })
    ]).then(function (rs) {
      var names = ['disputes', 'lessons', 'handoffs'];
      var outs = [];
      var failed = false;
      rs.forEach(function (res, i) {
        /* Always fill a slot: a partial failure must not leave state.lessons
         * undefined, which would throw two frames later inside the delta panel. */
        if (superseded(res)) { outs.push([]); failed = true; return; }
        if (reportNetFailure(res, 'GET ' + base + '/' + names[i], names[i])) { outs.push([]); failed = true; return; }
        if (!res.ok) { outs.push([]); failed = true; return; }
        clearIssue(names[i]);
        outs.push(listOf(res.json, [names[i], 'items', 'records']));
      });
      if (failed) markStale('coord'); else markFresh('coord');
      state.disputes = outs[0]; state.lessons = outs[1]; state.handoffs = outs[2];
      state.coordLoadedFor = state.runId;
      renderCoordination();
      maybeRecordRun();
      return rs;
    });
  }

  function renderCoordination() {
    var host = clear($('#coord-host'));
    var sn = staleNote('coord', 'these disputes, lessons and handoffs');
    if (sn) host.appendChild(sn);

    host.appendChild(el('div', { class: 'section-label', text: 'disputes · ' + state.disputes.length }));
    if (!state.disputes.length) {
      host.appendChild(el('p', { class: 'empty', text: state.runId ? 'No disputes recorded for this run. Disagreement is authored by agents as artefacts, not inferred from text.' : 'No run_id yet.' }));
    } else state.disputes.forEach(function (d) { host.appendChild(disputeCard(d)); });

    host.appendChild(el('div', { class: 'section-label', text: 'lessons · ' + state.lessons.length }));
    if (!state.lessons.length) {
      host.appendChild(el('p', { class: 'empty', text: state.runId ? 'No lessons stored for this run. A lesson is written when a dispute resolves.' : 'No run_id yet.' }));
    } else state.lessons.forEach(function (l) { host.appendChild(lessonCard(l)); });

    host.appendChild(el('div', { class: 'section-label', text: 'what changed because of a lesson' }));
    host.appendChild(deltaPanel());

    host.appendChild(el('div', { class: 'section-label', text: 'handoffs · ' + state.handoffs.length }));
    if (!state.handoffs.length) {
      host.appendChild(el('p', { class: 'empty', text: state.runId ? 'No handoff records returned. Handoffs seen on the live stream are drawn in the chain rail above.' : 'No run_id yet.' }));
    } else {
      var tbl = el('div', {});
      state.handoffs.slice(0, 24).forEach(function (h) {
        var src = str(pick(h, ['decision_source']));
        var fb = src && FALLBACK_SOURCES[src.toLowerCase()];
        tbl.appendChild(el('div', { class: 'delta__row' }, [
          el('span', { class: 'badge badge--sm', 'data-agent': normAgent(pick(h, ['from_agent'])), text: normAgent(pick(h, ['from_agent'])) }),
          el('span', { class: 'delta__k', text: '──▶' }),
          el('span', { class: 'badge badge--sm', 'data-agent': normAgent(pick(h, ['to_agent'])), text: normAgent(pick(h, ['to_agent'])) }),
          el('span', { class: 'delta__k', text: str(pick(h, ['reason'])) || '(no reason sent)' }),
          el('span', { class: 'chip chip--' + (fb ? 'warn' : src ? 'good' : 'muted'), text: src || 'source not sent' })
        ]));
      });
      host.appendChild(tbl);
      if (state.handoffs.length > 24) host.appendChild(el('p', { class: 'note note--tight', text: (state.handoffs.length - 24) + ' more not shown.' }));
    }
  }

  function disputeCard(d) {
    var sev = str(pick(d, ['severity'])) || 'unknown';
    return el('div', { class: 'disp', 'data-severity': sev }, [
      el('div', { class: 'disp__head' }, [
        el('span', { class: 'chip chip--' + (sev === 'blocking' ? 'bad' : sev === 'medium' ? 'warn' : 'muted'), text: 'severity ' + sev }),
        el('span', { class: 'chip chip--' + (str(pick(d, ['status'])) === 'resolved' ? 'good' : 'accent'), text: 'status ' + (str(pick(d, ['status'])) || 'not sent') }),
        el('span', { class: 'chip chip--muted', text: 'zone ' + (str(pick(d, ['zone'])) || 'not sent') }),
        numOrNull(pick(d, ['rounds'])) !== null ? el('span', { class: 'chip chip--muted', text: 'rounds ' + numOrNull(pick(d, ['rounds'])) }) : null
      ]),
      el('div', { class: 'disp__vs' }, [
        el('span', { class: 'badge', 'data-agent': normAgent(pick(d, ['claimant'])), text: normAgent(pick(d, ['claimant'])) }),
        el('span', { class: 'disp__side' }, [el('b', { text: 'position' }), ' ',
          str(pick(d, ['position'])) || absent('no position sent')]),
        el('span', { class: 'disp__side' }, [el('b', { text: 'counter-position' }), ' ',
          str(pick(d, ['counter_position'])) || absent('no counter-position sent')]),
        Array.isArray(pick(d, ['evidence'])) && d.evidence.length
          ? el('span', { class: 'disp__side' }, [el('b', { text: 'evidence' }), ' ', d.evidence.join(', ')])
          : el('span', { class: 'disp__side' }, [el('b', { text: 'evidence' }), ' ', absent('no evidence refs sent')])
      ]),
      str(pick(d, ['resolution'])) || str(pick(d, ['resolved_by']))
        ? el('div', { class: 'disp__resolution', text: 'resolved' + (str(pick(d, ['resolved_by'])) ? ' by ' + pick(d, ['resolved_by']) : '') + ': ' + (str(pick(d, ['resolution'])) || '(no resolution text sent)') })
        : null
    ]);
  }

  function lessonCard(l) {
    var rule = str(pick(l, ['rule']));
    return el('div', { class: 'lesson' }, [
      el('div', { class: 'lesson__head' }, [
        agentBadge(pick(l, ['author'])),
        numOrNull(pick(l, ['confidence'])) !== null ? el('span', { class: 'chip chip--muted', text: 'confidence ' + numOrNull(pick(l, ['confidence'])).toFixed(2) }) : null,
        str(pick(l, ['lesson_id'])) ? el('span', { class: 'chip chip--muted', text: str(pick(l, ['lesson_id'])) }) : null,
        str(pick(l, ['trigger_dispute_id'])) ? el('span', { class: 'chip chip--warn', text: 'from ' + str(pick(l, ['trigger_dispute_id'])) }) : null
      ]),
      el('div', { class: 'lesson__trigger', text: 'trigger: ' + (str(pick(l, ['trigger'])) || 'not sent') }),
      rule ? el('div', { class: 'lesson__rule', text: 'rule: ' + rule }) : el('div', { class: 'lesson__rule' }, [absent('no rule text sent')]),
      str(pick(l, ['correction'])) ? el('div', { class: 'lesson__correction', text: 'correction: ' + pick(l, ['correction']) }) : null
    ]);
  }

  /* ---------- run history + delta panel ---------- */

  function runHistory() {
    try {
      var v = JSON.parse(lsGet(LS.history, '[]'));
      return Array.isArray(v) ? v : [];
    } catch (e) { return []; }
  }
  function runHistoryPush(rec) {
    var h = runHistory().filter(function (r) { return r.run_id !== rec.run_id; });
    h.push(rec);
    while (h.length > 6) h.shift();
    lsSet(LS.history, JSON.stringify(h));
  }

  function maybeRecordRun() {
    if (!state.runId || !state.summary) return;
    if (state.coordLoadedFor !== state.runId) return;
    if (!Array.isArray(state.lessons) || !Array.isArray(state.disputes)) return;
    var s = state.summary;
    runHistoryPush({
      run_id: state.runId, at: new Date().toISOString(),
      lessons: state.lessons.map(function (l) {
        return { id: str(pick(l, ['lesson_id'])), rule: str(pick(l, ['rule'])), trigger: str(pick(l, ['trigger'])) };
      }),
      disputes: state.disputes.length,
      event_count: numOrNull(s.event_count), handoff_count: numOrNull(s.handoff_count),
      decision_counts: s.decision_counts || null, degraded: numOrNull(s.degraded_decision_count),
      wall_clock_ms: numOrNull(s.wall_clock_ms), gaps: numOrNull(s.distinct_gap_values)
    });
    renderCoordination();
  }

  function deltaPanel() {
    var h = runHistory();
    var wrap = el('div', { class: 'delta' });
    if (h.length < 2) {
      wrap.appendChild(el('div', { class: 'delta__title', text: 'needs two runs' }));
      wrap.appendChild(el('p', { class: 'note note--tight',
        text: h.length === 1
          ? '1 run recorded in this browser (' + h[0].run_id + '). Run the pipeline again under a new run_id and this panel will diff the two — including which lessons are new and what moved as a result.'
          : 'No runs recorded yet. Every run this browser loads is recorded here (run_id, lessons, counts) so the next run can be compared against it.' }));
      wrap.appendChild(el('button', { type: 'button', class: 'btn btn--xs btn--ghost',
        onclick: function () { lsDel(LS.history); renderCoordination(); toast('run history cleared', 'good'); },
        text: 'clear recorded runs' }));
      return wrap;
    }
    var prev = h[h.length - 2], cur = h[h.length - 1];
    wrap.appendChild(el('div', { class: 'delta__title', text: cur.run_id + '  vs  ' + prev.run_id }));
    var prevRules = {};
    prev.lessons.forEach(function (l) { if (l.id) prevRules[l.id] = l; if (l.rule) prevRules['r:' + l.rule] = l; });

    var fresh = cur.lessons.filter(function (l) { return !prevRules[l.id] && !prevRules['r:' + l.rule]; });
    if (!fresh.length) {
      wrap.appendChild(el('p', { class: 'note note--tight', text: 'No new lessons in this run. A lesson only appears once a dispute has resolved.' }));
    } else {
      fresh.forEach(function (l) {
        var used = eventsMention(l.id);
        wrap.appendChild(el('div', { class: 'delta__row' }, [
          el('span', { class: 'chip chip--warn', text: 'new lesson' }),
          el('span', { class: 'delta__k', text: (l.trigger || 'trigger not sent').slice(0, 70) })
        ]));
        wrap.appendChild(el('div', { class: 'lesson__rule', text: l.rule || 'rule text not sent' }));
        wrap.appendChild(el('div', { class: 'delta__row' }, [
          el('span', { class: 'delta__now', text: used ? 'cited by ' + used + ' event(s) in this trace' : 'no event in this trace cites it by id' })
        ]));
      });
    }
    [['events', 'event_count'], ['handoffs', 'handoff_count'], ['degraded decisions', 'degraded'],
     ['distinct gap values', 'gaps'], ['disputes', 'disputes'], ['lessons', null]].forEach(function (pair) {
      var a = prev[pair[1] || pair[0]], b = cur[pair[1] || pair[0]];
      if (a === null || a === undefined || b === null || b === undefined) return;
      var changed = a !== b;
      wrap.appendChild(el('div', { class: 'delta__row' }, [
        el('span', { class: 'delta__k', text: pair[0] + ':' }),
        el('span', { class: 'delta__was', text: String(a) }),
        el('span', { class: 'delta__k', text: '→' }),
        el('span', { class: changed ? 'delta__now' : '', text: String(b) })
      ]));
    });
    return wrap;
  }

  function eventsMention(id) {
    if (!id) return 0;
    var n = 0;
    for (var i = 0; i < state.events.length && i < 4000; i++) {
      try { if (JSON.stringify(state.events[i].ev).indexOf(id) >= 0) n++; } catch (e) { /* ignore */ }
    }
    return n;
  }

  /* =============================== summary =============================== */

  function applySummary(json) {
    if (!json || typeof json !== 'object') return;
    /* A terminal `done` frame carries terminated_by + event_count but no
     * summary content. It is owned by handleDone; accepting it here would
     * overwrite the real summary panel with bookkeeping. The live `summary`
     * frame always carries a nested summary object or gap statistics, and
     * GET /summary never carries terminated_by, so neither is affected. */
    if (json.terminated_by !== undefined
        && !(json.summary && typeof json.summary === 'object')
        && json.distinct_gap_values === undefined
        && json.decision_counts === undefined) return;
    var body = json.summary && typeof json.summary === 'object' ? json.summary : json;
    if (body.distinct_gap_values === undefined && body.event_count === undefined && body.decision_counts === undefined) return;
    state.summary = body;
    renderSummary();
    maybeRecordRun();
  }

  function refreshSummary() {
    if (!state.runId) {
      clear($('#summary-host')).appendChild(el('p', { class: 'empty', text: 'No run_id yet, so /summary cannot be called.' }));
      return Promise.resolve(null);
    }
    var p = '/api/runs/' + enc(state.runId) + '/summary';
    return request(p, { timeoutMs: 12000 }).then(function (res) {
      if (superseded(res)) return res;
      if (reportNetFailure(res, 'GET ' + p, 'summary')) {
        markStale('summary');
        clear($('#summary-host')).appendChild(el('p', { class: 'note note--bad', text: describeFailure(res, 'GET ' + p) }));
        return res;
      }
      if (!res.ok) {
        state.summary = null;
        clear($('#summary-host')).appendChild(el('p', { class: 'note note--bad',
          text: 'GET ' + p + ' → HTTP ' + res.status + ' — ' + (shortDetail(res.json) || 'no detail in body') +
            '. The backend is reachable; it simply has no summary for this run yet.' }));
        return res;
      }
      clearIssue('summary');
      var body = res.json && res.json.summary && typeof res.json.summary === 'object' ? res.json.summary : res.json;
      state.summary = body;
      markFresh('summary');
      renderSummary();
      maybeRecordRun();
      return res;
    });
  }

  function renderSummary() {
    var host = clear($('#summary-host'));
    var sn = staleNote('summary', 'this summary');
    if (sn) host.appendChild(sn);
    var s = state.summary;
    if (!s) {
      host.appendChild(el('p', { class: 'empty', text: 'No summary loaded for this run.' }));
      return;
    }
    host.appendChild(gapHero(s));
    host.appendChild(statGrid(s));
    host.appendChild(el('div', { class: 'section-label', text: 'decisions by source' }));
    host.appendChild(sourceRow(s));
    host.appendChild(el('div', { class: 'section-label', text: 'tool calls by status' }));
    host.appendChild(toolRow(s));
    host.appendChild(el('div', { class: 'section-label', text: 'provenance' }));
    host.appendChild(provenance(s));
  }

  function gapHero(s) {
    var gaps = Array.isArray(s.consecutive_start_gaps_ms) ? s.consecutive_start_gaps_ms : null;
    var gapCount = gaps ? gaps.length : null;
    var distinct = numOrNull(s.distinct_gap_values);
    var ops = numOrNull(s.span_count) || numOrNull(s.event_count) || 0;

    var verdict, cls;
    if (distinct === null) {
      verdict = 'the summary did not include distinct_gap_values';
      cls = 'gap-hero--unknown';
    } else if (ops < 3) {
      verdict = 'too few operations (' + ops + ') for the timing check to mean anything — read it together with the event count';
      cls = 'gap-hero--unknown';
    } else if (distinct === 1 && (gapCount === null || gapCount >= 3)) {
      verdict = 'every gap is the same value — that is the signature of a hand-written trace, not a capture';
      cls = 'gap-hero--suspect';
    } else if (distinct >= 8) {
      verdict = 'irregular inter-event timing, consistent with a genuine execution capture';
      cls = 'gap-hero';
    } else {
      verdict = 'inconclusive on its own — a few distinct values is possible in a very short run';
      cls = 'gap-hero--unknown';
    }

    var hero = el('div', { class: 'gap-hero ' + cls }, [
      el('div', { class: 'gap-hero__top' }, [
        distinct === null ? absent('not sent') : el('span', { class: 'gap-hero__n', text: String(distinct) }),
        el('span', { class: 'gap-hero__label', text: 'distinct inter-event gap values' }),
        gapCount !== null ? el('span', { class: 'chip chip--muted', text: gapCount + ' gaps from ' + ops + ' operations' }) : null
      ]),
      el('div', { class: 'gap-hero__verdict', text: verdict }),
      el('div', { class: 'gap-hero__explain',
        text: 'Why this is the anti-fabrication check: real work has irregular timing — a tool call that waits on the network, a model call that retries, a debate that settles early — so consecutive start times differ, to sub-millisecond resolution, on every run. A trace typed by hand advances on a round constant, so its gaps collapse to exactly one distinct value. Anyone can recompute this from the raw gap list below; nobody has to take the run’s word for it.' })
    ]);
    if (gaps && gaps.length) {
      var sample = gaps.slice(0, 14).map(function (g) { return fmtMs(g) || String(g); }).join('  ·  ');
      hero.appendChild(el('div', { class: 'gap-hero__gaps', text: 'first gaps: ' + sample + (gaps.length > 14 ? '  … (' + gaps.length + ' total)' : '') }));
    } else {
      hero.appendChild(el('div', { class: 'gap-hero__gaps' }, [absent('the summary did not include consecutive_start_gaps_ms, so the number cannot be audited here')]));
    }
    hero.appendChild(el('div', { class: 'note note--tight',
      text: 'Caveat, stated rather than hidden: this detects synthetic cadence, not sophistication. It is a tripwire against the cheap failure mode, and it says nothing about a run with fewer than three operations.' }));
    return hero;
  }

  function statGrid(s) {
    var grid = el('div', { class: 'stat-grid' });
    var degraded = numOrNull(s.degraded_decision_count);
    grid.appendChild(stat(numOrNull(s.event_count), 'events', ''));
    grid.appendChild(stat(numOrNull(s.handoff_count), 'handoffs', 'good'));
    grid.appendChild(stat(numOrNull(s.llm_call_count), 'llm calls', ''));
    grid.appendChild(stat(numOrNull(s.tool_call_count), 'tool calls', ''));
    grid.appendChild(stat(degraded, 'degraded decisions', degraded ? 'warn' : 'good'));
    grid.appendChild(stat(numOrNull(s.human_gate_count), 'human gates', ''));
    grid.appendChild(stat(numOrNull(s.span_count), 'spans', ''));
    grid.appendChild(stat(fmtMs(s.wall_clock_ms), 'wall clock', ''));
    return grid;
  }

  function stat(n, k, tone) {
    return el('div', { class: 'stat' + (tone ? ' stat--' + tone : '') }, [
      el('div', { class: 'stat__n' }, [typeof n === 'string' ? el('span', { text: n }) : (n === null ? absent() : el('span', { text: String(n) }))]),
      el('div', { class: 'stat__k', text: k })
    ]);
  }

  function sourceRow(s) {
    var counts = (s.decision_counts && typeof s.decision_counts === 'object') ? s.decision_counts : null;
    var row = el('div', { class: 'panel__tools', style: 'justify-content:flex-start' });
    if (!counts || !Object.keys(counts).length) {
      row.appendChild(el('span', { class: 'chip chip--muted', text: 'decision_counts not sent' }));
      return row;
    }
    Object.keys(counts).forEach(function (k) {
      var key = k.toLowerCase();
      var fb = !!FALLBACK_SOURCES[key], model = !!MODEL_SOURCES[key];
      row.appendChild(el('span', { class: 'chip chip--' + (fb ? 'warn' : model ? 'good' : 'muted'),
        title: fb ? 'deterministic fallback — not a model call' : model ? 'model-driven' : 'unrecognised source name',
        text: (fb ? 'fallback ' : model ? 'model ' : '') + k + ': ' + counts[k] }));
    });
    var total = Object.keys(counts).reduce(function (a, k) { return a + (Number(counts[k]) || 0); }, 0);
    row.appendChild(el('span', { class: 'chip chip--muted', text: 'total ' + total }));
    return row;
  }

  function toolRow(s) {
    var counts = (s.tool_status_counts && typeof s.tool_status_counts === 'object') ? s.tool_status_counts : null;
    var row = el('div', { class: 'panel__tools', style: 'justify-content:flex-start' });
    if (!counts || !Object.keys(counts).length) {
      row.appendChild(el('span', { class: 'chip chip--muted', text: 'tool_status_counts not sent (a run with no tools leaves this empty — that is a fact, not a gap)' }));
      return row;
    }
    Object.keys(counts).forEach(function (k) {
      var n = String(k).toLowerCase();
      var tone = /^(ok|live)$/.test(n) ? 'good' : /cache|fixture/.test(n) ? 'warn' : /fail|unavail|error/.test(n) ? 'bad' : 'muted';
      row.appendChild(el('span', { class: 'chip chip--' + tone, text: k + ': ' + counts[k] }));
    });
    return row;
  }

  function provenance(s) {
    var kv = el('div', { class: 'kv' });
    [['run_id', s.run_id], ['trace_file', s.trace_file], ['git_commit', s.git_commit],
     ['code_sha256', s.code_sha256], ['generated_at', s.generated_at], ['root_span_count', s.root_span_count]]
      .forEach(function (pair) {
        kv.appendChild(el('span', { class: 'kv__k', text: pair[0] }));
        var v = pair[1];
        if (v === null || v === undefined || v === '') kv.appendChild(el('span', { class: 'kv__v' }, [absent()]));
        else {
          var text = String(v);
          kv.appendChild(el('span', { class: 'kv__v', title: text,
            text: /sha256|commit/.test(pair[0]) ? text.slice(0, 18) + (text.length > 18 ? '…' : '') : text }));
        }
      });
    return kv;
  }

  function copySummary() {
    if (!state.summary) { toast('no summary loaded — press Refresh first', 'bad'); return; }
    var text = JSON.stringify(state.summary, null, 2);
    var done = function () { toast('summary JSON copied to the clipboard (' + text.length + ' chars)', 'good'); };
    if (navigator.clipboard && navigator.clipboard.writeText && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
    } else fallbackCopy(text, done);
  }

  function fallbackCopy(text, done) {
    var ta = el('textarea', { style: 'position:fixed;left:-9999px;top:0' });
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    ta.remove();
    if (ok) { done(); return; }
    /* Clipboard refused (usually: not a user gesture, or an insecure origin).
     * Showing the JSON on the page is the honest fallback — the presenter can
     * still select it by hand. */
    showJsonFallback(text);
    try { console.log('[paytriq] summary JSON:\n' + text); } catch (e) { /* ignore */ }
  }

  function showJsonFallback(text) {
    var old = document.getElementById('json-fallback');
    if (old) old.remove();
    var pre = el('pre', { class: 'pre pre--scroll', text: text });
    var box = el('div', { class: 'gap-hero gap-hero--unknown', id: 'json-fallback', role: 'dialog', 'aria-label': 'summary JSON' }, [
      el('div', { class: 'gap-hero__top' }, [
        el('span', { class: 'gap-hero__label', text: 'clipboard refused — summary JSON below, select and copy by hand' }),
        el('button', { type: 'button', class: 'btn btn--xs', text: 'close',
          onclick: function () { box.remove(); } })
      ]),
      pre
    ]);
    try { pre.select && pre.select(); } catch (e) { /* ignore */ }
    var host = $('#summary-host');
    host.parentNode.insertBefore(box, host);
    box.scrollIntoView({ block: 'center' });
  }

  /* =============================== keyboard ============================== */

  function typing(e) {
    var t = e.target;
    if (!t) return false;
    var tag = (t.tagName || '').toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select' || t.isContentEditable === true;
  }

  function onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (typing(e)) return;
    var k = e.key;
    if (k >= '1' && k <= '7') {
      var s = STAGES[Number(k) - 1];
      if (s) { e.preventDefault(); runStage(s); }
      return;
    }
    var lower = k.toLowerCase();
    if (lower === 'n') { e.preventDefault(); nextStage(); }
    else if (lower === 'p') { e.preventDefault(); togglePause(); }
    else if (lower === 'g') { e.preventDefault(); refreshGates(); }
    else if (lower === 'r') { e.preventDefault(); refreshBoard(); refreshSummary(); refreshCoordination(); refreshHealth(); }
    else if (lower === 'c') { e.preventDefault(); copySummary(); }
    else if (lower === 'f') {
      e.preventDefault();
      document.body.classList.toggle('focus-trace');
      toast(document.body.classList.contains('focus-trace') ? 'trace focus on — press f to restore the rails' : 'rails restored', 'good');
    }
  }

  function nextStage() {
    for (var i = 0; i < STAGES.length; i++) {
      var st = state.stageState[STAGES[i].key];
      if (!st || st.state === 'idle') { runStage(STAGES[i]); return; }
    }
    toast('every stage has been run at least once', 'good');
  }

  function applyRunId() {
    var v = $('#run-id').value.trim();
    /* Pasted a different id (or the first one): the feed on screen belongs to
     * the previous run, so clear it before streaming the new one. */
    if (v && v !== state.runId) {
      state.runId = v;
      state.threadId = '';
      lsSet(LS.run, v);
      $('#trace-connect').disabled = !v;
      clearIssue('nrun');
      resetForRun(v);
      connectTrace({ fresh: true });
      refreshGates(); refreshSummary(); refreshBoard(); refreshCoordination();
      return;
    }
    state.runId = v;
    state.threadId = '';
    lsSet(LS.run, v);
    $('#trace-connect').disabled = !v;
    if (!v) {
      disconnectTrace();
      setIssue('nrun', 'info', 'run_id cleared',
        'Every stream and summary call is keyed by run_id, so nothing can be fetched until one is known.', null);
      return;
    }
    clearIssue('nrun');
    connectTrace();
    refreshGates(); refreshSummary(); refreshBoard(); refreshCoordination();
  }

  /* ================================= init ================================ */

  function wire() {
    on($('#api-apply'), 'click', function () {
      var b = applyBase();
      toast('API base = ' + b + ' — reconnected', 'good');
      if (state.runId) connectTrace();
    });
    on($('#api-base'), 'keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); applyBase(); } });
    on($('#health-refresh'), 'click', refreshHealth);
    on($('#banner-retry'), 'click', function () {
      clearIssue('health'); clearIssue('sse'); clearIssue('nrun'); clearIssue('gate-noid');
      bannerDismissed = false;
      refreshHealth();
      if (state.runId) { connectTrace(); refreshGates(); refreshSummary(); refreshBoard(); refreshCoordination(); }
      else toast('no run_id yet — create an event and run a stage', 'bad');
    });
    on($('#banner-dismiss'), 'click', function () { bannerDismissed = true; renderBanner(); });
    on($('#run-id'), 'change', function () { applyRunId(); });
    on($('#run-id'), 'keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); applyRunId(); } });
    on($('#trace-connect'), 'click', connectTrace);
    on($('#trace-disconnect'), 'click', function () { disconnectTrace(); toast('stream stopped', 'good'); });
    on($('#trace-pause'), 'click', togglePause);
    on($('#trace-fetch'), 'click', fetchTrace);
    on($('#trace-clear'), 'click', function () { clearFeed(); });
    on($('#trace-follow'), 'click', function () {
      state.follow = !state.follow;
      this.setAttribute('aria-pressed', state.follow ? 'true' : 'false');
      toast(state.follow ? 'following the newest event' : 'follow off — the feed will not scroll itself', 'good');
    });
    on($('#event-prefill'), 'click', prefillForm);
    on($('#event-schema-toggle'), 'click', function () {
      var d = $('#schema-disclosure');
      d.open = !d.open;
    });
    on($('#event-form'), 'submit', createEvent);
    on($('#pipe-reset'), 'click', function () {
      state.stageState = {};
      renderStages();
      toast('local stage states cleared — the backend was not touched', 'good');
    });
    on($('#gates-refresh'), 'click', refreshGates);
    on($('#reply-random'), 'click', function () {
      var p = PRESETS[Math.floor(Math.random() * PRESETS.length)];
      selectPreset(p.key);
    });
    on($('#reply-send'), 'click', function () { runStage(STAGES[3]); });
    on($('#board-refresh'), 'click', refreshBoard);
    on($('#board-tree-toggle'), 'click', function () {
      var on = this.getAttribute('aria-pressed') === 'true';
      this.setAttribute('aria-pressed', on ? 'false' : 'true');
      if (on) renderTree();
      else if (!state.tree) refreshBoard();
      else renderTree();
    });
    on($('#coord-refresh'), 'click', refreshCoordination);
    on($('#summary-refresh'), 'click', refreshSummary);
    on($('#summary-copy'), 'click', copySummary);
    on(document, 'keydown', onKey);
    on(window, 'error', function (e) {
      /* A JS error in a live demo must be on screen, not in devtools. */
      setIssue('js-error', 'bad', 'JavaScript error in the console',
        (e.message || 'unknown error') + (e.filename ? ' at ' + String(e.filename).split('/').pop() + ':' + e.lineno : '') +
        '. The panel that raised it may be stale or empty; everything else on this page is still real data from the backend.', null);
    });
    on(window, 'unhandledrejection', function (e) {
      var r = e.reason;
      setIssue('js-rejection', 'bad', 'A background request failed inside the console',
        String((r && r.message) || r || 'unknown rejection') + '. This is a fault in the console, not a result from the backend.', null);
    });
    on(window, 'resize', function () { if (state.follow) { var f = $('#trace-feed'); f.scrollTop = f.scrollHeight; } });
  }

  function init() {
    reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    state.follow = true;
    restoreConfig();
    wire();
    renderStages();
    renderGates();
    renderKindFilters();
    renderDecisions();
    renderCoordination();
    renderBoard();
    renderSummary();
    renderChain();
    setConn('idle');
    buildPresets();
    $('#trace-connect').disabled = !state.runId;

    if (state.eventId) {
      setStatus('#event-status', 'event ' + state.eventId + ' restored from this browser — press any stage to run it', 'ok');
    }
    if (state.runId) {
      $('#conn-text').textContent = 'run_id ' + state.runId + ' known — press Connect to stream it';
    }

    refreshHealth();
    loadSchema();
    if (state.runId) {
      refreshGates(); refreshSummary(); refreshBoard(); refreshCoordination();
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

  /* exposed for the console only — no behaviour depends on it */
  window.__paytriq = { state: state, request: request, refresh: scheduleRefresh };
})();