// LoL Draft Lab front end (no build step).
const ORDER = [
  ["Blue","ban",0],["Red","ban",0],["Blue","ban",1],["Red","ban",1],["Blue","ban",2],["Red","ban",2],
  ["Blue","pick",0],["Red","pick",0],["Red","pick",1],["Blue","pick",1],["Blue","pick",2],["Red","pick",2],
  ["Red","ban",3],["Blue","ban",3],["Red","ban",4],["Blue","ban",4],
  ["Red","pick",3],["Blue","pick",3],["Blue","pick",4],["Red","pick",4],
];
const ROLE_LABEL = {top:"Top", jng:"Jungle", mid:"Mid", bot:"Bot", sup:"Support"};
const $ = (id) => document.getElementById(id);

let SETUP = null, DD = {version:null, ids:{}};
let recSort = "pro";  // pro-pick prediction is the validated signal (top-5 ~51%); win-chance gain is small
let S = null;            // current draft
let roleFilter = null, busy = false, gridOpen = false, moreOpen = false;

// Phone view tabs (Board / Suggestions / Plan). Wider screens show everything, so this only matters at phone width.
function setMTab(m) {
  $("draft").dataset.mtab = m;
  document.querySelectorAll(".mtabs button").forEach(b => b.classList.toggle("on", b.dataset.m === m));
}

const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

async function loadDDragon() {
  try {
    const v = (await (await fetch("https://ddragon.leagueoflegends.com/api/versions.json")).json())[0];
    const data = (await (await fetch(`https://ddragon.leagueoflegends.com/cdn/${v}/data/en_US/champion.json`)).json()).data;
    DD.version = v;
    for (const [id, c] of Object.entries(data)) { DD.ids[norm(c.name)] = id; DD.ids[norm(id)] = id; }
  } catch (e) { console.warn("Champion icons unavailable (offline?)", e); }
}
function icon(name, cls = "") {
  const id = DD.ids[norm(name)];
  if (DD.version && id) return `<img class="${cls}" alt="${name}" title="${name}" src="https://ddragon.leagueoflegends.com/cdn/${DD.version}/img/champion/${id}.png">`;
  return `<div class="ph ${cls}" title="${name}">${name.slice(0, 2)}</div>`;
}
async function api(path, body) {
  const r = await fetch(path, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : {});
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

// ------------------------------------------------------------------ setup
async function init() {
  loadDDragon().then(() => { if (S?.resp) render(); });   // icons arrive in the background
  ["myTeam", "oppTeam"].forEach(id => { $(id).innerHTML = "<option>Loading teams…</option>"; $(id).disabled = true; });
  const setup = await api("/api/setup").catch(e => ({error: e.message}));
  ["myTeam", "oppTeam"].forEach(id => { $(id).disabled = false; });
  if (setup.error) { $("setup").innerHTML = `<h2>Not ready</h2><p>${setup.error}</p>`; return; }
  SETUP = setup;
  $("info").textContent = `${setup.info.patch ? "Patch " + setup.info.patch + " · " : ""}Data ${setup.info.data_range?.join(" → ") ?? ""} · ${setup.champions.length} champions · ${setup.teams.length} teams`;
  const opts = groupTeams(setup.teams);
  $("myTeam").innerHTML = opts; $("oppTeam").innerHTML = opts;
  const has = (n) => setup.teams.some(t => t.name === n);
  if (has("T1") && has("Gen.G")) { $("myTeam").value = "T1"; $("oppTeam").value = "Gen.G"; }
  else if (setup.teams.length > 1) $("oppTeam").selectedIndex = 1;
  $("playerList").innerHTML = setup.players.map(p => `<option value="${p.name}">${p.role ? ROLE_LABEL[p.role] : ""} · ${p.games} wtd games</option>`).join("");
  $("myTeam").onchange = () => fillRoster("myRoster", $("myTeam").value);
  $("oppTeam").onchange = () => fillRoster("oppRoster", $("oppTeam").value);
  fillRoster("myRoster", $("myTeam").value); fillRoster("oppRoster", $("oppTeam").value);
  $("champList").innerHTML = setup.champions.map(c => `<option value="${c.name}">`).join("");
  $("cxBtn").onclick = lookupCounters;
  $("cxChamp").onkeydown = (e) => { if (e.key === "Enter") lookupCounters(); };
  $("startBtn").onclick = start;
  $("undoBtn").onclick = undo;
  document.addEventListener("keydown", (e) => {
    if ($("draft").classList.contains("hidden")) return;
    const typing = e.target.closest?.("input, select, textarea, button, summary, [role=button]");
    // "/" jumps to champion search
    if (e.key === "/" && document.activeElement !== $("search")) { e.preventDefault(); setMTab("sugg"); $("search").focus(); }
    // Enter with nothing focused locks the top suggestion
    else if (e.key === "Enter" && !typing && !e.repeat && S?.resp?.recommendations?.length) { e.preventDefault(); choose(S.resp.recommendations[0].champion); }
  });
  // Enter in search locks the first available match; Escape clears it
  $("search").onkeydown = (e) => {
    if (e.key === "Enter") { const first = $("grid").querySelector(".champ:not(.off)"); if (first && $("search").value.trim()) { e.preventDefault(); choose(first.dataset.c); } }
    else if (e.key === "Escape") { $("search").value = ""; renderGrid(); $("search").blur(); }
  };
  $("gridBtn").onclick = () => { gridOpen = !gridOpen; renderGrid(); };
  document.querySelectorAll(".mtabs button").forEach(b => b.onclick = () => { setMTab(b.dataset.m); window.scrollTo({top: Math.min(scrollY, $("draft").offsetTop)}); });
  // Keep the sticky side column just under the sticky top bar, whatever its height.
  if (window.ResizeObserver) new ResizeObserver(() => document.documentElement.style.setProperty("--tbh", `${$("topbar").offsetHeight}px`)).observe($("topbar"));
  $("resetBtn").onclick = () => {
    if (SERIES && SERIES.games.length && !confirm("Abandon this series and go back to setup?")) return;
    $("draft").classList.add("hidden"); $("setup").classList.remove("hidden"); $("counters").classList.remove("hidden");
  };
  $("search").oninput = () => { setMTab("sugg"); renderGrid(); };
  $("roleFilter").innerHTML = ["all", ...setup.roles].map(r => `<button data-r="${r}" class="${r === "all" ? "on" : ""}">${r === "all" ? "All" : ROLE_LABEL[r]}</button>`).join("");
  $("roleFilter").onclick = (e) => {
    const r = e.target.dataset.r; if (!r) return;
    roleFilter = r === "all" ? null : r;
    if (roleFilter) gridOpen = true;
    [...$("roleFilter").children].forEach(b => b.classList.toggle("on", b.dataset.r === r));
    renderGrid();
  };
}
function teamTier(elo, all) {
  // Tier from the league-adjusted rating, relative to all active teams (S top 5%, A next 15%, B next 30%, C rest).
  const sorted = all.map(t => t.elo).sort((a, b) => b - a);
  const rank = sorted.indexOf(elo) / Math.max(1, sorted.length - 1);
  return rank <= 0.05 ? "S" : rank <= 0.2 ? "A" : rank <= 0.5 ? "B" : "C";
}
function groupTeams(teams) {
  const by = {};
  teams.forEach(t => (by[t.league || "Other"] ??= []).push(t));
  const MAJOR = ["LCK", "LPL", "LEC", "LCS", "LCP", "CBLOL", "LTA N", "LTA S", "LCKC", "EM", "LFL", "PRM"];
  const rank = (l) => (MAJOR.indexOf(l) + 1 || 99);
  return Object.keys(by).sort((a, b) => rank(a) - rank(b) || a.localeCompare(b)).map(l => `<optgroup label="${l}">` +
    by[l].map(t => `<option value="${esc(t.name)}">${esc(t.name)} · tier ${teamTier(t.elo, teams)}</option>`).join("") + `</optgroup>`).join("");
}
function fillRoster(el, teamName) {
  const t = SETUP.teams.find(t => t.name === teamName);
  $(el).innerHTML = SETUP.roles.map(r => `<div class="row"><span>${ROLE_LABEL[r]}</span>
    <input list="playerList" data-role="${r}" value="${t?.roster[r]?.name ?? ""}"></div>`).join("");
}
function readRoster(el) {
  const out = {};
  $(el).querySelectorAll("input").forEach(inp => {
    const p = SETUP.players.find(p => p.name === inp.value) || SETUP.players.find(p => p.name.toLowerCase() === inp.value.toLowerCase());
    out[inp.dataset.role] = p ? p.key : inp.value;
  });
  return out;
}

// ------------------------------------------------------------------ counter explorer
async function lookupCounters() {
  const champ = $("cxChamp").value.trim(); if (!champ) return;
  const role = $("cxRole").value;
  let r;
  try { r = await api(`/api/matchups?champ=${encodeURIComponent(champ)}&role=${role}`); }
  catch (e) { $("cxOut").innerHTML = `<p class="muted">${e.message}</p>`; return; }
  const table = (rows, key, title) => `<div><h3>${title}</h3>${rows.length ? `<table class="cx"><tr><th></th><th>Champion</th><th>Role</th><th>Games</th><th>WR</th><th>Lift</th><th>z</th><th>Seasons ahead</th><th>Years</th></tr>` +
    rows.map(x => `<tr class="${x.z >= 2 ? "strong" : x.z < 1 ? "weak" : ""}"><td>${icon(x[key], "mini")}</td><td>${x[key]}</td><td>${ROLE_LABEL[x.role] ?? "Any (team)"}</td><td>${x.games}</td>
      <td>${(x.win_rate * 100).toFixed(0)}%</td><td class="${x.lift >= 0 ? "pos" : "neg"}">${x.lift >= 0 ? "+" : ""}${(x.lift * 100).toFixed(1)}%</td>
      <td>${x.z.toFixed(1)}</td><td>${x.seasons_ahead}/${x.seasons}</td><td>${x.first}–${x.last}</td></tr>`).join("") + `</table>` : `<p class="muted">Not enough games.</p>`}</div>`;
  $("cxOut").innerHTML = table(r.good_into_it, "champ", `Good into ${champ}`) + table(r.it_beats, "vs", `${champ} is good into`) +
    `<p class="muted" style="grid-column:1/-1">Lift = win rate above what both champions' strength that season predicts. z = how far that is from zero in standard errors. There are thousands of matchups, so about 1 in 20 passes |z| &gt; 2 by pure chance: treat single matchups as hints, not proof.</p>`;
}

// ------------------------------------------------------------------ draft
// A series holds one or more games; S is the game currently being drafted.
let SERIES = null;

function start() {
  const my = $("myTeam").value, opp = $("oppTeam").value;
  if (my === opp) { alert("Pick two different teams"); return; }
  const bestOf = parseInt($("format").value, 10);
  SERIES = {
    bestOf, need: Math.floor(bestOf / 2) + 1, fearless: $("fearless").checked && bestOf > 1,
    my, opp, myRoster: readRoster("myRoster"), oppRoster: readRoster("oppRoster"),
    preLocked: $("locked").value.split(",").map(s => s.trim()).filter(Boolean),
    temperature: parseFloat($("temp").value), aiMode: $("aiMode").value, games: [], wins: {me: 0, opp: 0},
  };
  $("setup").classList.add("hidden"); $("counters").classList.add("hidden"); $("draft").classList.remove("hidden");
  startGame($("mySide").value);
}

function seriesLocked() {
  if (!SERIES.fearless) return [...SERIES.preLocked];
  const used = SERIES.games.flatMap(g => g.picks);
  return [...new Set([...SERIES.preLocked, ...used])];
}

function startGame(mySide) {
  const {my, opp, myRoster, oppRoster} = SERIES;
  S = {
    mySide, actions: [], history: [], gameNo: SERIES.games.length + 1,
    blue_team: mySide === "Blue" ? my : opp, red_team: mySide === "Blue" ? opp : my,
    blue_roster: mySide === "Blue" ? myRoster : oppRoster, red_roster: mySide === "Blue" ? oppRoster : myRoster,
    locked: seriesLocked(), temperature: SERIES.temperature, resp: null,
  };
  setDone(false);
  gridOpen = false; moreOpen = false; $("search").value = "";
  setMTab("sugg");
  insightsReset();
  $("blueName").textContent = S.blue_team + (mySide === "Blue" ? " (you)" : "");
  $("redName").textContent = S.red_team + (mySide === "Red" ? " (you)" : "");
  $("blueHead").textContent = S.blue_team + (mySide === "Blue" ? " · you" : "");
  $("redHead").textContent = S.red_team + (mySide === "Red" ? " · you" : "");
  renderSeriesBar();
  window.scrollTo({top: 0});
  refresh();
}

// Max further games after the current one: how much a fearless lock can still cost you.
const gamesRemainingAfterThis = () => Math.max(0, SERIES.bestOf - (SERIES.games.length + 1));

function renderSeriesBar() {
  const s = SERIES, locks = S.locked;
  // Compact series state lives in the sticky top bar; the (possibly long) fearless lock list sits just below it.
  $("seriesBar").innerHTML = s.bestOf === 1
    ? `<span class="tb-label">Single game</span><div class="score"><span class="me">${esc(s.my)}</span> <span class="muted">vs</span> ${esc(s.opp)}</div>`
    : `<span class="tb-label">Bo${s.bestOf}${s.fearless ? " · Fearless" : ""} · Game ${S.gameNo}</span>
       <div class="score"><span class="me">${esc(s.my)}</span> <b>${s.wins.me}–${s.wins.opp}</b> ${esc(s.opp)}</div>`;
  $("lockRow").classList.toggle("hidden", !locks.length);
  $("lockRow").innerHTML = locks.length ? `<span class="muted">${s.fearless ? "Locked for this series" : "Locked"} (${locks.length}):</span>${locks.map(c => icon(c)).join("")}` : "";
}
function setDone(done) {
  $("draft").classList.toggle("done", done);
  $("pickUI").classList.toggle("hidden", done);
  $("review").classList.toggle("hidden", !done);
  const b = document.querySelector('.mtabs [data-m="sugg"]'); if (b) b.textContent = done ? "Review" : "Suggestions";
}

const payload = () => ({blue_team: S.blue_team, red_team: S.red_team, blue_roster: S.blue_roster,
  red_roster: S.red_roster, actions: S.actions, locked: S.locked, temperature: S.temperature,
  games_remaining: SERIES && SERIES.fearless ? gamesRemainingAfterThis() : 0, sort: recSort,
  ai_mode: SERIES?.aiMode || "imitate"});

async function refresh() {
  busy = true;
  try {
    S.resp = await api("/api/state", payload());
  } catch (e) {
    busy = false;
    if (S.pending) { S.actions.pop(); S.history.pop(); S.pending = false; }
    flash(`Couldn't update the draft: ${e.message}`);
    return;
  }
  S.pending = false;
  busy = false;
  render();
  renderEarly();
  renderBrief();
  renderCard();
  const nxt = S.resp.next;
  if (!nxt) return showReview();
  if (nxt.side !== S.mySide) {
    busy = true;
    $("turn").querySelector(".tsub")?.insertAdjacentHTML("beforeend", ` <span class="thinking">· thinking…</span>`);
    await new Promise(r => setTimeout(r, 650));
    let mv;
    try { mv = await api("/api/ai_move", payload()); }
    catch (e) { busy = false; flash(`The opponent couldn't move: ${e.message}`); return; }
    S.history.push({mine: false, champ: mv.champion});
    S.actions.push(mv.champion);
    busy = false;
    refresh();
  }
}
function flash(msg) {
  let el = document.getElementById("flash");
  if (!el) { el = document.createElement("div"); el.id = "flash"; el.className = "flash"; el.setAttribute("role", "status"); document.body.append(el); }
  el.textContent = msg; el.classList.add("show");
  clearTimeout(flash.t); flash.t = setTimeout(() => el.classList.remove("show"), 3500);
}
function choose(champ) {
  if (busy || !S.resp?.next || S.resp.next.side !== S.mySide) return;
  // Already picked, banned, or locked by fearless: say so instead of sending an invalid draft.
  if (S.actions.includes(champ) || S.locked.includes(champ)) {
    flash(S.locked.includes(champ) ? `${champ} is locked for this series (fearless).` : `${champ} is already picked or banned.`);
    return;
  }
  const recs = S.resp.recommendations;
  const rank = recs.findIndex(r => r.champion === champ);
  S.history.push({mine: true, champ, kind: S.resp.next.kind, step: S.actions.length,
    top: recs[0]?.champion, rank: rank >= 0 ? rank + 1 : null, recs: recs.slice(0, 3).map(r => r.champion),
    winBefore: S.resp.win_prob_blue});
  S.actions.push(champ);
  S.pending = true;
  refresh();
}
function undo() {
  if (busy || !S) return;
  // undo back to (and including) the user's last action
  while (S.history.length) {
    const h = S.history.pop(); S.actions.pop();
    if (h.mine) break;
  }
  setDone(false);
  setMTab("sugg");
  refresh();
}

// ------------------------------------------------------------------ insights: lanes map, game plan, win condition
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const goldFmt = (g) => `${g >= 0 ? "+" : "−"}${Math.abs(Math.round(g)).toLocaleString()}`;
const iconUrl = (name) => { const id = DD.ids[norm(name)]; return DD.version && id ? `https://ddragon.leagueoflegends.com/cdn/${DD.version}/img/champion/${id}.png` : null; };
let PC = null, pcSide = null, pcSeq = 0;   // plan card (after a full draft)
let EF = null, BR = null, insightTab = "lanes", planGold = 0, planTouched = false, laneSel = null, earlySeq = 0, briefSeq = 0;

function insightsReset() {
  EF = null; BR = null; PC = null; pcSide = null; laneSel = null; planTouched = false;
  if (insightTab === "card") insightTab = "lanes";
  renderStrip(); renderInsights();
}
async function renderEarly() {
  const seq = ++earlySeq;
  if (!S.actions.some((c, i) => c && ORDER[i][1] === "pick")) { EF = null; renderStrip(); renderInsights(); return; }
  try { const f = await api("/api/early", payload()); if (seq !== earlySeq) return; EF = f; }
  catch (e) { EF = {error: e.message}; }
  renderStrip(); renderInsights();
}
async function renderBrief() {
  const seq = ++briefSeq;
  const mine = S.actions.filter((c, i) => c && ORDER[i][1] === "pick" && ORDER[i][0] === S.mySide).length;
  const theirs = S.actions.filter((c, i) => c && ORDER[i][1] === "pick" && ORDER[i][0] !== S.mySide).length;
  if (mine < 3 || theirs < 3) { BR = null; renderStrip(); renderInsights(); return; }
  try {
    const b = await api("/api/brief", {...payload(), brief_side: S.mySide}); if (seq !== briefSeq) return;
    const first = !BR; BR = b;
    if ((first || !planTouched) && b.forecast) planGold = Math.max(-6000, Math.min(6000, Math.round(b.forecast.gold / 250) * 250));
  } catch (e) { BR = {error: e.message}; }
  renderStrip(); renderInsights();
}

// ---------- summary strip (under the win bar)
function renderStrip() {
  const el = $("istrip"); if (!el) return;
  const chips = [];
  if (EF && !EF.error) {
    const g = EF.team_gold15 * (S.mySide === "Blue" ? 1 : -1);
    chips.push(`<button class="chip" data-tab="lanes"><span>Gold at 15</span><b class="${g >= 0 ? "pos" : "neg"}">${goldFmt(g)}</b></button>`);
  }
  if (BR && !BR.error) {
    chips.push(`<button class="chip" data-tab="plan"><span>Break-even</span><b>${goldFmt(BR.break_even_gold)}</b></button>`);
    const c = (BR.win_condition?.carries || []).map(x => esc(x.champion)).join(" · ");
    if (c) chips.push(`<button class="chip" data-tab="wincon"><span>Win condition</span><b>${c}</b></button>`);
    chips.push(`<button class="chip" data-tab="plan"><span>Comps</span><b>${esc(BR.own_comp.timing)} vs ${esc(BR.enemy_comp.timing)}</b></button>`);
  }
  if (PC && !PC.error) chips.push(`<button class="chip chip-card" data-tab="card"><span>Plan card</span><b>ready</b></button>`);
  el.innerHTML = chips.length ? chips.join("") : `<span class="muted">Early-game insights appear as champions are picked.</span>`;
  el.querySelectorAll("[data-tab]").forEach(b => b.onclick = () => { insightTab = b.dataset.tab; setMTab("plan"); renderInsights(); $("insights").scrollIntoView({behavior: "smooth", block: "nearest"}); });
}

// ---------- tabs
function renderInsights() {
  const box = $("insights"); if (!box) return;
  box.querySelectorAll(".itab").forEach(b => { b.classList.toggle("on", b.dataset.tab === insightTab); b.onclick = () => { insightTab = b.dataset.tab; renderInsights(); }; });
  const body = $("ibody");
  if (insightTab === "lanes") body.innerHTML = lanesView();
  else if (insightTab === "plan") body.innerHTML = planView();
  else if (insightTab === "card") body.innerHTML = cardView();
  else body.innerHTML = winconView();
  if (insightTab === "lanes") wireLanes();
  if (insightTab === "card") wireCard();
  if (insightTab === "plan") wirePlan();
}

// ---------- Lanes: mini-map
const LANES = {  // SVG paths in a 300x300 map; blue base bottom-left, red base top-right
  top: {d: "M40,250 L40,40 L250,40", mid: [40, 40], blue: [40, 120], red: [120, 40]},
  mid: {d: "M55,245 L245,55", mid: [150, 150], blue: [125, 175], red: [175, 125]},
  bot: {d: "M50,260 L260,260 L260,50", mid: [260, 260], blue: [165, 260], red: [260, 165]},
  jng: {blue: [78, 165], red: [222, 135]},
  sup: {blue: [212, 238], red: [238, 212]},
};
function lanesView() {
  if (!EF) return `<p class="muted">Pick a few champions to see the lane forecast.</p>`;
  if (EF.error) return `<p class="muted">${esc(EF.error)}</p>`;
  const sgn = S.mySide === "Blue" ? 1 : -1, max = Math.max(1500, ...EF.lanes.map(l => Math.abs(l.gold15)));
  const by = Object.fromEntries(EF.lanes.map(l => [l.role, l]));
  const col = (g) => { const a = Math.min(1, Math.abs(g) / max); return g >= 0 ? `rgba(34,197,94,${0.25 + 0.75 * a})` : `rgba(248,113,113,${0.25 + 0.75 * a})`; };
  const meSide = S.mySide, themSide = S.mySide === "Blue" ? "Red" : "Blue";
  const portrait = (name, x, y, side, role) => {
    if (!name) return `<circle cx="${x}" cy="${y}" r="13" class="mm-empty"/>`;
    const u = iconUrl(name), cid = `c${side}${role}`;
    return `<g class="mm-champ" data-role="${role}"><clipPath id="${cid}"><circle cx="${x}" cy="${y}" r="13"/></clipPath>
      ${u ? `<image href="${u}" x="${x - 13}" y="${y - 13}" width="26" height="26" clip-path="url(#${cid})"/>` : `<circle cx="${x}" cy="${y}" r="13" fill="#334"/>`}
      <circle cx="${x}" cy="${y}" r="13.5" fill="none" stroke="${side === "Blue" ? "#3b82f6" : "#ef4444"}" stroke-width="2.5"/><title>${esc(name)}</title></g>`;
  };
  const laneLabel = (role, x, y) => { const l = by[role]; if (!l) return ""; const g = l.gold15 * sgn;
    return `<g class="mm-label" data-role="${role}"><rect x="${x - 30}" y="${y - 11}" width="60" height="22" rx="11"/><text x="${x}" y="${y + 4}" text-anchor="middle" class="${g >= 0 ? "t-pos" : "t-neg"}">${goldFmt(g)}</text></g>`; };
  const obj = (k) => (EF.objectives || []).find(o => o.key === k);
  const pitText = (o, x, y, label) => { if (!o) return ""; const p = S.mySide === "Blue" ? o.blue_prob : 1 - o.blue_prob;
    return `<g class="mm-pit"><circle cx="${x}" cy="${y}" r="16"/><text x="${x}" y="${y - 22}" text-anchor="middle" class="mm-small">${label}</text>
      <text x="${x}" y="${y + 4}" text-anchor="middle" class="${p >= 0.5 ? "t-pos" : "t-neg"}">${Math.round(p * 100)}%</text></g>`; };
  const lanePaths = ["top", "mid", "bot"].map(r => by[r] ? `<path d="${LANES[r].d}" class="mm-lane ${laneSel === r ? "sel" : ""}" data-role="${r}" stroke="${col(by[r].gold15 * sgn)}"/>` : "").join("");
  const champs = ["top", "mid", "bot", "jng", "sup"].map(r => { const l = by[r]; if (!l) return "";
    // Your champions always sit on the bottom-left side of the map, with your side's ring colour.
    const mineC = meSide === "Blue" ? l.blue : l.red, theirC = meSide === "Blue" ? l.red : l.blue;
    return portrait(mineC, ...LANES[r].blue, meSide, r) + portrait(theirC, ...LANES[r].red, themSide, r); }).join("");
  const mine = S.mySide === "Blue" ? S.blue_team : S.red_team;
  const svg = `<svg viewBox="0 0 300 300" class="minimap" role="img" aria-label="Map of predicted lane leads at 15 minutes">
    <rect x="5" y="5" width="290" height="290" rx="18" class="mm-bg"/>
    <path d="M30,30 L270,270" class="mm-river"/>
    <circle cx="35" cy="265" r="22" class="mm-base ${meSide.toLowerCase()}"/><circle cx="265" cy="35" r="22" class="mm-base ${themSide.toLowerCase()}"/>
    <text x="35" y="300" text-anchor="middle" class="mm-small">you</text>
    ${lanePaths}${pitText(obj("grubs"), 95, 95, "Grubs")}${pitText(obj("first_dragon"), 205, 205, "Drake")}
    ${laneLabel("top", 40, 40)}${laneLabel("mid", 150, 150)}${laneLabel("bot", 260, 260)}
    ${champs}</svg>`;
  const sel = laneSel ? by[laneSel] : null;
  const detail = sel ? `<h4>${esc(sel.label)}: ${goldFmt(sel.gold15 * sgn)} gold at 15 for ${esc(mine)}</h4>
      <ul>${sel.reasons.map(r => `<li>${esc(r)}</li>`).join("") || "<li>No strong signal either way.</li>"}</ul>
      ${sel.complete ? "" : `<p class="muted">Not both champions picked yet: players and teams only.</p>`}`
    : `<h4>Click a lane or champion</h4><p class="muted">Your base is bottom-left. Lanes are green where ${esc(mine)} is forecast ahead at 15, red where behind. A typical lane forecast misses by about ±${(EF.typical_error?.lane || 700).toLocaleString()} gold, so treat small numbers as even.</p>`;
  const others = ["jng", "sup"].filter(r => by[r]).map(r => { const g = by[r].gold15 * sgn;
    return `<button class="lane-pill ${laneSel === r ? "on" : ""}" data-role="${r}">${esc(by[r].label)} <b class="${g >= 0 ? "pos" : "neg"}">${goldFmt(g)}</b></button>`; }).join("");
  const objs = (EF.objectives || []).map(o => { const p = S.mySide === "Blue" ? o.blue_prob : 1 - o.blue_prob;
    return `<div class="objrow"><span>${esc(o.label)}</span><div class="objbar"><div style="width:${(p * 100).toFixed(1)}%"></div></div><b>${Math.round(p * 100)}%</b></div>`; }).join("");
  const tg = Math.round(EF.team_gold15 * sgn / 100) * 100;
  const low = (EF.low_data || []).length ? `<p class="lowdata">Low data: Oracle's Elixir has 15-minute gold for only about 15% of LPL games, so ${esc(EF.low_data.join(" and "))} forecasts lean on other leagues.</p>` : "";
  return `<div class="lanes-grid">${svg}<div class="lane-side">${low}
      <div class="bigstat"><span class="muted">Team gold at 15 (sum of lanes)</span><b class="${tg >= 0 ? "pos" : "neg"}">${goldFmt(tg)}</b><span class="muted">± ~${(EF.typical_error?.team || 2000).toLocaleString()}</span></div>
      <div class="lane-pills">${others}</div>
      <div class="lane-detail">${detail}</div>
      <div class="objs2"><span class="muted">Your chance of each early objective</span>${objs}</div></div></div>`;
}
function wireLanes() {
  $("ibody").querySelectorAll("[data-role]").forEach(el => el.addEventListener("click", () => { laneSel = el.dataset.role; renderInsights(); }));
}

// ---------- Game plan: interactive gold slider and win-chance curve
const ZONES = [{key: "bad", max: -1250}, {key: "even", max: 1000}, {key: "good", max: 3000}, {key: "best", max: Infinity}];
function zoneFor(g) { return ZONES.find(z => g <= z.max).key; }
function pAt(g) {
  const c = BR.curve; if (!c?.length) return 0.5;
  const i = Math.max(0, Math.min(c.length - 2, Math.floor((g - c[0].gold) / 250)));
  const a = c[i], b = c[i + 1], t = (g - a.gold) / (b.gold - a.gold);
  return a.p + (b.p - a.p) * Math.max(0, Math.min(1, t));
}
function planView() {
  if (!BR) return `<p class="muted">The game plan appears once both teams have picked three champions.</p>`;
  if (BR.error) return `<p class="muted">${esc(BR.error)}</p>`;
  const comp = (c) => `<b>${esc(c.timing)}</b>${c.main_style ? ` · ${esc(c.main_style)}` : ""}`;
  const st = BR.style || {favoured: [], unfavoured: []};
  const styleTxt = st.score > 0.05 ? "favours you" : st.score < -0.05 ? "favours them" : "is about neutral";
  if (Math.abs(st.score) <= 0.05) st.favoured = st.unfavoured = [];   // don't show chips that contradict "neutral"
  const scenBtns = BR.scenarios.map(s => `<button class="scen-btn scen-${s.key}" data-gold="${s.gold}">${esc(s.label)}<small>${goldFmt(s.gold)}</small></button>`).join("");
  const partial = BR.picks.own < 5 || BR.picks.enemy < 5 ? ` <span class="muted">(${BR.picks.own}/5 vs ${BR.picks.enemy}/5 picks so far)</span>` : "";
  return `<div class="plan-top">
      <div><span class="muted">${esc(BR.team)} (you)</span><div>${comp(BR.own_comp)}</div></div>
      <div class="vs">vs</div>
      <div><span class="muted">${esc(BR.enemy)}</span><div>${comp(BR.enemy_comp)}</div></div>
      <div class="plan-style"><span class="muted">Style matchup ${styleTxt}</span><div>${[...st.favoured.map(x => `<span class="tag pos">${esc(x)}</span>`), ...st.unfavoured.map(x => `<span class="tag neg">${esc(x)}</span>`)].join(" ")}</div></div>
    </div>
    <div class="plan-grid">
      <div class="chartbox"><svg id="wpchart" viewBox="0 0 520 240" role="img" aria-label="Win chance by gold lead at 15"></svg>
        <input id="goldSlider" type="range" min="-6000" max="6000" step="250" value="${planGold}" aria-label="Gold lead at 15">
        <div class="scen-btns">${scenBtns}</div></div>
      <div class="readout" id="readout"></div>
    </div>
    <details class="plan-more"><summary>Plan to 15 minutes${partial}</summary><ul>${BR.plan.map(l => `<li>${esc(l)}</li>`).join("")}</ul>
      <p class="muted">At the same gold at 15, teams that took first dragon went on to win about ${Math.round(BR.objective_value.first_dragon * 100)} points more often, first tower ${Math.round(BR.objective_value.first_tower * 100)}, void grubs ${Math.round(BR.objective_value.grubs * 100)}. This is an association (objectives also reflect how the game is going), not a gain you bank by taking them.</p></details>`;
}
function drawChart() {
  const svg = $("wpchart"); if (!svg || !BR?.curve) return;
  const W = 520, H = 240, ml = 40, mr = 34, mt = 14, mb = 28;
  const x = (g) => ml + (g + 6000) / 12000 * (W - ml - mr), y = (p) => mt + (1 - p) * (H - mt - mb);
  const pts = BR.curve.map(c => `${x(c.gold).toFixed(1)},${y(c.p).toFixed(1)}`).join(" ");
  const grid = [0.25, 0.5, 0.75].map(p => `<line x1="${ml}" x2="${W - mr}" y1="${y(p)}" y2="${y(p)}" class="g"/><text x="${ml - 6}" y="${y(p) + 4}" text-anchor="end" class="ax">${p * 100}%</text>`).join("");
  const xt = [-6000, -3000, 0, 3000, 6000].map(g => `<text x="${x(g)}" y="${H - 8}" text-anchor="middle" class="ax">${g === 0 ? "even" : goldFmt(g)}</text>`).join("");
  const zones = [[-6000, -1250, "bad"], [-1250, 1000, "even"], [1000, 3000, "good"], [3000, 6000, "best"]]
    .map(([a, b, k]) => `<rect x="${x(a)}" y="${mt}" width="${x(b) - x(a)}" height="${H - mt - mb}" class="z z-${k}"/>`).join("");
  const be = BR.break_even_gold, fc = BR.forecast;
  const beLine = be > -6000 && be < 6000 ? `<line x1="${x(be)}" x2="${x(be)}" y1="${mt}" y2="${H - mb}" class="be"/><text x="${x(be) + 4}" y="${mt + 12}" class="ax">break-even ${goldFmt(be)}</text>` : "";
  const fcMark = fc ? `<g class="fc"><path d="M${x(fc.gold)},${y(pAt(fc.gold)) - 9} l7,9 l-7,9 l-7,-9 z"/><text x="${x(fc.gold)}" y="${y(pAt(fc.gold)) + 24}" text-anchor="middle" class="ax">forecast</text></g>` : "";
  const g = planGold, p = pAt(g);
  svg.innerHTML = `${zones}${grid}${xt}<line x1="${x(0)}" x2="${x(0)}" y1="${mt}" y2="${H - mb}" class="g"/>${beLine}
    <polyline points="${pts}" class="curve"/>${fcMark}
    <line x1="${x(g)}" x2="${x(g)}" y1="${y(p)}" y2="${H - mb}" class="cur"/><circle cx="${x(g)}" cy="${y(p)}" r="6" class="dot"/>`;
}
function updateReadout() {
  const el = $("readout"); if (!el) return;
  const g = planGold, p = pAt(g), z = zoneFor(g);
  const s = BR.scenarios.find(x => x.key === z);
  const advice = s ? s.advice.replace(/^\d+%\.\s*/, "") : "";
  el.className = `readout r-${z}`;
  el.innerHTML = `<span class="muted">If you are</span><div class="ro-gold">${goldFmt(g)} <small>gold at 15</small></div>
    <div class="ro-p">${Math.round(p * 100)}%<small> win chance</small></div>
    <div class="ro-zone">${esc(s?.label || "")}</div><p>${esc(advice)}</p>`;
}
function wirePlan() {
  const sl = $("goldSlider");
  sl.oninput = () => { planGold = +sl.value; planTouched = true; drawChart(); updateReadout(); };
  $("ibody").querySelectorAll(".scen-btn").forEach(b => b.onclick = () => { planGold = +b.dataset.gold; planTouched = true; sl.value = planGold; drawChart(); updateReadout(); });
  const svg = $("wpchart");
  const move = (ev) => { const r = svg.getBoundingClientRect(); const fx = (ev.clientX - r.left) / r.width * 520;
    planGold = Math.max(-6000, Math.min(6000, Math.round(((fx - 40) / (520 - 52) * 12000 - 6000) / 250) * 250)); sl.value = planGold; drawChart(); updateReadout(); };
  svg.onpointerdown = (ev) => { move(ev); svg.setPointerCapture(ev.pointerId); svg.onpointermove = move; };
  svg.onpointerup = () => { svg.onpointermove = null; };
  drawChart(); updateReadout();
}

// ---------- Plan card (early-game rules, after a full draft)
async function renderCard() {
  const seq = ++pcSeq;
  const picks = (side) => S.actions.filter((c, i) => c && ORDER[i][1] === "pick" && ORDER[i][0] === side).length;
  if (picks("Blue") < 5 || picks("Red") < 5) { PC = null; renderStrip(); if (insightTab === "card") renderInsights(); return; }
  try {
    const r = await api("/api/plan_card", payload()); if (seq !== pcSeq) return;
    const first = !PC; PC = r;
    if (first) insightTab = "card";
  } catch (e) { PC = {error: e.message}; }
  renderStrip(); renderInsights();
}
function evTag(ev) {
  const cls = {"Supported": "pos", "Contradicted": "neg", "Unclear": "unc"}[ev] || "none";
  const tip = {"Supported": "Past games back this rule (95% interval excludes zero in its direction)",
    "Contradicted": "Past games point the other way (95% interval excludes zero against the rule)",
    "Unclear": "Past games can't tell either way (95% interval includes zero)"}[ev] || "Evidence from past games";
  return `<span class="ev ev-${cls}" title="${tip}">${esc(ev)}</span>`;
}
const ruleTag = (id) => id ? `<span class="rid">${esc(id)}</span>` : "";
function valuesList(vals) {
  return (vals || []).map(v => `<li>${esc(v.words)}: <b>${esc(v.value)}</b>${v.how ? ` <span class="muted">(${esc(v.how)})</span>` : ""}</li>`).join("");
}
function cardLine(l) {
  const off = l.status === "overridden" || l.status === "unless";
  const head = off ? `<s>${esc(l.then)}</s>` : esc(l.then);
  const note = l.status === "overridden" ? `<div class="pc-note">Replaced: ${esc(l.override_reason)}</div>`
    : l.status === "unless" ? `<div class="pc-note">Not applied, because ${esc(l.unless)}.</div>`
    : l.unless_missing ? `<div class="pc-note">Unless ${esc(l.unless)} (can't check yet: ${esc(l.unless_missing.join(", "))} not labelled).</div>` : "";
  return `<li class="pc-line ${off ? "off" : ""}">
    <div class="pc-head">${ruleTag(l.rule)}<span class="pc-then">${head}</span>${evTag(l.evidence)}</div>${note}
    <details class="pc-why"><summary>Why this line</summary>
      <p>${esc(l.why)}</p>
      <p class="muted">Sheet: “${esc(l.if_text)}”</p>
      <p>Checked as: ${esc(l.condition)}</p><ul>${valuesList(l.values)}</ul>
      ${l.live_check ? `<p>Confirm in game: ${esc(l.live_check)}</p>` : ""}
      ${l.approximation ? `<p class="muted">Approximation: ${esc(l.approximation)}</p>` : ""}
      <p class="muted">Sources: ${esc(l.sources.join(", "))}</p>
      ${l.evidence_detail ? `<p class="muted">Evidence: ${esc(l.evidence_detail)}</p>` : ""}
    </details></li>`;
}
function cardView() {
  if (!PC) return `<p class="muted">The plan card appears when both teams have picked all five champions.</p>`;
  if (PC.error) return `<p class="muted">${esc(PC.error)}</p>`;
  const side = pcSide || S.mySide, c = PC.cards[side];
  const toggle = `<div class="pc-toggle" role="group" aria-label="Whose card">
      ${["Blue", "Red"].map(s => `<button data-side="${s}" class="${s === side ? "on" : ""}">${esc(PC.cards[s].team)}${s === S.mySide ? " (you)" : ""}</button>`).join("")}</div>`;
  const g = c.gold15 || {};
  const fo = c.first_objective;
  const top = `<div class="pc-top">
      <div class="pc-box"><span class="muted">Level 1</span><div>${ruleTag(c.level1.rule)} ${esc(c.level1.text)}</div></div>
      <div class="pc-box"><span class="muted">Jungle start</span><div>${ruleTag(c.jungle.rule)} ${esc(c.jungle.path)}</div></div>
      <div class="pc-box"><span class="muted">First objective</span><div>${ruleTag(fo.rule)} <b>${esc(fo.choice)}</b>: ${esc(fo.text)}</div>
        <div class="pc-tags">${fo.forecast.map(o => `<span class="tag">${esc(o.label)} ${Math.round(o.prob * 100)}%</span>`).join(" ")}</div>
        ${fo.note ? `<div class="pc-note">${esc(fo.note)}</div>` : ""}</div>
      <div class="pc-box"><span class="muted">Gold at 15</span><div>Break-even <b>${g.break_even != null ? goldFmt(g.break_even) : "–"}</b>
        · forecast <b class="${(g.forecast ?? 0) >= 0 ? "pos" : "neg"}">${g.forecast != null ? goldFmt(g.forecast) : "–"}</b></div>
        ${(c.win_condition?.carries || []).length ? `<div class="muted">Win condition: ${c.win_condition.carries.map(x => esc(x.champion)).join(" · ")}</div>` : ""}</div>
    </div>`;
  const lanes = `<div class="pc-lanes">${c.lanes.filter(l => ["top", "mid", "bot"].includes(l.role)).map(l => `
      <div class="pc-lane prio-${esc((l.prio || "unknown").toLowerCase())}"><span class="muted">${esc(l.label)} priority</span>
        <b>${esc(l.prio || "?")}</b><span>${esc(l.ours || "")} vs ${esc(l.theirs || "")}</span>
        <span class="muted">${esc(l.how || "")}</span></div>`).join("")}</div>`;
  const secs = c.sections.map(s => {
    if (!s.lines.length && !s.needs_labels.length) return "";
    const need = s.needs_labels.length ? `<details class="pc-need"><summary>${s.needs_labels.length} rule${s.needs_labels.length > 1 ? "s" : ""} need labels</summary><ul>
        ${s.needs_labels.map(l => `<li>${ruleTag(l.rule)} ${esc(l.then)} <span class="muted">— needs: ${esc(l.missing.join(", "))}</span></li>`).join("")}</ul></details>` : "";
    return `<section class="pc-sec"><h4>${esc(s.title)}</h4>${s.lines.length ? `<ul class="pc-lines">${s.lines.map(cardLine).join("")}</ul>` : `<p class="muted">No rule applies.</p>`}${need}</section>`;
  }).join("");
  const conflicts = c.conflicts.filter(x => x.active).map(x => `<li>${esc(x.text)} ${esc(x.note)}</li>`).join("");
  const watch = `<details class="pc-extra"><summary>Watch for in game (${c.watch_for.length})</summary><ul>
      ${c.watch_for.map(w => `<li>${ruleTag(w.rule)} <b>If</b> ${esc(w.when)} → ${esc(w.then)} ${evTag(w.evidence)}</li>`).join("")}</ul></details>`;
  const scout = `<details class="pc-extra"><summary>Scouting before the game (${c.scouting.length})</summary><ul>
      ${c.scouting.map(w => `<li>${ruleTag(w.rule)} ${esc(w.when)} → ${esc(w.then)}</li>`).join("")}</ul></details>`;
  const n = c.summary.missing_tags.length;
  const missing = n ? `<p class="pc-missing">${c.summary.needs_labels} rule checks are waiting on tags. Fill these in the Champion Tags sheet: ${esc(c.summary.missing_tags.slice(0, 8).join("; "))}${n > 8 ? "…" : ""}</p>` : "";
  return `<div class="pc">${toggle}
    <p class="muted">Early-game plan for ${esc(c.team)} vs ${esc(c.enemy)}, from the coaching rules sheet. Each line shows its rule ID; open “Why this line” for the condition, the values that triggered it and the sources.</p>
    ${top}${lanes}${conflicts ? `<ul class="pc-conf">${conflicts}</ul>` : ""}${secs}${watch}${scout}${missing}</div>`;
}
function wireCard() {
  $("ibody").querySelectorAll(".pc-toggle button").forEach(b => b.onclick = () => { pcSide = b.dataset.side; renderInsights(); });
}

// ---------- Win condition
function winconView() {
  if (!BR) return `<p class="muted">The win condition appears once both teams have picked three champions.</p>`;
  if (BR.error) return `<p class="muted">${esc(BR.error)}</p>`;
  const w = BR.win_condition || {};
  const cards = (w.carries || []).map(c => { const u = iconUrl(c.champion);
    return `<div class="carry">${u ? `<img src="${u}" alt="">` : ""}<div><b>${esc(c.champion)}</b> <span class="muted">${esc(c.role_label)} · ${Math.round(c.dmg_share * 100)}% of damage · ${c.games} games</span>
      <div class="ab"><span>Lane ahead</span><div class="abbar"><div class="pos-bg" style="width:${(c.wr_ahead * 100).toFixed(0)}%"></div></div><b>${Math.round(c.wr_ahead * 100)}%</b></div>
      <div class="ab"><span>Lane behind</span><div class="abbar"><div class="neg-bg" style="width:${(c.wr_behind * 100).toFixed(0)}%"></div></div><b>${Math.round(c.wr_behind * 100)}%</b></div></div></div>`; }).join("");
  const verdict = {end_early: ["Wants to end early", "neg"], end_late: ["Can play long", "pos"], flexible: ["No clear preference", ""]}[w.timing || "flexible"];
  return `<div class="wc-grid">
      <div><h4>Who needs to be doing well</h4>${cards || `<p class="muted">Not enough games for these champions.</p>`}
        <p class="muted">Team win rate when this champion's lane is 500+ gold ahead vs 500+ behind at 15.</p></div>
      <div><h4>Game length</h4><div class="verdict ${verdict[1]}">${verdict[0]}</div><p>${esc(w.timing_text || "")}</p>
        <p class="muted">Compared with comps that had the same gold at 15, so an early lead alone doesn't explain it. Based on the ${w.n_similar || 0} most similar comps this season.</p></div></div>`;
}

function render() {
  const r = S.resp, wp = r.win_prob_blue;
  $("blueBar").style.width = `${(wp * 100).toFixed(1)}%`;
  $("winText").textContent = `${Math.round(wp * 100)}% · ${Math.round((1 - wp) * 100)}%`;
  const lv = r.win_prob_blue_level;
  $("levelText").textContent = lv == null ? "" : `Draft alone: ${Math.round(lv * 100)}% · ${Math.round((1 - lv) * 100)}%`;
  const nxt = r.next, mine = nxt && nxt.side === S.mySide;
  const step = S.actions.length + 1;
  $("turn").className = `turn ${nxt ? nxt.side : "done"} ${mine ? "mine" : ""}`;
  $("turn").innerHTML = nxt
    ? `<div class="tmain">${mine ? "Your" : "Opponent"} ${nxt.kind}</div>
       <div class="tsub"><span class="sidetag ${nxt.side}">${nxt.side}</span> ${nxt.kind} ${nxt.index + 1} · step ${step}/20${mine ? ` <span class="khint">Enter = top pick</span>` : ""}</div>`
    : `<div class="tmain">Draft complete</div><div class="tsub">Review below${SERIES.bestOf > 1 ? " · record the result" : ""}</div>`;
  for (const side of ["Blue", "Red"]) renderSide(side, nxt);
  renderMStrip();
  renderRecs(); renderGrid();
}
// Phone-only strip in the sticky bar: both teams' picks so far, so the enemy's picks are always on screen.
function renderMStrip() {
  const row = (side) => ORDER.map(([s, k], n) => (s === side && k === "pick" ? {c: S.actions[n]} : null)).filter(Boolean)
    .map(({c}) => c ? icon(c) : `<span class="empty"></span>`).join("");
  $("mstrip").innerHTML = `<div class="mrow Blue">${row("Blue")}</div><div class="mrow Red">${row("Red")}</div>`;
}
function renderSide(side, nxt) {
  const bans = [], picks = [];
  ORDER.forEach(([s, k, i], n) => { if (s === side) (k === "ban" ? bans : picks)[i] = {champ: S.actions[n], n}; });
  $(`${side}-bans`).innerHTML = bans.map(b => `<div class="ban ${nxt && b.n === S.actions.length ? "active" : ""}">${b.champ ? icon(b.champ) : ""}</div>`).join("");
  const roles = S.resp.roles[side] || {}, roster = side === "Blue" ? S.blue_roster : S.red_roster;
  const byRole = {}; Object.entries(roles).forEach(([c, r]) => byRole[r] = c);
  const nm = (k) => SETUP.players.find(p => p.key === k)?.name ?? k;
  $(`${side}-picks`).innerHTML = SETUP.roles.map(role => {
    const c = byRole[role];
    return `<div class="slot"><div class="ico">${c ? icon(c) : ""}</div>
      <div><div class="role">${ROLE_LABEL[role]}</div><div class="pl">${nm(roster[role]) || "—"}</div>
      <div class="ch">${c ?? ""}</div></div></div>`;
  }).join("");
}
function renderRecs() {
  const r = S.resp, mine = r.next && r.next.side === S.mySide;
  if (!r.next) { $("recs").innerHTML = ""; return; }
  if (!mine) {
    $("recs").innerHTML = `<div class="waitcard ${r.next.side}"><b>Opponent's ${r.next.kind}</b><span class="muted">Your suggestions appear as soon as it's your turn. Search below to check a champion meanwhile.</span></div>`;
    return;
  }
  const isPick = r.next.kind === "pick";
  const sortBar = isPick ? `<div class="sortbar"><span class="muted">Top 3 by</span>${[["pro", "Pro priority"], ["blend", "Blend"], ["win", "Win estimate"]]
      .map(([k, l]) => `<button class="${recSort === k ? "on" : ""}" data-sort="${k}">${l}</button>`).join("")}</div>`
    : `<div class="sortbar"><span class="muted">Top 3 bans: how likely they take it × how much it would hurt you</span></div>`;
  const card = (x, n, big) => {
    const chips = isPick
      ? [`<span class="rchip" title="How likely pros are to pick this here: the validated part of the model">${Math.round(x.model_prob * 100)}% pro pick</span>`,
         x.edge ? `<span class="rchip ${x.edge === "slight edge" ? "good" : x.edge === "slight drawback" ? "bad" : ""}" title="Draft effect on win chance. Small, and not significant in backtests.">${esc(x.edge)}</span>` : ""]
      : [`<span class="rchip" title="Chance the enemy picks it next if you leave it open">${Math.round(x.threat * 100)}% they pick it</span>`];
    const warn = (x.warnings || []).map(w => `<div class="rwarn">${esc(w)}</div>`).join("");
    const badge = isPick && x.pro_rank && x.pro_rank <= 3 ? `<span class="probadge">Pro pick #${x.pro_rank}</span>` : "";
    return `<div class="rec ${big ? "big" : "small"}" role="button" tabindex="0" data-c="${esc(x.champion)}" aria-label="${isPick ? "Pick" : "Ban"} ${esc(x.champion)}${n === 0 ? " (top suggestion)" : ""}">
      <div class="ricon">${icon(x.champion)}<span class="rank">${n + 1}</span></div>
      <div class="rbody"><div class="nm">${esc(x.champion)} ${badge}</div>
        <div class="rhead">${esc(x.headline || x.reasons[0] || "")}</div>
        ${big ? `<div class="rchips">${chips.join("")}</div>${warn}
        <details class="rmore"><summary>Why</summary><ul>${x.reasons.map(t => `<li>${esc(t)}</li>`).join("")}</ul></details>` : ""}</div>
      <div class="rgo">${isPick ? "Pick" : "Ban"}</div></div>`;
  };
  const top = r.recommendations.slice(0, 3), rest = r.recommendations.slice(3);
  $("recs").innerHTML = sortBar + `<div class="top3">${top.map((x, n) => card(x, n, true)).join("")}</div>` +
    (rest.length ? `<details class="morerecs" ${moreOpen ? "open" : ""}><summary>${rest.length} more options</summary><div class="restlist">${rest.map((x, n) => card(x, n + 3, false)).join("")}</div></details>` : "");
  const more = $("recs").querySelector(".morerecs"); if (more) more.ontoggle = () => { moreOpen = more.open; };
  $("recs").querySelectorAll(".rec").forEach(el => {
    el.onclick = (e) => { if (e.target.closest("details")) return; choose(el.dataset.c); };
    el.onkeydown = (e) => { if ((e.key === "Enter" || e.key === " ") && !e.target.closest("details")) { e.preventDefault(); choose(el.dataset.c); } };
  });
  $("recs").querySelectorAll("[data-sort]").forEach(b => b.onclick = async () => {
    recSort = b.dataset.sort; $("recs").classList.add("loading");
    try { S.resp = await api("/api/state", payload()); render(); }
    catch (e) { flash(`Couldn't re-sort: ${e.message}`); }
    finally { $("recs").classList.remove("loading"); }
  });
}
function renderGrid() {
  if (!S) return;
  const q = norm($("search").value || ""), used = new Set([...S.actions.filter(Boolean), ...S.locked]);
  // The full grid stays folded away until you search, filter by role or ask for it.
  const show = gridOpen || !!q || !!roleFilter;
  $("grid").classList.toggle("hidden", !show);
  $("gridBtn").textContent = gridOpen ? "Hide all" : "Browse all";
  $("gridBtn").setAttribute("aria-expanded", String(show));
  $("gridBtn").classList.toggle("on", gridOpen);
  if (!show) { $("grid").innerHTML = ""; return; }
  const list = SETUP.champions.filter(c => (!q || norm(c.name).includes(q)) && (!roleFilter || c.roles.includes(roleFilter)))
    .sort((a, b) => a.name.localeCompare(b.name));
  $("grid").innerHTML = list.map(c => { const off = used.has(c.name);
    return `<div class="champ ${off ? "off" : ""}" role="button" tabindex="${off ? -1 : 0}" aria-disabled="${off}" data-c="${esc(c.name)}">${icon(c.name)}${esc(c.name)}</div>`; }).join("")
    || `<p class="muted" style="grid-column:1/-1">No champion matches “${esc($("search").value)}”.</p>`;
  $("grid").querySelectorAll(".champ").forEach(el => {
    el.onclick = () => choose(el.dataset.c);
    el.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(el.dataset.c); } };
  });
}
function showReview() {
  const mine = S.history.filter(h => h.mine);
  const wp = S.resp.win_prob_blue, myWp = S.mySide === "Blue" ? wp : 1 - wp;
  const followed = mine.filter(h => h.rank === 1).length;
  setDone(true);
  setMTab("sugg");
  $("review").innerHTML = `<h3>Draft review — your final draft win probability: ${Math.round(myWp * 100)}%</h3>
    <p class="muted">You took the top suggestion ${followed}/${mine.length} times. Win probability includes team strength, side and pick order, so compare drafts for the same matchup.</p>
    <table><tr><th>Action</th><th>You chose</th><th>Rank in suggestions</th><th>Top 3 suggested</th></tr>
    ${mine.map(h => `<tr><td>${h.kind}</td><td>${h.champ}</td><td>${h.rank ?? "not in top 8"}</td><td>${h.recs.join(", ")}</td></tr>`).join("")}
    </table>
    <div class="result"><strong>Game ${S.gameNo} result:</strong>
      <button class="win" data-r="me">${SERIES.my} won</button>
      <button class="loss" data-r="opp">${SERIES.opp} won</button>
      <button data-r="sim">Simulate (${(myWp * 100).toFixed(0)}% for ${SERIES.my})</button>
    </div>`;
  $("review").querySelectorAll(".result button").forEach(b => b.onclick = () => {
    const r = b.dataset.r === "sim" ? (Math.random() < myWp ? "me" : "opp") : b.dataset.r;
    recordResult(r, myWp);
  });
}

function recordResult(winner, myWp) {
  const picks = S.actions.filter((c, i) => c && ORDER[i][1] === "pick");
  const myPicks = S.actions.filter((c, i) => c && ORDER[i][1] === "pick" && ORDER[i][0] === S.mySide);
  SERIES.games.push({no: S.gameNo, mySide: S.mySide, picks, myPicks,
    oppPicks: picks.filter(c => !myPicks.includes(c)), winner, myWp});
  SERIES.wins[winner] += 1;
  renderSeriesBar();
  const over = SERIES.wins.me >= SERIES.need || SERIES.wins.opp >= SERIES.need || SERIES.games.length >= SERIES.bestOf;
  const gamesHtml = `<div class="games">${SERIES.games.map(g => `<div class="g"><strong>G${g.no}</strong>
      <span class="pill">${g.mySide}</span><span class="${g.winner === "me" ? "pos" : "neg"}">${g.winner === "me" ? "W" : "L"}</span>
      ${g.myPicks.map(c => icon(c)).join("")} <span class="muted">vs</span> ${g.oppPicks.map(c => icon(c)).join("")}
      <span class="muted">draft win prob ${(g.myWp * 100).toFixed(0)}%</span></div>`).join("")}</div>`;
  if (over) {
    const won = SERIES.wins.me > SERIES.wins.opp;
    $("review").innerHTML = `<h3>Series ${won ? "won" : "lost"} ${SERIES.wins.me}–${SERIES.wins.opp}</h3>${gamesHtml}
      ${SERIES.fearless ? `<p class="muted">${seriesLocked().length} champions locked over the series.</p>` : ""}
      <div class="result"><button id="againBtn" class="primary">New series</button></div>`;
    $("againBtn").onclick = () => $("resetBtn").onclick();
    return;
  }
  // Pro rule: the loser of the previous game chooses the side for the next one.
  const loserIsMe = winner === "opp";
  const next = SERIES.wins.me + SERIES.wins.opp + 1;
  $("review").innerHTML = `<h3>${SERIES.my} ${SERIES.wins.me}–${SERIES.wins.opp} ${SERIES.opp}</h3>${gamesHtml}
    <div class="result"><strong>Game ${next}:</strong>
      ${loserIsMe ? `<span class="muted">You lost, so you choose the side.</span>
        <button data-s="Blue">Play Blue</button><button data-s="Red">Play Red</button>`
      : `<span class="muted">${SERIES.opp} lost and chooses the side.</span>
        <button data-s="Red">They take Blue (you Red)</button><button data-s="Blue">They take Red (you Blue)</button>`}
    </div>`;
  $("review").querySelectorAll(".result button").forEach(b => b.onclick = () => startGame(b.dataset.s));
}
init();
