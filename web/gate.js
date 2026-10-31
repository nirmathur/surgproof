/* SurgProof Final Pass & Release Dataset Component (web/gate.js)
 *
 * Implements the "People Stay in Control" Final Pass panel/drawer for the
 * SurgProof web studio (promo/DESIGN.md dark neutral palette: #0A0B0E canvas,
 * #12141C card surface, #232734 1px border, #F4F5F8/#8A909E text, #10B981
 * accept, #F59E0B amber, #EF4444 reject, SF Pro / Inter typography).
 *
 * Talks to GET /api/gate and POST /api/gate on web/serve.py:
 *   - Displays the shrinking 4-round Amber Loop funnel (Round 0 -> Round 3)
 *   - Displays residual AMBER items with frame preview, zoomed box crop,
 *     model proposal, confidence, and verbatim rationale/evidence
 *   - Keyboard-first review: A/Y = Approve, C/N = Correct/Flip, Z = Undo,
 *     S = Sign off 20-item accepted audit sample
 *   - Keeps "Release dataset" strictly disabled until 0 residual AMBERs remain
 *     and the 20-item accepted audit sample is signed off
 *   - Never reads or displays ground truth
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    const api = factory();
    root.FinalPassGate = api;
    root.Gate = api;
  }
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  const TOKENS = {
    bgCanvas: "#0A0B0E",
    bgCard: "#12141C",
    border: "#232734",
    textPrimary: "#F4F5F8",
    textSecondary: "#8A909E",
    textBody: "#D8DCE6",
    accent: "#3B82F6",
    acceptText: "#10B981",
    acceptFill: "#092E24",
    amberText: "#F59E0B",
    amberFill: "#362407",
    rejectText: "#EF4444",
    rejectFill: "#331014",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display', 'Inter', sans-serif",
  };

  const state = {
    mountedEl: null,
    scope: "test_5",
    reviewer: "clinical_reviewer",
    summary: null,
    funnels: {},
    pendingAmbers: [],
    clearedAmbers: [],
    auditSample: [],
    selectedIndex: 0,
    lastReleaseMsg: "",
    loading: false,
    onUpdate: null,
  };

  function escHtml(val) {
    return String(val == null ? "" : val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtConf(val) {
    const num = Number(val);
    if (!Number.isFinite(num)) return "pending";
    return num.toFixed(2);
  }

  async function fetchGateState(scope) {
    const sc = scope || state.scope || "test_5";
    for (const base of ["/api/gate", "/gate"]) {
      try {
        const res = await fetch(`${base}?scope=${encodeURIComponent(sc)}`, {
          method: "GET",
          headers: { Accept: "application/json" },
        });
        if (res.ok) {
          const data = await res.json();
          applyServerState(data);
          return data;
        }
      } catch (_err) {
        // continue to fallback
      }
    }
    return null;
  }

  async function postGate(payload) {
    const body = Object.assign({ scope: state.scope, reviewer: state.reviewer }, payload || {});
    for (const ep of ["/api/gate", "/gate"]) {
      try {
        const res = await fetch(ep, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data && data.state) {
          applyServerState(data.state);
        }
        return { status: res.status, ok: res.ok, data };
      } catch (_err) {
        // continue
      }
    }
    return { status: 0, ok: false, data: null };
  }

  function applyServerState(data) {
    if (!data || typeof data !== "object") return;
    if (data.scope) state.scope = data.scope;
    state.summary = data.summary || null;
    state.funnels = data.funnels || {};
    state.pendingAmbers = Array.isArray(data.pending_ambers) ? data.pending_ambers : [];
    state.clearedAmbers = Array.isArray(data.cleared_ambers) ? data.cleared_ambers : [];
    state.auditSample = Array.isArray(data.audit_sample) ? data.audit_sample : [];
    if (state.selectedIndex >= state.pendingAmbers.length) {
      state.selectedIndex = Math.max(0, state.pendingAmbers.length - 1);
    }
    syncHostBadges();
    render();
    if (typeof state.onUpdate === "function") {
      state.onUpdate(getState());
    }
  }

  function syncHostBadges() {
    if (typeof document === "undefined" || !state.summary) return;
    const sum = state.summary;
    const badge = document.getElementById("amber-count-badge");
    if (badge) {
      badge.textContent = `${sum.n_pending_amber} AMBER`;
    }
    const outerReleaseBtn = document.getElementById("btn-release-dataset");
    const outerLockMsg = document.getElementById("gate-lock-msg");
    const canRelease = Boolean(sum.can_release);
    if (outerReleaseBtn) {
      outerReleaseBtn.disabled = !canRelease;
      if (!outerReleaseBtn._gateBound) {
        outerReleaseBtn._gateBound = true;
        outerReleaseBtn.addEventListener("click", (e) => {
          e.preventDefault();
          releaseDataset();
        });
      }
    }
    if (outerLockMsg) {
      outerLockMsg.style.color = canRelease ? TOKENS.acceptText : TOKENS.textSecondary;
      outerLockMsg.textContent = state.lastReleaseMsg
        ? state.lastReleaseMsg
        : canRelease
          ? "Ambers left: 0 · Audit sample: 20/20 signed off. Ready to release dataset!"
          : `Locked: Ambers left: ${sum.n_pending_amber} · Audit sample: ${sum.audit_signed_off ? "20/20" : "0/20"}.`;
    }
  }

  function renderFunnelHtml(summary) {
    if (!summary || !Array.isArray(summary.rounds)) {
      return `<div style="color:${TOKENS.textSecondary};font-size:13px;">Amber Loop rounds: pending</div>`;
    }
    const maxCount = Math.max(1, Number(summary.round_0_amber) || 1);
    const rowsHtml = summary.rounds
      .map((r) => {
        const rem = Number(r.amber_remaining) || 0;
        const widthPct = Math.max(4, Math.round((rem / maxCount) * 100));
        const deltaTxt = r.resolved_delta > 0 ? `(-${r.resolved_delta} resolved)` : "(initial double-check)";
        return (
          `<div style="display:grid;grid-template-columns:240px 1fr 130px;align-items:center;gap:10px;margin:5px 0;font-size:12.5px;">` +
          `<div style="color:${TOKENS.textBody};">${escHtml(r.label)}</div>` +
          `<div style="background:#1A1D28;border-radius:4px;height:10px;overflow:hidden;border:1px solid ${TOKENS.border};">` +
          `<div style="width:${widthPct}%;height:100%;background:${rem === summary.round_3_amber ? TOKENS.acceptText : TOKENS.amberText};transition:width 0.35s cubic-bezier(0.22,1,0.36,1);"></div>` +
          `</div>` +
          `<div style="text-align:right;color:${TOKENS.textPrimary};font-variant-numeric:tabular-nums;">` +
          `<strong>${rem}</strong> <span style="color:${TOKENS.textSecondary};font-size:11px;">${escHtml(deltaTxt)}</span>` +
          `</div>` +
          `</div>`
        );
      })
      .join("");

    return (
      `<div style="background:${TOKENS.bgCanvas};border:1px solid ${TOKENS.border};border-radius:8px;padding:12px 14px;margin-bottom:12px;">` +
      `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">` +
      `<span style="font-size:12px;font-weight:600;color:${TOKENS.textSecondary};letter-spacing:0.04em;text-transform:uppercase;">Multi-Round Amber Loop Funnel (out/amber_rounds.jsonl)</span>` +
      `<span style="font-size:12px;color:${TOKENS.acceptText};">Round 0: ${summary.round_0_amber} &rarr; Round 3: ${summary.round_3_amber} residual AMBERs (-${summary.resolved_by_loop} resolved by Argon)</span>` +
      `</div>` +
      rowsHtml +
      `</div>`
    );
  }

  function renderActiveAmberCardHtml(item, idx, total) {
    if (!item) {
      return (
        `<div style="background:${TOKENS.acceptFill};border:1px solid ${TOKENS.acceptText};border-radius:8px;padding:14px 16px;color:${TOKENS.textPrimary};margin-bottom:12px;">` +
        `<div style="font-size:15px;font-weight:600;color:${TOKENS.acceptText};">&#10003; All residual AMBER cases in this scope are cleared (0 remaining).</div>` +
        `<div style="font-size:13px;color:${TOKENS.textBody};margin-top:4px;">Every borderline disagreement has been reviewed and logged to <code>out/gate_log.jsonl</code>.</div>` +
        `</div>`
      );
    }
    const proposal = item.final_present ? "PRESENT (1)" : "ABSENT (0)";
    const flipped = item.final_present ? "ABSENT (0)" : "PRESENT (1)";
    const imgBlock = item.img_url
      ? `<img src="${escHtml(item.img_url)}" alt="${escHtml(item.frame_id)}" style="width:200px;height:112px;object-fit:cover;border-radius:6px;border:1px solid ${TOKENS.border};background:#000;" />`
      : `<div style="width:200px;height:112px;border-radius:6px;border:1px solid ${TOKENS.border};display:flex;align-items:center;justify-content:center;color:${TOKENS.textSecondary};font-size:12px;">Frame preview</div>`;

    return (
      `<div style="background:${TOKENS.bgCanvas};border:1px solid ${TOKENS.amberText};border-radius:8px;padding:14px;margin-bottom:12px;">` +
      `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">` +
      `<div style="font-size:13px;color:${TOKENS.amberText};font-weight:600;">Active Residual AMBER (${idx + 1} of ${total}) &middot; Rapid Keyboard Deck</div>` +
      `<div style="font-size:12px;color:${TOKENS.textSecondary};">Shortcuts: <kbd>A</kbd>/<kbd>Y</kbd> Approve &middot; <kbd>C</kbd>/<kbd>N</kbd> Flip &middot; <kbd>Z</kbd> Undo</div>` +
      `</div>` +
      `<div style="display:grid;grid-template-columns:208px 1fr auto;gap:14px;align-items:center;">` +
      imgBlock +
      `<div>` +
      `<div style="font-size:15px;font-weight:600;color:${TOKENS.textPrimary};"><code>${escHtml(item.frame_id)}</code> &middot; <span style="color:${TOKENS.amberText};">${escHtml(item.key)}</span></div>` +
      `<div style="font-size:13px;color:${TOKENS.textBody};margin-top:4px;">Model proposal: <strong>${proposal}</strong> &middot; conf <code>${fmtConf(item.conf)}</code> &middot; rounds <code>${escHtml(item.status_round_0)} &rarr; ${escHtml(item.status_round_1)} &rarr; ${escHtml(item.status_round_2)} &rarr; ${escHtml(item.status_round_3)}</code></div>` +
      `<div style="font-size:12.5px;color:${TOKENS.textSecondary};margin-top:6px;line-height:1.45;">${escHtml(item.rationale || item.evidence || "Disagreement between independent passes.")}</div>` +
      `</div>` +
      `<div style="display:flex;flex-direction:column;gap:8px;min-width:170px;">` +
      `<button type="button" data-gate-act="approve-one" data-fid="${escHtml(item.frame_id)}" data-key="${escHtml(item.key)}" data-pres="${item.final_present ? 1 : 0}" style="background:${TOKENS.acceptFill};color:${TOKENS.acceptText};border:1px solid ${TOKENS.acceptText};border-radius:6px;padding:8px 12px;font-weight:600;font-size:13px;cursor:pointer;">&#10003; Approve ${proposal} (A)</button>` +
      `<button type="button" data-gate-act="correct-one" data-fid="${escHtml(item.frame_id)}" data-key="${escHtml(item.key)}" data-pres="${item.final_present ? 0 : 1}" style="background:${TOKENS.amberFill};color:${TOKENS.amberText};border:1px solid ${TOKENS.amberText};border-radius:6px;padding:8px 12px;font-weight:600;font-size:13px;cursor:pointer;">&#8644; Flip to ${flipped} (C)</button>` +
      `</div>` +
      `</div>` +
      `</div>`
    );
  }

  function renderAuditSampleHtml(auditSample, signedOff) {
    const itemsHtml = auditSample
      .slice(0, 8)
      .map((item) => {
        const pTxt = item.final_present ? "present" : "absent";
        return (
          `<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 10px;border-bottom:1px solid ${TOKENS.border};font-size:12px;">` +
          `<div><strong style="color:${TOKENS.textPrimary};">${escHtml(item.frame_id)}</strong> &middot; <code>${escHtml(item.key)}</code> &middot; <span style="color:${TOKENS.acceptText};">${pTxt} (conf ${fmtConf(item.conf)})</span></div>` +
          `<div style="color:${TOKENS.textSecondary};max-width:420px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(item.evidence || "")}</div>` +
          `</div>`
        );
      })
      .join("");

    return (
      `<div style="background:${TOKENS.bgCanvas};border:1px solid ${TOKENS.border};border-radius:8px;padding:12px 14px;margin-bottom:12px;">` +
      `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">` +
      `<div>` +
      `<span style="font-size:12px;font-weight:600;color:${TOKENS.textSecondary};text-transform:uppercase;letter-spacing:0.04em;">Random 20-Item Auto-Accepted Audit Sample</span>` +
      `<span style="font-size:12px;color:${signedOff ? TOKENS.acceptText : TOKENS.amberText};margin-left:10px;">${signedOff ? "Signed off (20/20)" : "Awaiting sign-off (0/20)"}</span>` +
      `</div>` +
      `<button type="button" id="gate-js-sign-off-audit" data-gate-act="sign-off-audit" style="background:${signedOff ? TOKENS.acceptFill : "#1A1D28"};color:${signedOff ? TOKENS.acceptText : TOKENS.textPrimary};border:1px solid ${signedOff ? TOKENS.acceptText : TOKENS.border};border-radius:6px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;">` +
      `${signedOff ? "&#10003; 20-Item Audit Signed Off" : "&#10003; Sign Off 20-Item Accepted Audit Sample (S)"}` +
      `</button>` +
      `</div>` +
      `<div style="max-height:135px;overflow-y:auto;border:1px solid ${TOKENS.border};border-radius:6px;">${itemsHtml}</div>` +
      `</div>`
    );
  }

  function render() {
    if (!state.mountedEl) return;
    const sum = state.summary || {
      n_total: 0,
      n_auto_accept: 0,
      n_residual_amber: 0,
      n_cleared_amber: 0,
      n_pending_amber: 0,
      audit_sample_size: state.auditSample.length,
      audit_signed_off: false,
      can_release: false,
      gate_log_rows: 0,
    };

    const currentAmber = state.pendingAmbers[state.selectedIndex] || state.pendingAmbers[0] || null;
    const canRelease = Boolean(sum.can_release);

    state.mountedEl.innerHTML =
      `<div class="surgproof-final-pass-panel" style="font-family:${TOKENS.fontFamily};color:${TOKENS.textPrimary};background:${TOKENS.bgCard};border:1px solid ${TOKENS.border};border-radius:10px;padding:16px;">` +
      `<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:12px;border-bottom:1px solid ${TOKENS.border};padding-bottom:10px;">` +
      `<div>` +
      `<div style="font-size:18px;font-weight:600;color:${TOKENS.textPrimary};">Final Pass &middot; Release Dataset</div>` +
      `<div style="font-size:12.5px;color:${TOKENS.textSecondary};margin-top:2px;">Today: a human at every step. <strong>SurgProof: human clears only amber cases.</strong> Zero ground truth shown.</div>` +
      `</div>` +
      `<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">` +
      `<select id="gate-js-scope-select" style="background:${TOKENS.bgCanvas};color:${TOKENS.textPrimary};border:1px solid ${TOKENS.border};border-radius:6px;padding:5px 9px;font-size:12px;">` +
      `<option value="test_5" ${state.scope === "test_5" ? "selected" : ""}>Held-Out Test &middot; 5 Headline Structures (8 AMBER)</option>` +
      `<option value="video12" ${state.scope === "video12" ? "selected" : ""}>video12 Clip</option>` +
      `<option value="all" ${state.scope === "all" ? "selected" : ""}>All Splits (6,060 labels)</option>` +
      `</select>` +
      `<span id="gate-stat-amber" style="background:${sum.n_pending_amber === 0 ? TOKENS.acceptFill : TOKENS.amberFill};color:${sum.n_pending_amber === 0 ? TOKENS.acceptText : TOKENS.amberText};border:1px solid ${sum.n_pending_amber === 0 ? TOKENS.acceptText : TOKENS.amberText};border-radius:6px;padding:4px 10px;font-size:12px;font-weight:600;">Ambers left: ${sum.n_pending_amber}</span>` +
      `<span id="gate-stat-audit" style="background:${sum.audit_signed_off ? TOKENS.acceptFill : "#1A1D28"};color:${sum.audit_signed_off ? TOKENS.acceptText : TOKENS.textSecondary};border:1px solid ${TOKENS.border};border-radius:6px;padding:4px 10px;font-size:12px;font-weight:500;">Audit sample: ${sum.audit_signed_off ? `${sum.audit_sample_size}/${sum.audit_sample_size}` : `0/${sum.audit_sample_size}`}</span>` +
      `</div>` +
      `</div>` +
      renderFunnelHtml(sum) +
      `<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;">` +
      `<div style="font-size:12.5px;color:${TOKENS.textSecondary};">Auto-accepted: <strong id="gate-stat-accept" style="color:${TOKENS.acceptText};">${sum.n_auto_accept}</strong> &middot; Cleared by human: <strong>${sum.n_cleared_amber}</strong> &middot; Gate log rows: <code>${sum.gate_log_rows}</code></div>` +
      `<div style="display:flex;gap:8px;">` +
      `<button type="button" id="btn-clear-all-ambers" data-gate-act="batch-approve" ${sum.n_pending_amber === 0 ? "disabled" : ""} style="background:${TOKENS.amberFill};color:${TOKENS.amberText};border:1px solid ${TOKENS.amberText};border-radius:6px;padding:5px 10px;font-size:12px;font-weight:600;cursor:pointer;opacity:${sum.n_pending_amber === 0 ? 0.45 : 1};">&#9889; Approve All ${sum.n_pending_amber} Residual Ambers</button>` +
      `<button type="button" data-gate-act="undo" ${sum.n_cleared_amber === 0 ? "disabled" : ""} style="background:#1A1D28;color:${TOKENS.textSecondary};border:1px solid ${TOKENS.border};border-radius:6px;padding:5px 10px;font-size:12px;cursor:pointer;opacity:${sum.n_cleared_amber === 0 ? 0.45 : 1};">&#8630; Undo (Z)</button>` +
      `</div>` +
      `</div>` +
      `<div id="gate-amber-list">${renderActiveAmberCardHtml(currentAmber, state.selectedIndex, state.pendingAmbers.length)}</div>` +
      `<div id="gate-audit-list">${renderAuditSampleHtml(state.auditSample, sum.audit_signed_off)}</div>` +
      `<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;padding-top:8px;border-top:1px solid ${TOKENS.border};">` +
      `<div id="gate-js-lock-banner" style="font-size:13px;color:${canRelease ? TOKENS.acceptText : TOKENS.textSecondary};">` +
      (state.lastReleaseMsg
        ? escHtml(state.lastReleaseMsg)
        : canRelease
          ? "Ready: 0 residual AMBERs + 20-item audit signed off. Click 'Release dataset' to export out/verified_dataset.jsonl."
          : `Locked until ${sum.n_pending_amber} residual AMBER case(s) are cleared and the 20-item audit sample is signed off.`) +
      `</div>` +
      `<button type="button" id="gate-js-release-btn" data-gate-act="release-dataset" ${canRelease ? "" : "disabled"} style="background:${canRelease ? TOKENS.acceptFill : "#1A1D28"};color:${canRelease ? TOKENS.acceptText : TOKENS.textSecondary};border:1px solid ${canRelease ? TOKENS.acceptText : TOKENS.border};border-radius:6px;padding:9px 16px;font-size:13px;font-weight:600;cursor:${canRelease ? "pointer" : "not-allowed"};opacity:${canRelease ? 1 : 0.5};">` +
      `Release dataset &rarr; out/verified_dataset.jsonl` +
      `</button>` +
      `</div>` +
      `</div>`;

    bindMountedControls();
  }

  function bindMountedControls() {
    if (!state.mountedEl) return;
    const scopeSel = state.mountedEl.querySelector("#gate-js-scope-select");
    if (scopeSel) {
      scopeSel.addEventListener("change", (e) => {
        state.scope = e.target.value || "test_5";
        state.lastReleaseMsg = "";
        fetchGateState(state.scope);
      });
    }

    state.mountedEl.querySelectorAll("[data-gate-act]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const act = btn.getAttribute("data-gate-act");
        if (act === "approve-one") {
          await decide(btn.getAttribute("data-fid"), btn.getAttribute("data-key"), "approve", Number(btn.getAttribute("data-pres")));
        } else if (act === "correct-one") {
          await decide(btn.getAttribute("data-fid"), btn.getAttribute("data-key"), "correct", Number(btn.getAttribute("data-pres")));
        } else if (act === "batch-approve") {
          await batchApprove();
        } else if (act === "undo") {
          await undo();
        } else if (act === "sign-off-audit") {
          await signOffAudit();
        } else if (act === "release-dataset") {
          await releaseDataset();
        }
      });
    });
  }

  async function decide(frameId, key, humanAction, humanPresent) {
    if (!frameId || !key) return null;
    state.lastReleaseMsg = "";
    return postGate({
      action: "decide",
      frame_id: frameId,
      key,
      human_action: humanAction || "approve",
      human_present: humanPresent,
    });
  }

  async function batchApprove() {
    state.lastReleaseMsg = "";
    return postGate({ action: "batch_approve_ambers" });
  }

  async function undo() {
    state.lastReleaseMsg = "";
    return postGate({ action: "undo" });
  }

  async function signOffAudit() {
    state.lastReleaseMsg = "";
    return postGate({ action: "sign_off_audit_sample", count: 20 });
  }

  async function releaseDataset() {
    const res = await postGate({ action: "release_dataset" });
    if (res && res.ok && res.data) {
      state.lastReleaseMsg = `Released ${res.data.exported_count || ""} verified labels to ${res.data.verified_dataset_path || "out/verified_dataset.jsonl"}`;
      render();
    } else if (res && res.data && res.data.error) {
      state.lastReleaseMsg = res.data.error;
      render();
    }
    return res;
  }

  function mount(el, opts) {
    const options = opts || {};
    if (options.scope) state.scope = options.scope;
    if (options.reviewer) state.reviewer = options.reviewer;
    if (typeof options.onUpdate === "function") state.onUpdate = options.onUpdate;
    state.mountedEl = typeof el === "string" ? document.querySelector(el) : el;
    render();
    fetchGateState(state.scope);
    return api;
  }

  function getState() {
    return {
      scope: state.scope,
      summary: state.summary,
      pendingAmbers: state.pendingAmbers.slice(),
      clearedAmbers: state.clearedAmbers.slice(),
      auditSample: state.auditSample.slice(),
    };
  }

  const api = {
    mount,
    refresh: fetchGateState,
    decide,
    batchApprove,
    undo,
    signOffAudit,
    releaseDataset,
    getState,
    TOKENS,
  };

  return api;
});
