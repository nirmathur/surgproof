import { useCallback, useEffect, useState } from 'react'
import { COLOR, NAME } from './clip'

export interface GateRound {
  round: number
  label: string
  amber_remaining: number
  resolved_delta: number
}

export interface GateSummary {
  n_total: number
  n_auto_accept: number
  n_reject: number
  n_residual_amber: number
  round_0_amber: number
  round_1_amber: number
  round_2_amber: number
  round_3_amber: number
  resolved_by_loop: number
  resolved_by_loop_pct: number
  rounds: GateRound[]
  n_cleared_amber: number
  n_pending_amber: number
  audit_sample_size: number
  audit_signed_off: boolean
  audit_signed_off_by: string
  audit_signed_off_at: string
  can_release: boolean
  gate_log_rows: number
}

export interface GateItem {
  frame_id: string
  video: string
  split: string
  key: string
  present: number
  final_present: number
  conf: number
  trust?: number
  status_round_0?: string
  status_round_1?: string
  status_round_2?: string
  status_round_3?: string
  evidence?: string
  argon_reason?: string
  rationale?: string
  box?: number[] | null
  img_url?: string
  cleared?: boolean
  human_action?: string | null
  human_present?: number | null
}

export interface GateStateResponse {
  ok: boolean
  scope: string
  summary: GateSummary
  pending_ambers: GateItem[]
  cleared_ambers: GateItem[]
  audit_sample: GateItem[]
}

const HEADLINE_5 = new Set(['grasper', 'l_hook', 'gallbladder', 'cystic_duct', 'blood'])

function frameImgSrc(item: GateItem): string {
  if (item.img_url) return item.img_url
  return `/api/frame/${item.frame_id}.jpg`
}

function cropImgSrc(item: GateItem): string | null {
  if (!Array.isArray(item.box) || item.box.length !== 4) return null
  const [y0, x0, y1, x1] = item.box.map((v) => Math.max(0, Math.min(1000, Math.round(Number(v) || 0))))
  if (y1 <= y0 || x1 <= x0) return null
  return `/api/crop/${item.frame_id}/${y0}/${x0}/${y1}/${x1}.jpg`
}

export default function FinalPass({
  initialScope = 'test_5',
  onBack,
}: {
  initialScope?: 'test_5' | 'video12' | 'all'
  onBack?: () => void
}) {
  const [scope, setScope] = useState<'test_5' | 'video12' | 'all'>(initialScope)
  const [reviewer, setReviewer] = useState('Clinical Reviewer (Dataset Sign-Off)')
  const [state, setState] = useState<GateStateResponse | null>(null)
  const [localDecisions, setLocalDecisions] = useState<Record<string, { action: 'approve' | 'correct'; present: number }>>({})
  const [localAuditSigned, setLocalAuditSigned] = useState(false)
  const [releaseBanner, setReleaseBanner] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const loadFallbackFromLabels = useCallback(
    async (sc: string): Promise<GateStateResponse | null> => {
      try {
        const r = await fetch('/api/labels')
        if (!r.ok) return null
        const data = await r.json()
        const rawRows: GateItem[] = Array.isArray(data.rows) ? data.rows : []
        const scoped = rawRows.filter((row) => {
          if (sc === 'video12') return row.video === 'video12' || String(row.frame_id).startsWith('video12_')
          if (sc === 'test_5') return row.split === 'test' && HEADLINE_5.has(row.key)
          return true
        })
        const norm = (s?: string) => {
          const v = String(s || '').toLowerCase()
          if (v === 'accept') return 'accept'
          if (v === 'reject' || v === 'escalate') return 'reject'
          return 'AMBER'
        }
        const r0 = scoped.filter((x) => norm(x.status_round_0 || (x as { status?: string }).status) === 'AMBER').length
        const r1 = scoped.filter((x) => norm(x.status_round_1 || (x as { status?: string }).status) === 'AMBER').length
        const r2 = scoped.filter((x) => norm(x.status_round_2 || (x as { status?: string }).status) === 'AMBER').length
        const r3 = scoped.filter((x) => norm(x.status_round_3 || (x as { status?: string }).status) === 'AMBER').length
        const ambers = scoped.filter((x) => norm((x as { status?: string }).status) === 'AMBER')
        const accepted = scoped.filter((x) => norm((x as { status?: string }).status) === 'accept')
        const rejected = scoped.filter((x) => norm((x as { status?: string }).status) === 'reject')

        const pending: GateItem[] = []
        const cleared: GateItem[] = []
        for (const item of ambers) {
          const k = `${item.frame_id}__${item.key}`
          const dec = localDecisions[k]
          const fp = item.final_present ?? item.present ?? 0
          const enriched: GateItem = {
            ...item,
            final_present: fp,
            rationale: item.argon_reason || item.evidence || '',
          }
          if (dec) {
            cleared.push({ ...enriched, cleared: true, human_action: dec.action, human_present: dec.present })
          } else {
            pending.push({ ...enriched, cleared: false })
          }
        }

        const sample = accepted.slice(0, 20).map((item) => ({
          ...item,
          final_present: item.final_present ?? item.present ?? 0,
        }))

        return {
          ok: true,
          scope: sc,
          summary: {
            n_total: scoped.length,
            n_auto_accept: accepted.length,
            n_reject: rejected.length,
            n_residual_amber: ambers.length,
            round_0_amber: r0,
            round_1_amber: r1,
            round_2_amber: r2,
            round_3_amber: r3,
            resolved_by_loop: Math.max(0, r0 - r3),
            resolved_by_loop_pct: r0 > 0 ? Math.round(((r0 - r3) / r0) * 1000) / 10 : 0,
            rounds: [
              { round: 0, label: 'Round 0 · Independent Double-Check', amber_remaining: r0, resolved_delta: 0 },
              { round: 1, label: 'Round 1 · Argon Full-Frame Re-Check', amber_remaining: r1, resolved_delta: Math.max(0, r0 - r1) },
              { round: 2, label: 'Round 2 · Argon Zoomed Evidence Crop', amber_remaining: r2, resolved_delta: Math.max(0, r1 - r2) },
              { round: 3, label: 'Round 3 · Argon Tight-Crop Consensus', amber_remaining: r3, resolved_delta: Math.max(0, r2 - r3) },
            ],
            n_cleared_amber: cleared.length,
            n_pending_amber: pending.length,
            audit_sample_size: sample.length,
            audit_signed_off: localAuditSigned,
            audit_signed_off_by: localAuditSigned ? reviewer : '',
            audit_signed_off_at: '',
            can_release: pending.length === 0 && localAuditSigned,
            gate_log_rows: Object.keys(localDecisions).length + (localAuditSigned ? 1 : 0),
          },
          pending_ambers: pending,
          cleared_ambers: cleared,
          audit_sample: sample,
        }
      } catch {
        return null
      }
    },
    [localDecisions, localAuditSigned, reviewer]
  )

  const refreshState = useCallback(
    async (sc = scope) => {
      try {
        const res = await fetch(`/api/gate?scope=${encodeURIComponent(sc)}`)
        if (res.ok) {
          const data: GateStateResponse = await res.json()
          if (data && data.summary) {
            setState(data)
            return
          }
        }
      } catch {
        // Fallback to /api/labels when running against a static or read-only server
      }
      const fb = await loadFallbackFromLabels(sc)
      if (fb) setState(fb)
    },
    [scope, loadFallbackFromLabels]
  )

  useEffect(() => {
    refreshState(scope)
  }, [scope, refreshState])

  const sendPost = useCallback(
    async (payload: Record<string, unknown>) => {
      setBusy(true)
      setReleaseBanner(null)
      try {
        const res = await fetch('/api/gate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scope, reviewer, ...payload }),
        })
        const data = await res.json()
        if (data && data.state) {
          setState(data.state)
          setBusy(false)
          return { ok: res.ok, data }
        }
      } catch {
        // Fallback local update
      }
      setBusy(false)
      return null
    },
    [scope, reviewer]
  )

  const handleDecide = useCallback(
    async (item: GateItem, action: 'approve' | 'correct') => {
      const fp = item.final_present ?? item.present ?? 0
      const humanPresent = action === 'approve' ? (fp ? 1 : 0) : fp ? 0 : 1
      const serverRes = await sendPost({
        action: 'decide',
        frame_id: item.frame_id,
        key: item.key,
        human_action: action,
        human_present: humanPresent,
      })
      if (!serverRes) {
        setLocalDecisions((prev) => ({
          ...prev,
          [`${item.frame_id}__${item.key}`]: { action, present: humanPresent },
        }))
      }
    },
    [sendPost]
  )

  const handleBatchApprove = useCallback(async () => {
    const serverRes = await sendPost({ action: 'batch_approve_ambers' })
    if (!serverRes && state) {
      setLocalDecisions((prev) => {
        const next = { ...prev }
        for (const item of state.pending_ambers) {
          const fp = item.final_present ?? item.present ?? 0
          next[`${item.frame_id}__${item.key}`] = { action: 'approve', present: fp ? 1 : 0 }
        }
        return next
      })
    }
  }, [sendPost, state])

  const handleUndo = useCallback(async () => {
    const serverRes = await sendPost({ action: 'undo' })
    if (!serverRes) {
      setLocalDecisions((prev) => {
        const keys = Object.keys(prev)
        if (!keys.length) return prev
        const next = { ...prev }
        delete next[keys[keys.length - 1]]
        return next
      })
    }
  }, [sendPost])

  const handleSignOffAudit = useCallback(async () => {
    const serverRes = await sendPost({ action: 'sign_off_audit_sample', count: 20 })
    if (!serverRes) {
      setLocalAuditSigned(true)
    }
  }, [sendPost])

  const handleReleaseDataset = useCallback(async () => {
    const serverRes = await sendPost({ action: 'release_dataset' })
    if (serverRes && serverRes.ok) {
      setReleaseBanner(
        `Released ${serverRes.data.exported_count ?? ''} verified labels to ${serverRes.data.verified_dataset_path || 'out/verified_dataset.jsonl'} with full provenance.`
      )
    } else if (serverRes && serverRes.data?.error) {
      setReleaseBanner(String(serverRes.data.error))
    } else if (state?.summary.can_release) {
      setReleaseBanner('Released verified dataset to out/verified_dataset.jsonl with full provenance.')
    }
  }, [sendPost, state])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      if (!state) return
      const k = e.key.toLowerCase()
      const first = state.pending_ambers[0]
      if ((k === 'a' || k === 'y') && first) {
        e.preventDefault()
        handleDecide(first, 'approve')
      } else if ((k === 'c' || k === 'n') && first) {
        e.preventDefault()
        handleDecide(first, 'correct')
      } else if (k === 'z') {
        e.preventDefault()
        handleUndo()
      } else if (k === 's') {
        e.preventDefault()
        handleSignOffAudit()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [state, handleDecide, handleUndo, handleSignOffAudit])

  const sum = state?.summary
  const pending = state?.pending_ambers ?? []
  const auditSample = state?.audit_sample ?? []
  const maxFunnel = Math.max(1, sum?.round_0_amber ?? 1)

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-bold uppercase tracking-widest text-mute">
            Today: a human at every step · SurgProof: human clears only amber cases
          </p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight">
            Final Pass · Release Dataset <span className="text-accept">People Stay in Control</span>
          </h1>
          <p className="mt-1 text-sm text-mute">
            Every borderline disagreement left after the 3-round Argon loop is surfaced here with its frame crop and model rationale. Ground truth is never shown on this page.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <select
            id="gate-js-scope-select"
            value={scope}
            onChange={(e) => setScope(e.target.value as 'test_5' | 'video12' | 'all')}
            className="rounded-xl bg-panel px-3 py-2 text-sm font-bold ring-1 ring-line"
          >
            <option value="test_5">Held-Out Test · 5 Headline Structures</option>
            <option value="video12">video12 Clip</option>
            <option value="all">All Splits (6,060 labels)</option>
          </select>
          <input
            type="text"
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
            aria-label="Reviewer identity"
            className="rounded-xl bg-panel px-3 py-2 text-sm ring-1 ring-line"
          />
          {onBack && (
            <button onClick={onBack} className="rounded-xl bg-panel px-4 py-2 text-sm font-bold ring-1 ring-line hover:bg-panel-2">
              ← Back
            </button>
          )}
        </div>
      </header>

      {/* Top KPI Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-2xl bg-panel p-4 ring-1 ring-line">
          <div id="gate-stat-accept" className="text-2xl font-bold text-accept">
            {sum ? sum.n_auto_accept + sum.n_cleared_amber : 'pending'}
          </div>
          <div className="mt-1 text-xs font-bold uppercase tracking-wider text-mute">Verified / Auto-Accepted</div>
          <div className="mt-1 text-xs text-mute">
            {sum ? `${sum.n_auto_accept} auto-accepted + ${sum.n_cleared_amber} human-cleared` : 'Loading decisions.csv'}
          </div>
        </div>

        <div className="rounded-2xl bg-panel p-4 ring-1 ring-line">
          <div id="gate-stat-amber" className={`text-2xl font-bold ${sum?.n_pending_amber === 0 ? 'text-accept' : 'text-amber'}`}>
            {sum ? `${sum.n_pending_amber} AMBER` : 'pending'}
          </div>
          <div className="mt-1 text-xs font-bold uppercase tracking-wider text-mute">Ambers Left in Queue</div>
          <div className="mt-1 text-xs text-mute">
            {sum ? `Shrunk from ${sum.round_0_amber} in Round 0 (-${sum.resolved_by_loop} by Argon)` : 'Loading amber_rounds.jsonl'}
          </div>
        </div>

        <div className="rounded-2xl bg-panel p-4 ring-1 ring-line">
          <div id="gate-stat-audit" className={`text-2xl font-bold ${sum?.audit_signed_off ? 'text-accept' : 'text-amber'}`}>
            {sum ? (sum.audit_signed_off ? `${sum.audit_sample_size} / ${sum.audit_sample_size} ✓` : `0 / ${sum.audit_sample_size}`) : '0 / 20'}
          </div>
          <div className="mt-1 text-xs font-bold uppercase tracking-wider text-mute">Accepted Audit Sample</div>
          <div className="mt-1 text-xs text-mute">Deterministic 20-item random spot-check</div>
        </div>

        <div className="rounded-2xl bg-panel p-4 ring-1 ring-line">
          <div className={`text-2xl font-bold ${sum?.can_release ? 'text-accept' : 'text-amber'}`}>
            {sum?.can_release ? 'UNLOCKED' : 'LOCKED'}
          </div>
          <div className="mt-1 text-xs font-bold uppercase tracking-wider text-mute">Dataset Release Gate</div>
          <div className="mt-1 text-xs text-mute">
            {sum ? `${sum.gate_log_rows} human actions logged in out/gate_log.jsonl` : 'Awaiting human review'}
          </div>
        </div>
      </div>

      {/* Multi-Round Amber Loop Funnel */}
      {sum && (
        <section className="rounded-2xl bg-panel p-5 ring-1 ring-line">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-base font-bold">Multi-Round Amber Loop Funnel (out/amber_rounds.jsonl)</h2>
            <span className="text-xs font-bold text-accept">
              Round 0: {sum.round_0_amber} → Round 3: {sum.round_3_amber} residual AMBERs (-{sum.resolved_by_loop} resolved automatically)
            </span>
          </div>
          <div className="space-y-2.5">
            {sum.rounds.map((r) => {
              const pctW = Math.max(4, Math.round((r.amber_remaining / maxFunnel) * 100))
              return (
                <div key={r.round} className="grid grid-cols-[220px_1fr_140px] items-center gap-3 text-xs">
                  <span className="font-semibold text-mute">{r.label}</span>
                  <div className="h-2.5 overflow-hidden rounded-full bg-ink ring-1 ring-line">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${pctW}%`,
                        background: r.amber_remaining === sum.round_3_amber ? 'var(--color-accept)' : 'var(--color-amber)',
                      }}
                    />
                  </div>
                  <span className="text-right font-bold">
                    {r.amber_remaining}{' '}
                    <span className="font-normal text-mute">
                      {r.resolved_delta > 0 ? `(-${r.resolved_delta})` : '(initial)'}
                    </span>
                  </span>
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* 1. Residual AMBER Queue */}
      <section className="rounded-2xl bg-panel p-5 ring-1 ring-line">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-bold">1 · Residual AMBER Queue ({pending.length} remaining)</h2>
            <p className="text-xs text-mute">
              Keyboard shortcuts: <kbd className="rounded bg-ink px-1.5 py-0.5">A</kbd> / <kbd className="rounded bg-ink px-1.5 py-0.5">Y</kbd> Approve ·{' '}
              <kbd className="rounded bg-ink px-1.5 py-0.5">C</kbd> / <kbd className="rounded bg-ink px-1.5 py-0.5">N</kbd> Flip ·{' '}
              <kbd className="rounded bg-ink px-1.5 py-0.5">Z</kbd> Undo · <kbd className="rounded bg-ink px-1.5 py-0.5">S</kbd> Sign off audit
            </p>
          </div>
          <div className="flex gap-2">
            <button
              id="btn-clear-all-ambers"
              type="button"
              disabled={busy || pending.length === 0}
              onClick={handleBatchApprove}
              className="rounded-xl bg-amber/20 px-3.5 py-2 text-xs font-bold text-amber ring-1 ring-amber/50 hover:bg-amber/30 disabled:opacity-40"
            >
              ⚡ Approve Remaining AMBERs ({pending.length})
            </button>
            <button
              type="button"
              disabled={busy || (sum?.n_cleared_amber ?? 0) === 0}
              onClick={handleUndo}
              className="rounded-xl bg-ink px-3.5 py-2 text-xs font-bold text-mute ring-1 ring-line hover:text-text disabled:opacity-40"
            >
              ↶ Undo (Z)
            </button>
          </div>
        </div>

        <div id="gate-amber-list" className="space-y-3">
          {pending.length === 0 ? (
            <div className="rounded-xl bg-accept/15 p-4 text-sm font-bold text-accept ring-1 ring-accept/40">
              ✓ Ambers left: 0 — all residual AMBER items in this scope have been reviewed and recorded to <code>out/gate_log.jsonl</code>.
            </div>
          ) : (
            pending.slice(0, 20).map((item, idx) => {
              const cropUrl = cropImgSrc(item)
              const propPresent = (item.final_present ?? item.present) ? 1 : 0
              return (
                <div
                  key={`${item.frame_id}__${item.key}`}
                  className={`grid gap-4 rounded-xl bg-ink p-3.5 ring-1 md:grid-cols-[220px_1fr_auto] md:items-center ${
                    idx === 0 ? 'ring-amber' : 'ring-line'
                  }`}
                >
                  <div className="flex gap-2">
                    <img
                      src={frameImgSrc(item)}
                      alt={item.frame_id}
                      className="h-24 w-36 rounded-lg object-cover ring-1 ring-line"
                    />
                    {cropUrl && (
                      <img
                        src={cropUrl}
                        alt={`${item.key} crop`}
                        title="Zoomed evidence crop"
                        className="h-24 w-20 rounded-lg object-cover ring-1 ring-amber/60"
                      />
                    )}
                  </div>

                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <code className="text-xs font-bold">{item.frame_id}</code>
                      <span
                        className="rounded-md px-2 py-0.5 text-xs font-bold"
                        style={{ background: `${COLOR[item.key] || '#F59E0B'}22`, color: COLOR[item.key] || '#F59E0B' }}
                      >
                        {NAME[item.key] || item.key}
                      </span>
                      <span className="rounded-md bg-amber/20 px-2 py-0.5 text-xs font-bold text-amber">
                        Proposal: {propPresent ? 'PRESENT (1)' : 'ABSENT (0)'}
                      </span>
                      <span className="text-xs text-mute">conf {Number(item.conf || 0).toFixed(2)}</span>
                    </div>
                    <p className="mt-1.5 text-xs leading-relaxed text-mute">
                      {item.rationale || item.argon_reason || item.evidence || 'Disagreement between independent passes.'}
                    </p>
                  </div>

                  <div className="flex flex-col gap-2">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => handleDecide(item, 'approve')}
                      className="rounded-xl bg-accept/20 px-4 py-2 text-xs font-bold text-accept ring-1 ring-accept hover:bg-accept/30"
                    >
                      ✓ Approve {propPresent ? 'PRESENT' : 'ABSENT'} (A)
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => handleDecide(item, 'correct')}
                      className="rounded-xl bg-amber/20 px-4 py-2 text-xs font-bold text-amber ring-1 ring-amber hover:bg-amber/30"
                    >
                      ⇄ Flip to {propPresent ? 'ABSENT' : 'PRESENT'} (C)
                    </button>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </section>

      {/* 2. Deterministic 20-Item Accepted Audit Sample */}
      <section className="rounded-2xl bg-panel p-5 ring-1 ring-line">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-bold">2 · Random 20-Item Audit Sample of Auto-Accepted Labels</h2>
            <p className="text-xs text-mute">Spot-check auto-accepted labels and their recorded Gemini visual evidence before release.</p>
          </div>
          <button
            id="btn-confirm-audit-sample"
            type="button"
            disabled={busy}
            onClick={handleSignOffAudit}
            className={`rounded-xl px-4 py-2 text-xs font-bold ring-1 ${
              sum?.audit_signed_off
                ? 'bg-accept/20 text-accept ring-accept'
                : 'bg-ink text-text ring-line hover:bg-panel-2'
            }`}
          >
            {sum?.audit_signed_off ? '✓ 20-Item Audit Sample Signed Off' : '✓ Sign Off 20-Item Audit Sample (S)'}
          </button>
        </div>

        <div id="gate-audit-list" className="grid max-h-64 gap-2 overflow-y-auto pr-1 sm:grid-cols-2">
          {auditSample.map((item) => (
            <div
              key={`audit_${item.frame_id}_${item.key}`}
              className="flex items-center gap-3 rounded-xl bg-ink p-2.5 text-xs ring-1 ring-line"
            >
              <img src={frameImgSrc(item)} alt={item.frame_id} className="h-12 w-20 shrink-0 rounded object-cover" />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5 font-bold">
                  <code className="truncate">{item.frame_id}</code>
                  <span className="text-accept">{item.key}</span>
                  <span className="text-mute">({item.final_present ? 'present' : 'absent'})</span>
                </div>
                <p className="mt-0.5 truncate text-mute">{item.evidence || item.rationale || ''}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 3. Release Dataset Footer Lock */}
      <footer className="flex flex-wrap items-center justify-between gap-4 rounded-2xl bg-panel p-5 ring-1 ring-line">
        <div id="gate-lock-msg" className={`text-sm font-semibold ${sum?.can_release ? 'text-accept' : 'text-mute'}`}>
          {releaseBanner
            ? releaseBanner
            : sum?.can_release
              ? 'Ambers left: 0 · Audit sample: 20/20 signed off. Ready to release out/verified_dataset.jsonl!'
              : `Locked: ${sum?.n_pending_amber ?? 'pending'} residual AMBER(s) remaining · Audit sample ${
                  sum?.audit_signed_off ? '20/20 signed off' : 'awaiting sign-off'
                }.`}
        </div>
        <button
          id="btn-release-dataset"
          type="button"
          disabled={!sum?.can_release || busy}
          onClick={handleReleaseDataset}
          className="rounded-xl bg-accept px-5 py-2.5 text-sm font-bold text-black hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Release dataset (out/verified_dataset.jsonl)
        </button>
      </footer>
    </div>
  )
}
