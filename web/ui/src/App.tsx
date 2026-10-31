import { useEffect, useMemo, useState } from 'react'
import { api, normStatus, pct, type Claim, type LabelRow, type Overview, type Status } from './api'

const KEY_NAME: Record<string, string> = {
  grasper: 'Grasper', l_hook: 'L-hook', liver: 'Liver', gallbladder: 'Gallbladder', cystic_duct: 'Cystic duct', blood: 'Blood',
}
const STATUS_COLOR: Record<Status, string> = { accept: 'var(--color-accept)', AMBER: 'var(--color-amber)', reject: 'var(--color-reject)' }
const STATUS_WORD: Record<Status, string> = { accept: 'Accepted', AMBER: 'Unsure', reject: 'Rejected' }

const STAGES = [
  { id: 'teach', title: 'Teaches itself', body: 'Writes its own manual from public teaching video and papers.' },
  { id: 'monitor', title: 'Labels real frames', body: 'Gemini Flash marks each structure and draws a box.' },
  { id: 'monitor', title: 'Double-checks', body: 'A second, independent pass: different prompt, zoomed crop.' },
  { id: 'amber', title: 'Clears the unsure', body: 'Argon re-examines every amber, round after round.' },
  { id: 'amber', title: 'Human final pass', body: 'A person signs off what is still unsure before release.' },
  { id: 'student', title: 'Trains a specialist', body: 'Only on labels that survived. Scored on unseen videos.' },
  { id: 'narrate', title: 'Narrates', body: 'Says only what it verified. Everything else is withheld.' },
]

export default function App() {
  const [ov, setOv] = useState<Overview | null>(null)
  const [labels, setLabels] = useState<{ source: string | null; has_status: boolean; rows: LabelRow[] } | null>(null)
  const [claims, setClaims] = useState<{ source: string | null; claims: Claim[] } | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.overview(), api.labels(), api.narration()])
      .then(([o, l, n]) => { setOv(o); setLabels(l); setClaims(n) })
      .catch((e) => setErr(String(e)))
  }, [])

  if (err) return <p className="p-10 text-reject">Can't reach the data server: {err}. Start it with <code>uvicorn web.server:app --port 8600</code>.</p>
  if (!ov || !labels) return <p className="p-10 text-mute">Loading pipeline outputs…</p>

  const res = ov.results ?? {}
  const nTest = res.n_test_labels
  const smoke = typeof nTest === 'number' && nTest < 1000

  return (
    <div className="min-h-screen">
      <header className="mx-auto flex max-w-7xl items-baseline justify-between gap-6 px-6 pt-6">
        <span className="text-xl font-bold tracking-tight">SurgProof</span>
        {smoke && (
          <span className="rounded-full bg-amber/15 px-3 py-1 text-sm text-drape">
            Interim smoke run: {nTest} held-out test labels scored. Numbers update when the full run lands.
          </span>
        )}
      </header>

      <Hero ov={ov} />
      <Pipeline />
      <Monitor labels={labels} testVideos={ov.test_videos} />
      <Teach ov={ov} />
      <AmberLoop ov={ov} />
      <Proof ov={ov} />
      <StudentSection ov={ov} />
      <Tracker ov={ov} />
      <Narration claims={claims} />
      <Atlas ov={ov} />
      <Footer />
    </div>
  )
}

function Section({ id, title, lead, children }: { id: string; title: string; lead?: string; children: React.ReactNode }) {
  return (
    <section id={id} className="mx-auto max-w-7xl scroll-mt-6 px-6 py-16">
      <h2 className="text-3xl font-bold tracking-tight">{title}</h2>
      {lead && <p className="mt-2 max-w-[70ch] text-lg text-mute">{lead}</p>}
      <div className="mt-8">{children}</div>
    </section>
  )
}

function Pending({ what }: { what: string }) {
  return <p className="rounded-lg border border-dashed border-steel p-6 text-mute">Waiting for {what}. Nothing is shown until the real file exists.</p>
}

function Hero({ ov }: { ov: Overview }) {
  const a5 = ov.results?.accepted_5_informative
  return (
    <section className="mx-auto max-w-7xl px-6 pb-10 pt-14">
      <h1 className="text-5xl font-bold leading-[1.05] tracking-tight sm:text-7xl">No human labels.<br />Only proof.</h1>
      <p className="mt-6 max-w-[62ch] text-xl leading-relaxed text-drape-2">
        Surgical AI is starved of expert labels. SurgProof teaches itself laparoscopic cholecystectomy from public
        sources, labels real operating video, and checks its own work, so a surgeon only looks at what it
        could not settle.
      </p>
      {a5?.accuracy !== undefined ? (
        <p className="mt-10 max-w-[48ch] text-2xl leading-snug">
          On six videos it never saw, the labels it accepted were{' '}
          <strong className="text-4xl font-bold text-accept">{pct(a5.accuracy)}</strong> correct, and it accepted{' '}
          <strong className="text-4xl font-bold">{pct(a5.coverage, 0)}</strong> of them.
          <span className="mt-2 block text-base text-mute">
            {a5.n_test_labels_excl_liver} test labels, 5 informative structures (liver is in every frame, so it is excluded). Ground truth was used only for this score.
          </span>
        </p>
      ) : (
        <p className="mt-10 text-mute">The held-out score appears here once results.json exists.</p>
      )}
    </section>
  )
}

function Pipeline() {
  return (
    <nav aria-label="Pipeline" className="border-y border-steel bg-white/50">
      <ol className="mx-auto grid max-w-7xl grid-cols-2 gap-px px-6 sm:grid-cols-4 lg:grid-cols-7">
        {STAGES.map((s, i) => (
          <li key={s.title}>
            <a href={`#${s.id}`} className="block h-full py-5 pr-4 hover:text-drape-2">
              <span className="text-sm text-mute">{i + 1}</span>
              <span className="block font-bold">{s.title}</span>
              <span className="block text-sm leading-snug text-mute">{s.body}</span>
            </a>
          </li>
        ))}
      </ol>
    </nav>
  )
}

function Monitor({ labels, testVideos }: { labels: { source: string | null; has_status: boolean; rows: LabelRow[] }; testVideos: string[] }) {
  const frames = useMemo(() => {
    const ids = [...new Set(labels.rows.map((r) => r.frame_id))]
    const isTest = (f: string) => testVideos.includes(f.split('_')[0])
    return ids.sort((a, b) => Number(isTest(b)) - Number(isTest(a)) || a.localeCompare(b, undefined, { numeric: true }))
  }, [labels, testVideos])
  const interesting = useMemo(() => {
    const amb = labels.rows.find((r) => normStatus(r.status) !== 'accept' && testVideos.includes(r.video))
    return amb?.frame_id ?? frames[0]
  }, [labels, frames, testVideos])
  const [fid, setFid] = useState<string | undefined>(undefined)
  const current = fid ?? interesting
  const rows = labels.rows.filter((r) => r.frame_id === current)
  const [focus, setFocus] = useState<string | null>(null)
  const idx = frames.indexOf(current ?? '')

  if (!frames.length) return <Section id="monitor" title="Real frames, labelled and checked"><Pending what="labels from the labeller" /></Section>

  return (
    <Section id="monitor" title="Real frames, labelled and checked"
      lead={`Every structure gets a label, a box and a second opinion. ${labels.has_status ? 'Box colour is the final verdict.' : 'Not yet verified: boxes are grey.'} Source: ${labels.source}.`}>
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="rounded-2xl bg-drape p-3 lg:sticky lg:top-6 shadow-[0_30px_60px_-30px_rgba(14,59,58,.6)]">
          <div className="relative overflow-hidden rounded-xl bg-black">
            <img src={`/api/frame/${current}.jpg`} alt={`Laparoscopic frame ${current}`} className="block w-full" />
            <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
              {rows.filter((r) => (r.final_present ?? r.present) === 1 && r.box).map((r) => {
                const [y0, x0, y1, x1] = r.box as number[]
                const st = normStatus(r.status)
                const col = st ? STATUS_COLOR[st] : '#C9D6D2'
                const dim = focus && focus !== r.key
                return <rect key={r.key} x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke={col}
                  strokeWidth={focus === r.key ? 7 : 4} vectorEffect="non-scaling-stroke" opacity={dim ? 0.25 : 1} />
              })}
            </svg>
          </div>
          <div className="flex items-center justify-between gap-3 px-1 pt-3 text-wash">
            <button className="rounded-md px-3 py-1 hover:bg-drape-2 disabled:opacity-30" disabled={idx <= 0} onClick={() => setFid(frames[idx - 1])}>Previous frame</button>
            <label className="flex items-center gap-2 text-sm">
              <span className="sr-only">Frame</span>
              <select value={current} onChange={(e) => setFid(e.target.value)} className="rounded-md bg-drape-2 px-2 py-1">
                {frames.map((f) => <option key={f} value={f}>{f}{testVideos.includes(f.split('_')[0]) ? ' (held-out test)' : ''}</option>)}
              </select>
            </label>
            <button className="rounded-md px-3 py-1 hover:bg-drape-2 disabled:opacity-30" disabled={idx >= frames.length - 1} onClick={() => setFid(frames[idx + 1])}>Next frame</button>
          </div>
        </div>

        <ul className="divide-y divide-steel" onMouseLeave={() => setFocus(null)}>
          {rows.map((r) => {
            const st = normStatus(r.status)
            const present = (r.final_present ?? r.present) === 1
            return (
              <li key={r.key} className="py-3" onMouseEnter={() => setFocus(r.key)}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-lg font-bold">{KEY_NAME[r.key] ?? r.key} <span className="font-normal text-mute">{present ? 'present' : 'absent'}</span></span>
                  {st && <span className="shrink-0 rounded-full px-2.5 py-0.5 text-sm font-bold text-white" style={{ background: STATUS_COLOR[st] }}>{STATUS_WORD[st]}</span>}
                </div>
                {r.evidence && <p className="mt-1 text-sm text-drape-2">“{r.evidence}”{typeof r.conf === 'number' && <span className="text-mute"> confidence {r.conf.toFixed(2)}</span>}</p>}
                <p className="mt-1 text-sm text-mute">
                  {r.second_pass_agree === true && 'Second pass agreed. '}
                  {r.second_pass_agree === false && 'Second pass disagreed. '}
                  {typeof r.resolved_round === 'number' && r.resolved_round > 0 && `Settled by Argon in round ${r.resolved_round}. `}
                  {r.final_by === 'argon' && r.argon_reason ? r.argon_reason.slice(0, 220) + (r.argon_reason.length > 220 ? '…' : '') : ''}
                </p>
              </li>
            )
          })}
        </ul>
      </div>
    </Section>
  )
}

function Teach({ ov }: { ov: Overview }) {
  const m = ov.manual
  const [open, setOpen] = useState<string | null>(null)
  if (!m) return <Section id="teach" title="What it taught itself"><Pending what="manual.json" /></Section>
  return (
    <Section id="teach" title="What it taught itself" lead={`A manual for ${m.procedure ?? 'the procedure'}, written by the agent from ${m.sources?.length ?? 0} public sources. No human edited it.`}>
      <div className="grid gap-10 lg:grid-cols-2">
        <div>
          <h3 className="text-xl font-bold">How it tells structures apart</h3>
          <ul className="mt-3 divide-y divide-steel">
            {(m.structures ?? []).map((s) => (
              <li key={s.key}>
                <button className="flex w-full items-baseline justify-between py-3 text-left" aria-expanded={open === s.key} onClick={() => setOpen(open === s.key ? null : s.key)}>
                  <span className="font-bold">{s.name ?? s.key}</span>
                  <span className="text-sm text-mute">{open === s.key ? 'Hide' : 'Show'}</span>
                </button>
                {open === s.key && (
                  <dl className="grid gap-2 pb-4 text-sm leading-relaxed">
                    <dt className="font-bold">Looks like</dt><dd className="text-drape-2">{s.appearance}</dd>
                    <dt className="font-bold">Often confused with</dt><dd className="text-drape-2">{s.confused_with?.join(', ')}</dd>
                    <dt className="font-bold">How to tell</dt><dd className="text-drape-2">{s.how_to_tell_apart}</dd>
                  </dl>
                )}
              </li>
            ))}
          </ul>
          <h3 className="mt-10 text-xl font-bold">Phases it expects</h3>
          <ol className="mt-3 list-decimal space-y-1 pl-5 text-drape-2">
            {(m.phases ?? []).map((p) => <li key={p}>{p.replace(/^\d+\.\s*/, '').split(':')[0]}</li>)}
          </ol>
        </div>
        <div>
          <h3 className="text-xl font-bold">Where it learned it</h3>
          <ul className="mt-3 space-y-4">
            {(m.sources ?? []).map((s) => (
              <li key={s.url ?? s.title}>
                <a href={s.url} target="_blank" rel="noreferrer" className="font-bold underline decoration-steel underline-offset-4 hover:decoration-drape">{s.title}</a>
                {s.real_intraop_footage_note && <p className="mt-1 text-sm text-mute">{s.real_intraop_footage_note}</p>}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Section>
  )
}

function AmberLoop({ ov }: { ov: Overview }) {
  const loop = ov.results?.amber_loop
  const rounds = loop?.per_round_test
  if (!rounds?.length) return <Section id="amber" title="The unsure pile shrinks"><Pending what="amber loop results" /></Section>
  const max = Math.max(...rounds.map((r) => r.ambers_remaining), 1)
  const last = rounds[rounds.length - 1]
  return (
    <Section id="amber" title="The unsure pile shrinks"
      lead={`Labels the double-check could not settle go back to Gemini 4 Argon, up to ${loop?.max_rounds ?? rounds.length - 1} times. Only what is still unsure reaches a person.`}>
      <div className="space-y-3">
        {rounds.map((r) => (
          <div key={r.round} className="grid grid-cols-[9rem_1fr] items-center gap-4">
            <span className="text-sm text-mute">{r.round === 0 ? 'After double-check' : `After Argon round ${r.round}`}</span>
            <div className="flex items-center gap-3">
              <div className="h-9 rounded-md bg-amber" style={{ width: `${(r.ambers_remaining / max) * 100}%`, minWidth: 6 }} />
              <span className="text-lg font-bold">{r.ambers_remaining}</span>
              <span className="text-sm text-mute">{r.accepted_count} accepted, {r.rejected_count} rejected</span>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-6 max-w-[62ch] text-lg">
        {last.ambers_remaining} test label{last.ambers_remaining === 1 ? '' : 's'} left for the human final pass. Nothing is released as a dataset until a person signs off
        those plus a random audit sample of accepted labels.
        {ov.n_verified_labels !== null && <span className="text-mute"> Verified dataset so far: {ov.n_verified_labels} labels.</span>}
      </p>
    </Section>
  )
}

type Series = { name: string; points: [number, number][]; color: string; dashed?: boolean }

function LineChart({ series, xLabel, yLabel, yMin = 0, xMax = 1 }: { series: Series[]; xLabel: string; yLabel: string; yMin?: number; xMax?: number }) {
  const W = 640, H = 340, L = 52, B = 40, T = 12, R = 16
  const x = (v: number) => L + (v / xMax) * (W - L - R)
  const y = (v: number) => T + (1 - (v - yMin) / (1 - yMin)) * (H - T - B)
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  const yt = [yMin, yMin + (1 - yMin) / 2, 1]
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${yLabel} against ${xLabel}`}>
        {yt.map((t) => <g key={t}><line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--color-steel)" /><text x={L - 8} y={y(t) + 4} textAnchor="end" fontSize="12" fill="var(--color-mute)">{Math.round(t * 100)}%</text></g>)}
        {ticks.map((t) => <text key={t} x={x(t * xMax)} y={H - 18} textAnchor="middle" fontSize="12" fill="var(--color-mute)">{Math.round(t * xMax * 100)}%</text>)}
        <text x={(L + W - R) / 2} y={H - 2} textAnchor="middle" fontSize="12" fill="var(--color-mute)">{xLabel}</text>
        {series.map((s) => (
          <polyline key={s.name} fill="none" stroke={s.color} strokeWidth={s.dashed ? 2 : 3} strokeDasharray={s.dashed ? '6 5' : undefined}
            points={s.points.map(([a, b]) => `${x(a)},${y(Math.max(b, yMin))}`).join(' ')} />
        ))}
      </svg>
      <figcaption className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-sm">
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-2"><svg width="22" height="6"><line x1="0" x2="22" y1="3" y2="3" stroke={s.color} strokeWidth="3" strokeDasharray={s.dashed ? '5 4' : undefined} /></svg>{s.name}</span>
        ))}
      </figcaption>
    </figure>
  )
}

function Proof({ ov }: { ov: Overview }) {
  const r = ov.results
  if (!r) return <Section id="proof" title="Proof on held-out video"><Pending what="results.json" /></Section>
  const series: Series[] = [
    r.system_zero_label?.curve && { name: 'SurgProof checks (no human labels)', points: r.system_zero_label.curve, color: 'var(--color-drape)' },
    r.single_conf?.curve && { name: 'Labeller confidence alone', points: r.single_conf.curve, color: '#6E9C97' },
    r.blank_single_conf?.curve && { name: 'No manual (blank prompt)', points: r.blank_single_conf.curve, color: 'var(--color-reject)' },
    r.majority_vote?.curve && { name: 'Majority vote', points: r.majority_vote.curve, color: 'var(--color-amber)' },
    r.upper_bound_calibrated?.curve && { name: 'Calibrated on human labels (upper bound)', points: r.upper_bound_calibrated.curve, color: 'var(--color-mute)', dashed: true },
  ].filter(Boolean) as Series[]
  const perLabel = Object.entries(r.per_label_accepted_acc ?? {})
  return (
    <Section id="proof" title="Proof on held-out video"
      lead="Six CholecSeg8k videos were held out from everything. The agent never saw their labels; we use them only to score. Higher line = more accurate at the same share of labels kept.">
      <div className="grid gap-10 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        {series.length ? <LineChart series={series} xLabel="Share of labels kept" yLabel="Accuracy of kept labels" yMin={0.8} /> : <Pending what="accuracy curves" />}
        <div>
          <table className="w-full text-left">
            <caption className="pb-2 text-left font-bold">Raw label accuracy, before any checking (all 6 structures)</caption>
            <tbody className="divide-y divide-steel">
              <tr><td className="py-2">Blank prompt, no manual</td><td className="py-2 text-right font-bold">{pct(r.blank_raw_acc)}</td></tr>
              <tr><td className="py-2">With its self-written manual</td><td className="py-2 text-right font-bold">{pct(r.manual_raw_acc)}</td></tr>
              <tr><td className="py-2">Kept after its own checks</td><td className="py-2 text-right font-bold text-accept">{pct(r.accepted?.accuracy)}</td></tr>
            </tbody>
          </table>
          {perLabel.length > 0 && (
            <table className="mt-8 w-full text-left">
              <caption className="pb-2 text-left font-bold">Accuracy of accepted labels, per structure</caption>
              <tbody className="divide-y divide-steel">
                {perLabel.map(([k, v]) => <tr key={k}><td className="py-1.5">{KEY_NAME[k] ?? k}</td><td className="py-1.5 text-right">{pct(v)}</td></tr>)}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </Section>
  )
}

function StudentSection({ ov }: { ov: Overview }) {
  const s = ov.student
  const acc = s?.mean_acc_by_source
  if (!acc) return <Section id="student" title="A specialist trained only on verified labels"><Pending what="student.json" /></Section>
  const rows: { k: string; name: string; upper?: boolean }[] = [
    { k: 'single', name: 'Trained on raw labels' },
    { k: 'vote', name: 'Trained on majority-vote labels' },
    { k: 'verified', name: 'Trained on verified labels only' },
    { k: 'oracle_upper_bound', name: 'Trained on human labels (upper bound)', upper: true },
  ].filter((r) => typeof acc[r.k] === 'number')
  const lo = Math.max(0, Math.min(...rows.map((r) => acc[r.k])) - 0.1)
  return (
    <Section id="student" title="A specialist trained only on verified labels"
      lead={`A small, fast model (DINOv2 features + linear probe) learns from the agent's labels, then is scored on ${s?.n_test_frames ?? '?'} frames from ${s?.n_test_videos ?? '?'} unseen videos. Mean accuracy over ${s?.headline_labels?.length ?? 5} informative structures.`}>
      <div className="max-w-3xl space-y-4">
        {rows.map((r) => (
          <div key={r.k} className="grid grid-cols-[minmax(0,15rem)_1fr] items-center gap-4">
            <span className={r.k === 'verified' ? 'font-bold' : 'text-drape-2'}>{r.name}</span>
            <div className="flex items-center gap-3">
              <div className="h-8 rounded-md" style={{
                width: `${((acc[r.k] - lo) / (1 - lo)) * 100}%`,
                background: r.upper ? 'repeating-linear-gradient(135deg, var(--color-steel) 0 6px, transparent 6px 12px)' : r.k === 'verified' ? 'var(--color-accept)' : 'var(--color-drape-2)',
                border: r.upper ? '2px dashed var(--color-mute)' : undefined,
              }} />
              <span className="font-bold">{pct(acc[r.k])}</span>
              {s?.mean_n_train_by_source?.[r.k] && <span className="text-sm text-mute">{Math.round(s.mean_n_train_by_source[r.k])} training frames</span>}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-6 max-w-[62ch] text-sm text-mute">Bars start at {Math.round(lo * 100)}%. The human-label model is shown for reference only; it is not part of the zero-human-label claim.</p>
    </Section>
  )
}

function Tracker({ ov }: { ov: Overview }) {
  const t = ov.tracker
  if (!t.budget_curve) return null
  const keys = Object.keys(t.budget_curve[0] ?? {}).filter((k) => k !== 'budget')
  const colors = ['var(--color-drape)', 'var(--color-reject)', 'var(--color-amber)']
  const series: Series[] = keys.map((k, i) => ({ name: k, points: t.budget_curve!.map((r) => [r.budget, r[k]] as [number, number]), color: colors[i % 3], dashed: k.startsWith('random') }))
  const maxB = Math.max(...t.budget_curve.map((r) => r.budget))
  return (
    <Section id="tracker" title="Motion tells it where to look again"
      lead="Instruments move smoothly. A tracker follows each grasper and L-hook through the clip and flags labels that jump, vanish or swap, so rechecks go where errors are likely.">
      {t.source !== 'gemini' && (
        <p className="mb-6 max-w-[70ch] rounded-lg bg-amber/15 p-4">
          Tested on ground-truth boxes with deliberately injected errors, not on Gemini's labels yet. It is a development result, not part of the headline.
        </p>
      )}
      <div className="max-w-3xl">
        <LineChart series={series} xMax={maxB} xLabel="Share of labels rechecked" yLabel="Share of injected errors found" />
      </div>
    </Section>
  )
}

function Narration({ claims }: { claims: { source: string | null; claims: Claim[] } | null }) {
  const all = claims?.claims ?? []
  const clips = [...new Set(all.map((c) => c.clip_id))]
  const [clip, setClip] = useState<string | null>(null)
  const [mode, setMode] = useState<string>('grounded')
  const cur = clip ?? clips[0]
  const modes = [...new Set(all.map((c) => c.mode ?? 'grounded'))]
  const list = all.filter((c) => c.clip_id === cur && (c.mode ?? 'grounded') === mode).sort((a, b) => (a.t_start ?? 0) - (b.t_start ?? 0))
  const tagColor = { verified: 'var(--color-accept)', unsupported: 'var(--color-amber)', contradicted: 'var(--color-reject)' }
  return (
    <Section id="narrate" title="It only says what it can stand behind"
      lead="Every sentence is broken into claims and each claim is checked against the verified labels. The narration keeps the verified ones; the rest are shown here, struck out.">
      {!all.length ? (
        <Pending what={claims?.source === 'mock' ? 'narration on real Gemini labels (current output uses mock labels and is hidden)' : 'the narrator'} />
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-4">
            <select value={cur} onChange={(e) => setClip(e.target.value)} className="rounded-md border border-steel bg-white px-3 py-1.5" aria-label="Clip">
              {clips.map((c) => <option key={c}>{c}</option>)}
            </select>
            {modes.length > 1 && (
              <div role="group" aria-label="Narration mode" className="flex rounded-md border border-steel bg-white p-0.5">
                {modes.map((m) => <button key={m} onClick={() => setMode(m)} aria-pressed={mode === m} className={`rounded px-3 py-1 ${mode === m ? 'bg-drape text-wash' : ''}`}>{m === 'free' ? 'Unchecked' : 'Grounded'}</button>)}
              </div>
            )}
          </div>
          <p className="mt-6 max-w-[62ch] text-2xl leading-relaxed">
            {list.filter((c) => c.tag === 'verified').map((c) => c.text).join(' ') || <span className="text-mute">No verified claims in this clip.</span>}
          </p>
          <ul className="mt-8 max-w-4xl divide-y divide-steel">
            {list.map((c) => (
              <li key={c.claim_id} className="flex items-baseline gap-4 py-2.5">
                <span className="w-28 shrink-0 text-sm font-bold" style={{ color: tagColor[c.tag] }}>{c.tag}</span>
                <span className={c.tag === 'verified' ? '' : 'text-mute line-through decoration-1'}>{c.text}</span>
                <span className="ml-auto shrink-0 text-sm text-mute">frames {c.t_start}–{c.t_end}{c.failed_layers.length ? `, failed ${c.failed_layers.join(', ')}` : ''}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </Section>
  )
}

function Atlas({ ov }: { ov: Overview }) {
  if (!ov.atlas.length) return null
  return (
    <Section id="atlas" title="Any surgery: a showcase" lead="The same agent on public video of other procedures. There is no ground truth here, so nothing is scored.">
      <ul className="divide-y divide-steel">
        {ov.atlas.map((a) => (
          <li key={a.slug} className="flex flex-wrap items-baseline justify-between gap-4 py-3">
            <span className="text-lg font-bold">{a.procedure ?? a.slug}</span>
            <span className="text-mute">{a.timeline?.phases?.length ?? 0} phases found, {a.human_labels_used ?? 0} human labels used</span>
          </li>
        ))}
      </ul>
    </Section>
  )
}

function Footer() {
  return (
    <footer className="mt-10 bg-drape text-wash">
      <div className="mx-auto grid max-w-7xl gap-10 px-6 py-14 lg:grid-cols-2">
        <div>
          <h2 className="text-xl font-bold">What we claim, and what we don't</h2>
          <ul className="mt-4 max-w-[62ch] list-disc space-y-2 pl-5 text-steel">
            <li>No human labels are used to train anything we report as ours.</li>
            <li>Ground truth is used only to score the six held-out test videos.</li>
            <li>Anything that used human labels (the calibration split, the human-label student) is marked as an upper bound.</li>
            <li>“Any surgery” is a showcase without ground truth, not proof.</li>
          </ul>
        </div>
        <div>
          <h2 className="text-xl font-bold">Built on</h2>
          <p className="mt-4 max-w-[62ch] text-steel">
            SurgVLP (learning from surgical lectures, CAMMA Strasbourg), SurgSLOT (self-evolving data engine with human verification),
            Med-Gemini (zero-shot critical view of safety), and the CholecSeg8k, Endoscapes2023 and CholecT50 datasets. Our step: the agent
            replaces the human verifier, and verified labels train a better specialist.
          </p>
        </div>
      </div>
    </footer>
  )
}
