import { Fragment } from 'react'
import { COLOR, KEYS, KIND, LAYER_NAME, NAME, ST_COLOR, ST_WORD, V_COLOR, V_WORD, isPresent, mean, pct, st, why, type ClaimRow, type ClipData, type Label } from './clip'

export default function Report({ data, onWatch, next }: { data: ClipData; onWatch: () => void; next?: () => void }) {
  const L = data.labels
  const acc = L.filter((l) => st(l.status) === 'accept')
  const gtScored = L.filter((l) => typeof l.gt_correct === 'boolean')
  const gtAcc = acc.filter((l) => typeof l.gt_correct === 'boolean')
  const grounded = data.claims.filter((c) => c.mode === 'grounded')
  const free = data.claims.filter((c) => c.mode === 'free')
  const transcript = grounded.filter((c) => c.verdict !== 'drop').sort((a, b) => a.t_start - b.t_start)
  const frames = [...new Set(L.map((l) => l.frame))].sort((a, b) => a - b)

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-bold uppercase tracking-widest text-mute">Surgery report · laparoscopic cholecystectomy</p>
          <h1 className="mt-1 text-4xl font-bold tracking-tight">{data.video} <span className="text-mute">clip {data.frames[0]}–{data.frames[data.frames.length - 1]}</span></h1>
          <p className="mt-1 text-mute">Held-out test video. Labels and narration made by the agent with no human labels. Ground truth used only for the green check column.</p>
        </div>
        <div className="flex gap-2">
          <button onClick={onWatch} className="rounded-xl bg-panel px-4 py-2.5 font-bold ring-1 ring-line hover:bg-panel-2">← Watch again</button>
          {next && <button onClick={next} className="rounded-xl bg-accept px-4 py-2.5 font-bold text-black hover:brightness-110">Next video →</button>}
        </div>
      </header>

      {/* KPI tiles */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Ring value={acc.length / Math.max(1, L.length)} color={ST_COLOR.accept} title="Labels verified" sub={`${acc.length} of ${L.length} labels on ${frames.length} frames passed every check`} />
        <Ring value={mean(L.map((l) => l.conf ?? 0)) ?? 0} color="#38BDF8" title="Mean model confidence" sub="Gemini Flash's own confidence, before checking" />
        <Ring value={gtAcc.length ? gtAcc.filter((l) => l.gt_correct).length / gtAcc.length : null} color="#A3E635" title="Verified labels correct"
          sub={gtAcc.length ? `${gtAcc.filter((l) => l.gt_correct).length}/${gtAcc.length} match the held-out ground truth (scoring only)` : 'No ground truth for this clip'} />
        <Ring value={grounded.length ? grounded.filter((c) => c.verdict !== 'drop').length / grounded.length : null} color={V_COLOR.speak} title="Narration kept"
          sub={`${grounded.filter((c) => c.verdict !== 'drop').length} of ${grounded.length} checked claims spoken; raw model: ${free.filter((c) => c.verdict === 'drop').length} of ${free.length} withheld`} />
      </div>

      {/* What was seen */}
      <Card title="What was seen" lead="One card per structure. Confidence is the labeller's; trust combines it with the second pass and the critic.">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {KEYS.map((k) => <StructureCard key={k} k={k} clip={data.clip_id} rows={L.filter((l) => l.key === k).sort((a, b) => a.frame - b.frame)} />)}
        </div>
      </Card>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Card title="Confidence map" lead="Every label in this clip. Fill = present, outline = absent. Number = confidence. Colour of the ring = verdict.">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead><tr><th className="pb-2 text-left font-normal text-mute">Structure</th>{frames.map((f) => <th key={f} className="pb-2 font-normal text-mute">frame {f}</th>)}</tr></thead>
              <tbody>
                {KEYS.map((k) => (
                  <tr key={k}>
                    <td className="py-1.5 pr-3"><span className="flex items-center gap-2 font-bold"><span className="h-2.5 w-2.5 rounded-full" style={{ background: COLOR[k] }} />{NAME[k]}</span></td>
                    {frames.map((f) => {
                      const l = L.find((x) => x.key === k && x.frame === f)
                      if (!l) return <td key={f} className="text-center text-mute">·</td>
                      const s = st(l.status)
                      return (
                        <td key={f} className="p-1 text-center">
                          <span title={`${ST_WORD[s]}${l.evidence ? ` · ${l.evidence}` : ''}`} className="mx-auto flex h-11 w-full min-w-16 items-center justify-center gap-1 rounded-lg text-sm font-bold"
                            style={{ background: isPresent(l) ? `${COLOR[k]}${Math.round(25 + (l.conf ?? 0) * 60).toString(16)}` : 'transparent', boxShadow: `inset 0 0 0 2px ${ST_COLOR[s]}` }}>
                            {typeof l.conf === 'number' ? Math.round(l.conf * 100) : '–'}
                            {l.gt_correct === true && <span className="text-xs text-accept">✓</span>}
                            {l.gt_correct === false && <span className="text-xs text-reject">✗</span>}
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Legend />
          <p className="mt-2 text-xs text-mute">✓ / ✗ = agrees / disagrees with the held-out ground-truth mask ({gtScored.length} labels scored). The agent never saw it.</p>
        </Card>

        <Card title="Narration of the surgery" lead="Only what survived the checker. Cautious lines are said with a hedge.">
          <ol className="space-y-3">
            {transcript.length === 0 && <li className="text-mute">Nothing could be stated with evidence on this clip.</li>}
            {transcript.map((c) => (
              <li key={c.claim_id} className="flex gap-3">
                <span className="w-16 shrink-0 pt-1 text-right text-xs text-mute">f{c.t_start}</span>
                <p className="text-lg leading-snug" style={{ borderLeft: `3px solid ${V_COLOR[c.verdict]}`, paddingLeft: 12 }}>
                  {c.text}{c.verdict === 'hedge' && <span className="ml-2 rounded bg-amber px-1.5 align-middle text-xs font-bold text-black">cautious</span>}
                </p>
              </li>
            ))}
          </ol>
          <div className="mt-6 space-y-3">
            <Funnel name="Checked narration" claims={grounded} />
            <Funnel name="Raw model (unchecked)" claims={free} />
          </div>
        </Card>
      </div>

      <Card title="How each claim was checked" lead="Rows = claims, columns = independent checks. Any failed check withholds the claim. Hover a cell for the reason.">
        <ClaimMatrix claims={[...grounded, ...free]} />
      </Card>
    </div>
  )
}

function Card({ title, lead, children }: { title: string; lead?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl bg-panel p-5 ring-1 ring-line">
      <h2 className="text-xl font-bold">{title}</h2>
      {lead && <p className="mt-0.5 text-sm text-mute">{lead}</p>}
      <div className="mt-4">{children}</div>
    </section>
  )
}

function Ring({ value, color, title, sub }: { value: number | null; color: string; title: string; sub: string }) {
  const R = 34, C = 2 * Math.PI * R
  return (
    <div className="flex items-center gap-4 rounded-2xl bg-panel p-4 ring-1 ring-line">
      <svg width="88" height="88" viewBox="0 0 88 88" className="shrink-0" role="img" aria-label={`${title}: ${pct(value)}`}>
        <circle cx="44" cy="44" r={R} fill="none" stroke="var(--color-panel-2)" strokeWidth="9" />
        {value !== null && <circle cx="44" cy="44" r={R} fill="none" stroke={color} strokeWidth="9" strokeLinecap="round" strokeDasharray={`${C * value} ${C}`} transform="rotate(-90 44 44)" style={{ transition: 'stroke-dasharray .8s' }} />}
        <text x="44" y="50" textAnchor="middle" fontSize="19" fontWeight="700" fill="var(--color-text)">{value === null ? '–' : pct(value)}</text>
      </svg>
      <div><p className="font-bold">{title}</p><p className="text-xs leading-snug text-mute">{sub}</p></div>
    </div>
  )
}

function StructureCard({ k, clip, rows }: { k: string; clip: string; rows: Label[] }) {
  if (!rows.length) return null
  const pres = rows.filter(isPresent)
  const best = [...pres].filter((r) => r.box).sort((a, b) => Number(st(b.status) === 'accept') - Number(st(a.status) === 'accept') || (b.conf ?? 0) - (a.conf ?? 0))[0]
  const allOk = rows.every((r) => st(r.status) === 'accept')
  const anyReject = rows.some((r) => st(r.status) === 'reject')
  const verdict = pres.length === 0 ? 'Not seen' : allOk ? 'Present, verified' : anyReject ? 'Disputed' : 'Present, unsure'
  const vc = pres.length === 0 && allOk ? 'var(--color-mute)' : allOk ? ST_COLOR.accept : anyReject ? ST_COLOR.reject : ST_COLOR.amber
  const conf = mean(rows.map((r) => r.conf ?? 0)), trust = mean(rows.map((r) => r.trust ?? 0))
  const agree = rows.filter((r) => r.second_pass_agree === true).length
  const argon = rows.filter((r) => (r.resolved_round ?? 0) > 0)
  const unsure = rows.find((r) => st(r.status) !== 'accept' || r.argon_reason)
  return (
    <div className="overflow-hidden rounded-xl bg-ink/60 ring-1 ring-line">
      <div className="relative h-32 bg-black">
        {best?.box ? <img src={`/api/crop/${clip}_${best.frame}/${best.box.map((v) => Math.round(v)).join('/')}.jpg`} alt={`${NAME[k]} crop, frame ${best.frame}`} className="h-full w-full object-cover" />
          : <div className="grid h-full place-items-center text-sm text-mute">no box: not in view</div>}
        <span className="absolute left-2 top-2 rounded-md px-2 py-0.5 text-xs font-bold text-black" style={{ background: COLOR[k] }}>{NAME[k]} <span className="font-normal">{KIND[k]}</span></span>
        <span className="absolute right-2 top-2 rounded-full px-2 py-0.5 text-xs font-bold text-black" style={{ background: vc }}>{verdict}</span>
      </div>
      <div className="space-y-2.5 p-3 text-sm">
        <Meter label="Confidence" v={conf} color="#38BDF8" />
        <Meter label="Trust after checks" v={trust} color={vc} />
        <div className="flex items-center justify-between text-xs">
          <span className="text-mute">Per labelled frame</span>
          <span className="flex gap-1">{rows.map((r) => <span key={r.label_id} title={`frame ${r.frame}: ${isPresent(r) ? 'present' : 'absent'}, ${ST_WORD[st(r.status)]}`} className="h-3.5 w-3.5 rounded-full" style={{ background: isPresent(r) ? ST_COLOR[st(r.status)] : 'transparent', border: `2px solid ${ST_COLOR[st(r.status)]}` }} />)}</span>
        </div>
        <p className="text-xs text-mute">Second pass agreed {agree}/{rows.length}{argon.length ? ` · Argon settled ${argon.length} in round ${Math.max(...argon.map((r) => r.resolved_round ?? 0))}` : ''}</p>
        {unsure && (unsure.argon_reason || unsure.critic) && <p className="line-clamp-3 border-l-2 border-line pl-2 text-xs italic text-mute">{unsure.argon_reason || unsure.critic}</p>}
      </div>
    </div>
  )
}

function Meter({ label, v, color }: { label: string; v: number | null; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs"><span className="text-mute">{label}</span><b>{pct(v)}</b></div>
      <div className="mt-1 h-2 overflow-hidden rounded-full bg-panel-2"><div className="h-full rounded-full" style={{ width: `${(v ?? 0) * 100}%`, background: color }} /></div>
    </div>
  )
}

function Legend() {
  return (
    <div className="mt-3 flex flex-wrap gap-4 text-xs text-mute">
      {(['accept', 'amber', 'reject'] as const).map((s) => <span key={s} className="flex items-center gap-1.5"><span className="h-3 w-3 rounded" style={{ boxShadow: `inset 0 0 0 2px ${ST_COLOR[s]}` }} />{ST_WORD[s]}</span>)}
    </div>
  )
}

function Funnel({ name, claims }: { name: string; claims: ClaimRow[] }) {
  const n = (v: string) => claims.filter((c) => c.verdict === v).length
  return (
    <div>
      <div className="flex justify-between text-xs"><span className="font-bold">{name}</span><span className="text-mute">{claims.length} claims</span></div>
      <div className="mt-1 flex h-7 overflow-hidden rounded-lg bg-ink text-xs font-bold text-black">
        {(['speak', 'hedge', 'drop'] as const).map((v) => n(v) > 0 && (
          <span key={v} className="flex items-center justify-center" style={{ width: `${(n(v) / claims.length) * 100}%`, background: V_COLOR[v] }} title={`${n(v)} ${V_WORD[v]}`}>{n(v)}</span>
        ))}
      </div>
      <div className="mt-1 flex gap-3 text-[11px] text-mute">{(['speak', 'hedge', 'drop'] as const).map((v) => <span key={v}><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: V_COLOR[v] }} />{V_WORD[v]}</span>)}</div>
    </div>
  )
}

const CELL: Record<string, string> = { pass: '#22C55E', fail: '#EF4444', abstain: '#1E4148', hedge: '#F59E0B' }

function ClaimMatrix({ claims }: { claims: ClaimRow[] }) {
  const layers = ['L1', 'L2', 'L3', 'L4', 'P', 'L5', 'L5c', 'L6', 'L7'].filter((l) => claims.some((c) => c.layers[l] !== undefined))
  const order = { speak: 0, hedge: 1, drop: 2 }
  const rows = [...claims].sort((a, b) => (a.mode === b.mode ? 0 : a.mode === 'grounded' ? -1 : 1) || order[a.verdict] - order[b.verdict] || a.t_start - b.t_start)
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-mute">
            <th className="pb-2 text-left font-normal">Claim</th>
            {layers.map((l) => <th key={l} className="w-10 pb-2 font-normal" title={LAYER_NAME[l]}><span className="block h-24 whitespace-nowrap [writing-mode:vertical-rl] rotate-180 text-left">{LAYER_NAME[l]}</span></th>)}
            <th className="pb-2 pl-3 text-left font-normal">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c, i) => {
            const r = typeof c.reasons === 'object' ? c.reasons : {}
            const first = i === 0 || rows[i - 1].mode !== c.mode
            return (
              <Fragment key={c.claim_id}>
                {first && <tr key={`h-${c.mode}`}><td colSpan={layers.length + 2} className="pb-1 pt-4 text-xs font-bold uppercase tracking-widest text-mute">{c.mode === 'grounded' ? 'Checked narration' : 'Raw model narration'}</td></tr>}
                <tr key={c.claim_id} className="border-t border-line/60">
                  <td className="max-w-[520px] py-1.5 pr-3">
                    <span className={c.verdict === 'drop' ? 'text-mute line-through decoration-reject/60' : ''}>{c.text}</span>
                    {c.gt_correct === true && <span className="ml-2 text-xs text-accept">✓ ground truth</span>}
                    {c.gt_correct === false && <span className="ml-2 text-xs text-reject">✗ ground truth</span>}
                  </td>
                  {layers.map((l) => (
                    <td key={l} className="p-0.5"><span title={`${LAYER_NAME[l]}: ${c.layers[l] ?? '–'}${r[l] ? ` · ${r[l]}` : ''}`} className="block h-6 rounded" style={{ background: CELL[c.layers[l] ?? 'abstain'] ?? '#1E4148' }} /></td>
                  ))}
                  <td className="py-1.5 pl-3">
                    <span className="whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-bold text-black" style={{ background: V_COLOR[c.verdict] }}>{V_WORD[c.verdict]}</span>
                    <span className="mt-0.5 block max-w-[260px] truncate text-[11px] text-mute" title={why(c)}>{why(c)}</span>
                  </td>
                </tr>
              </Fragment>
            )
          })}
        </tbody>
      </table>
      <div className="mt-3 flex gap-4 text-xs text-mute">
        {(['pass', 'fail', 'abstain'] as const).map((k) => <span key={k} className="flex items-center gap-1.5"><span className="h-3 w-5 rounded" style={{ background: CELL[k] }} />{k === 'abstain' ? 'did not apply' : k}</span>)}
      </div>
    </div>
  )
}
