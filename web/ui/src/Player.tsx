import { useEffect, useMemo, useRef, useState } from 'react'
import { COLOR, KEYS, NAME, ST_COLOR, ST_WORD, V_COLOR, V_WORD, isPresent, st, why, type ClaimRow, type ClipData, type Label } from './clip'

const FPS = 10

export default function Player({ data, onReport }: { data: ClipData; onReport: () => void }) {
  const labelFrames = useMemo(() => [...new Set(data.labels.map((l) => l.frame))].sort((a, b) => a - b), [data])
  const stride = useMemo(() => {
    const d = labelFrames.slice(1).map((f, i) => f - labelFrames[i])
    return d.length ? Math.min(...d) : 8
  }, [labelFrames])
  // Play the stretch that has labels or narration, not 50 empty frames after it.
  const range = useMemo(() => {
    const f = data.frames
    const ends = [...labelFrames.map((x) => x + stride), ...data.claims.map((c) => c.t_end + 4)]
    const hi = Math.min(f[f.length - 1], Math.max(f[0] + 24, ...ends))
    return f.filter((x) => x <= hi)
  }, [data, labelFrames, stride])

  const [i, setI] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [mode, setMode] = useState<'grounded' | 'free'>('grounded')
  const [hover, setHover] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(0)
  const done = i >= range.length - 1
  const f = range[Math.min(i, range.length - 1)]

  // Preload every frame so playback is smooth.
  useEffect(() => {
    setI(0); setPlaying(true); setLoaded(0)
    let n = 0
    range.forEach((fr) => { const im = new Image(); im.onload = im.onerror = () => setLoaded(++n); im.src = `/api/frame/${data.clip_id}_${fr}.jpg` })
  }, [data.clip_id, range])
  const ready = loaded >= Math.min(range.length, 30)

  useEffect(() => {
    if (!playing || !ready) return
    const t = setInterval(() => setI((x) => (x >= range.length - 1 ? x : x + 1)), 1000 / (FPS * speed))
    return () => clearInterval(t)
  }, [playing, speed, range.length, ready])
  useEffect(() => { if (done) setPlaying(false) }, [done])

  const activeLF = [...labelFrames].reverse().find((x) => x <= f && f - x < stride)
  const active = data.labels.filter((l) => l.frame === activeLF)
  const onLabelFrame = activeLF === f
  const collected = data.labels.filter((l) => l.frame <= f)

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
      <div className="min-w-0">
        {/* Video + boxes */}
        <div className="relative overflow-hidden rounded-2xl bg-black ring-1 ring-line">
          <img src={`/api/frame/${data.clip_id}_${f}.jpg`} alt={`${data.clip_id} frame ${f}`} className="block aspect-[854/480] w-full object-cover" />
          <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
            {active.filter((l) => isPresent(l) && l.box).map((l) => {
              const [y0, x0, y1, x1] = l.box as number[]
              const s = st(l.status)
              const dim = hover && hover !== l.key
              return (
                <g key={l.key} opacity={dim ? 0.2 : 1} style={{ transition: 'opacity .2s' }}>
                  <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill={COLOR[l.key]} fillOpacity={hover === l.key ? 0.18 : 0.07}
                    stroke={s === 'reject' ? ST_COLOR.reject : COLOR[l.key]} strokeWidth={hover === l.key ? 4 : 2.5}
                    strokeDasharray={s === 'accept' ? undefined : s === 'amber' ? '10 7' : '3 6'} vectorEffect="non-scaling-stroke" />
                </g>
              )
            })}
          </svg>
          {/* box tags as HTML so text isn't stretched */}
          {active.filter((l) => isPresent(l) && l.box).map((l) => {
            const [y0, x0] = l.box as number[]
            const s = st(l.status)
            return (
              <span key={l.key} className="pointer-events-none absolute flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-xs font-bold text-black shadow"
                style={{ top: `calc(${y0 / 10}% + 4px)`, left: `calc(${x0 / 10}% + 4px)`, background: COLOR[l.key], opacity: hover && hover !== l.key ? 0.25 : 1 }}>
                {NAME[l.key]} {typeof l.conf === 'number' && <span className="font-normal">{Math.round(l.conf * 100)}%</span>}
                {s !== 'accept' && <span className="rounded px-1 text-white" style={{ background: ST_COLOR[s] }}>{ST_WORD[s]}</span>}
              </span>
            )
          })}
          <div className="absolute left-3 top-3 flex gap-2 text-xs">
            <span className="rounded-full bg-black/70 px-2.5 py-1 font-bold">frame {f}</span>
            {activeLF !== undefined ? (
              <span className={`rounded-full px-2.5 py-1 font-bold ${onLabelFrame ? 'bg-accept text-black' : 'bg-black/70 text-mute'}`}>
                {onLabelFrame ? '● labelled frame' : `boxes from frame ${activeLF}`}
              </span>
            ) : <span className="rounded-full bg-black/70 px-2.5 py-1 text-mute">no labels on this stretch</span>}
          </div>
          <Subtitle claims={data.claims.filter((c) => c.mode === mode)} f={f} />
          {!ready && <div className="absolute inset-0 grid place-items-center bg-black/70 text-mute">Loading frames {loaded}/{range.length}…</div>}
          {done && (
            <div className="rise absolute inset-0 grid place-items-center bg-black/60 backdrop-blur-[2px]">
              <div className="text-center">
                <p className="text-lg text-mute">Clip finished. {data.labels.length} labels and {data.claims.length} narration claims checked.</p>
                <div className="mt-4 flex justify-center gap-3">
                  <button onClick={() => { setI(0); setPlaying(true) }} className="rounded-xl bg-panel-2 px-5 py-3 font-bold ring-1 ring-line hover:bg-line">↺ Replay</button>
                  <button onClick={onReport} className="rounded-xl bg-accept px-6 py-3 text-lg font-bold text-black hover:brightness-110">See the surgery report →</button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Transport */}
        <div className="mt-3 flex items-center gap-3">
          <button onClick={() => (done ? (setI(0), setPlaying(true)) : setPlaying(!playing))} className="grid h-10 w-10 place-items-center rounded-full bg-text text-ink font-bold" aria-label={playing ? 'Pause' : 'Play'}>
            {playing ? '❚❚' : done ? '↺' : '▶'}
          </button>
          <div className="flex rounded-lg bg-panel p-0.5 text-sm ring-1 ring-line">
            {[0.5, 1, 2].map((s) => <button key={s} onClick={() => setSpeed(s)} aria-pressed={speed === s} className={`rounded-md px-2.5 py-1 ${speed === s ? 'bg-line font-bold' : 'text-mute'}`}>{s}×</button>)}
          </div>
          <span className="text-sm text-mute">{Math.round((i / Math.max(1, range.length - 1)) * 100)}% · CholecSeg8k {data.video}, held-out test video</span>
          <button onClick={onReport} className="ml-auto rounded-lg px-3 py-2 text-sm font-bold text-accept hover:bg-panel">Report →</button>
        </div>

        <Timeline data={data} range={range} f={f} labelFrames={labelFrames} mode={mode} hover={hover} onSeek={(fr) => { setI(range.indexOf(fr)); setPlaying(false) }} />
      </div>

      {/* Right column */}
      <aside className="flex min-h-0 flex-col gap-5">
        <Narration claims={data.claims} f={f} mode={mode} setMode={setMode} />
        <Dataset clip={data.clip_id} rows={collected} total={data.labels.length} hover={hover} setHover={setHover} current={activeLF} />
      </aside>
    </div>
  )
}

function Subtitle({ claims, f }: { claims: ClaimRow[]; f: number }) {
  const now = claims.filter((c) => c.verdict !== 'drop' && c.t_start <= f && f <= Math.max(c.t_end, c.t_start + 12))
  if (!now.length) return null
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 to-transparent px-6 pb-4 pt-10 text-center">
      {now.slice(-2).map((c) => (
        <p key={c.claim_id} className="rise text-xl font-bold leading-snug text-white drop-shadow sm:text-2xl">
          {c.verdict === 'hedge' && <span className="mr-2 rounded bg-amber px-1.5 align-middle text-sm text-black">cautious</span>}{c.text}
        </p>
      ))}
    </div>
  )
}

function Timeline({ data, range, f, labelFrames, mode, hover, onSeek }: { data: ClipData; range: number[]; f: number; labelFrames: number[]; mode: string; hover: string | null; onSeek: (f: number) => void }) {
  const a = range[0], b = range[range.length - 1]
  const x = (fr: number) => ((fr - a) / Math.max(1, b - a)) * 100
  const claims = data.claims.filter((c) => c.mode === mode)
  const ref = useRef<HTMLDivElement>(null)
  const seek = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect()
    const fr = Math.round(a + ((e.clientX - r.left) / r.width) * (b - a))
    onSeek(range.reduce((p, c) => (Math.abs(c - fr) < Math.abs(p - fr) ? c : p), range[0]))
  }
  return (
    <div className="mt-4 rounded-2xl bg-panel p-4 ring-1 ring-line">
      <div className="grid grid-cols-[96px_1fr] gap-x-3 gap-y-1.5 text-xs">
        {KEYS.map((k) => (
          <Row key={k} label={NAME[k]} color={COLOR[k]} dim={!!hover && hover !== k}>
            {data.labels.filter((l) => l.key === k).map((l) => {
              const s = st(l.status)
              return (
                <span key={l.label_id} title={`${NAME[k]} ${isPresent(l) ? 'present' : 'absent'} · ${ST_WORD[s]} · frame ${l.frame}`}
                  className="absolute top-1/2 h-4 -translate-y-1/2 rounded-sm"
                  style={{ left: `${x(l.frame)}%`, width: `max(10px, ${(8 / Math.max(1, b - a)) * 100}%)`, background: isPresent(l) ? COLOR[k] : 'transparent', border: `2px solid ${isPresent(l) ? ST_COLOR[s] === ST_COLOR.accept ? COLOR[k] : ST_COLOR[s] : '#2B5058'}`, opacity: f >= l.frame ? 1 : 0.35 }} />
              )
            })}
          </Row>
        ))}
        <Row label="Narration" color="#E6F1F0">
          {claims.map((c, j) => (
            <span key={c.claim_id} title={`${V_WORD[c.verdict]}: ${c.text}`} className="absolute h-1.5 rounded-full"
              style={{ left: `${x(c.t_start)}%`, width: `max(6px, ${x(Math.max(c.t_end, c.t_start + 2)) - x(c.t_start)}%)`, top: `${3 + (j % 3) * 7}px`, background: V_COLOR[c.verdict], opacity: f >= c.t_start ? 1 : 0.35 }} />
          ))}
        </Row>
      </div>
      <div className="mt-2 grid grid-cols-[96px_1fr] gap-x-3">
        <span />
        <div ref={ref} onClick={seek} className="relative h-6 cursor-pointer rounded bg-panel-2" role="slider" aria-valuemin={a} aria-valuemax={b} aria-valuenow={f} aria-label="Seek">
          {labelFrames.map((lf) => <span key={lf} className="absolute top-1 h-4 w-0.5 bg-accept/70" style={{ left: `${x(lf)}%` }} />)}
          <span className="absolute -top-[150px] bottom-0 w-0.5 bg-white shadow-[0_0_8px_white]" style={{ left: `${x(f)}%` }} />
          <span className="absolute inset-y-0 left-0 rounded bg-white/10" style={{ width: `${x(f)}%` }} />
        </div>
      </div>
      <div className="mt-1 grid grid-cols-[96px_1fr] gap-x-3 text-[11px] text-mute"><span /><div className="flex justify-between"><span>frame {a}</span><span>green ticks = labelled frames · click to seek</span><span>frame {b}</span></div></div>
    </div>
  )
}

function Row({ label, color, dim, children }: { label: string; color: string; dim?: boolean; children: React.ReactNode }) {
  return (
    <>
      <span className="flex items-center gap-2 truncate" style={{ opacity: dim ? 0.35 : 1 }}><span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />{label}</span>
      <div className="relative h-6 rounded bg-panel-2/60" style={{ opacity: dim ? 0.35 : 1 }}>{children}</div>
    </>
  )
}

function Narration({ claims, f, mode, setMode }: { claims: ClaimRow[]; f: number; mode: 'grounded' | 'free'; setMode: (m: 'grounded' | 'free') => void }) {
  const list = claims.filter((c) => c.mode === mode && c.t_start <= f).sort((a, b) => b.t_start - a.t_start || b.claim_id.localeCompare(a.claim_id, undefined, { numeric: true }))
  const all = claims.filter((c) => c.mode === mode)
  const n = (v: string) => all.filter((c) => c.verdict === v).length
  return (
    <section className="flex min-h-0 flex-col rounded-2xl bg-panel p-4 ring-1 ring-line">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-lg font-bold">Narration</h2>
        <div className="flex rounded-lg bg-ink p-0.5 text-xs ring-1 ring-line">
          <button onClick={() => setMode('grounded')} aria-pressed={mode === 'grounded'} className={`rounded-md px-2.5 py-1 ${mode === 'grounded' ? 'bg-accept font-bold text-black' : 'text-mute'}`}>Checked</button>
          <button onClick={() => setMode('free')} aria-pressed={mode === 'free'} className={`rounded-md px-2.5 py-1 ${mode === 'free' ? 'bg-reject font-bold text-white' : 'text-mute'}`}>Raw model</button>
        </div>
      </div>
      <p className="mt-1 text-xs text-mute">{mode === 'grounded' ? 'Built from verified labels, then every claim re-checked.' : 'What Gemini says unchecked, run through the same checker.'}</p>
      <div className="mt-3 flex h-2 overflow-hidden rounded-full bg-ink">
        {(['speak', 'hedge', 'drop'] as const).map((v) => <span key={v} style={{ width: `${(n(v) / Math.max(1, all.length)) * 100}%`, background: V_COLOR[v] }} />)}
      </div>
      <div className="mt-1 flex gap-3 text-[11px] text-mute">{(['speak', 'hedge', 'drop'] as const).map((v) => <span key={v}><b style={{ color: V_COLOR[v] }}>{n(v)}</b> {V_WORD[v].toLowerCase()}</span>)}</div>
      <ul className="mt-3 max-h-[340px] space-y-2 overflow-y-auto pr-1">
        {list.length === 0 && <li className="text-sm text-mute">Waiting for the first claim…</li>}
        {list.map((c) => (
          <li key={c.claim_id} className="rise rounded-xl bg-ink/60 p-3 ring-1 ring-line" style={{ borderLeft: `4px solid ${V_COLOR[c.verdict]}` }}>
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="font-bold uppercase tracking-wide" style={{ color: V_COLOR[c.verdict] }}>{V_WORD[c.verdict]}</span>
              <span className="text-mute">frames {c.t_start}{c.t_end !== c.t_start && `–${c.t_end}`}</span>
            </div>
            <p className={`mt-1 text-sm leading-snug ${c.verdict === 'drop' ? 'text-mute line-through decoration-reject/70' : ''}`}>{c.text}</p>
            <p className="mt-1 text-[11px] text-mute">{why(c)}</p>
          </li>
        ))}
      </ul>
    </section>
  )
}

function Dataset({ clip, rows, total, hover, setHover, current }: { clip: string; rows: Label[]; total: number; hover: string | null; setHover: (k: string | null) => void; current?: number }) {
  const acc = rows.filter((r) => st(r.status) === 'accept').length
  const sorted = [...rows].sort((a, b) => b.frame - a.frame || KEYS.indexOf(a.key as never) - KEYS.indexOf(b.key as never))
  return (
    <section className="flex min-h-0 flex-col rounded-2xl bg-panel p-4 ring-1 ring-line" onMouseLeave={() => setHover(null)}>
      <div className="flex items-baseline justify-between">
        <h2 className="text-lg font-bold">Dataset being built</h2>
        <span className="text-sm"><b className="text-xl">{rows.length}</b><span className="text-mute">/{total} labels</span></span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-ink">
        <div className="h-full bg-accept transition-all" style={{ width: `${(acc / Math.max(1, total)) * 100}%` }} />
      </div>
      <p className="mt-1 text-[11px] text-mute">{acc} verified, {rows.length - acc} unsure or rejected. No human labels used.</p>
      <ul className="mt-3 max-h-[300px] space-y-1 overflow-y-auto pr-1">
        {sorted.map((r) => {
          const s = st(r.status)
          const live = r.frame === current
          return (
            <li key={r.label_id} onMouseEnter={() => setHover(live ? r.key : null)}
              className={`rise flex items-center gap-2.5 rounded-lg p-1.5 text-sm ${live ? 'bg-panel-2' : 'opacity-60'} ${hover === r.key && live ? 'ring-1 ring-white/40' : ''}`}>
              {isPresent(r) && r.box ? (
                <img src={`/api/crop/${clip}_${r.frame}/${r.box.map((v) => Math.round(v)).join('/')}.jpg`} alt="" className="h-9 w-12 shrink-0 rounded object-cover ring-2" style={{ ['--tw-ring-color' as string]: COLOR[r.key] }} />
              ) : <span className="grid h-9 w-12 shrink-0 place-items-center rounded bg-ink text-[10px] text-mute ring-1 ring-line">absent</span>}
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1.5 font-bold"><span className="h-2 w-2 rounded-full" style={{ background: COLOR[r.key] }} />{NAME[r.key]}</span>
                <span className="block truncate text-[11px] text-mute">f{r.frame} · {r.evidence ?? ''}</span>
              </span>
              <span className="w-14 shrink-0">
                <span className="block h-1.5 overflow-hidden rounded-full bg-ink"><span className="block h-full" style={{ width: `${(r.conf ?? 0) * 100}%`, background: ST_COLOR[s] }} /></span>
                <span className="block text-right text-[10px] text-mute">{typeof r.conf === 'number' ? `${Math.round(r.conf * 100)}%` : ''}</span>
              </span>
              <span className="w-[62px] shrink-0 rounded-full px-1.5 py-0.5 text-center text-[10px] font-bold text-black" style={{ background: ST_COLOR[s] }}>{ST_WORD[s]}</span>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
