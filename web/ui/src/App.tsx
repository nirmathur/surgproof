import { useEffect, useState } from 'react'
import { clipApi, type ClipData, type ClipSummary } from './clip'
import Player from './Player'
import Report from './Report'

const parse = () => {
  const [, clip, view] = window.location.hash.replace(/^#\/?/, '').split('/')
  return { clip: clip || null, view: view === 'report' ? 'report' : 'watch' } as { clip: string | null; view: 'watch' | 'report' }
}

export default function App() {
  const [route, setRoute] = useState(parse)
  const [clips, setClips] = useState<ClipSummary[] | null>(null)
  const [data, setData] = useState<ClipData | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => { const h = () => setRoute(parse()); window.addEventListener('hashchange', h); return () => window.removeEventListener('hashchange', h) }, [])
  useEffect(() => { clipApi.clips().then((r) => setClips(r.clips)).catch((e) => setErr(String(e))) }, [])
  const clipId = route.clip ?? clips?.[0]?.clip_id ?? null
  useEffect(() => {
    if (!clipId) return
    setData(null)
    clipApi.clip(clipId).then(setData).catch((e) => setErr(String(e)))
  }, [clipId])
  useEffect(() => { window.scrollTo({ top: 0 }) }, [route.view, clipId])

  const go = (clip: string, view: 'watch' | 'report' = 'watch') => { window.location.hash = `/clip/${clip}${view === 'report' ? '/report' : ''}` }
  const idx = clips?.findIndex((c) => c.clip_id === clipId) ?? -1
  const next = clips && idx >= 0 && idx < clips.length - 1 ? clips[idx + 1] : undefined

  if (err) return <p className="p-10 text-reject">Can't reach the data server: {err}. Start it with <code>uvicorn web.server:app --port 8600</code>.</p>

  return (
    <div className="mx-auto max-w-[1600px] px-5 pb-12 pt-4">
      <header className="mb-4 flex flex-wrap items-center gap-4">
        <span className="text-xl font-bold tracking-tight">Surg<span className="text-accept">Proof</span></span>
        <nav aria-label="Videos" className="flex flex-1 gap-2 overflow-x-auto pb-1">
          {(clips ?? []).map((c) => {
            const on = c.clip_id === clipId
            return (
              <button key={c.clip_id} onClick={() => go(c.clip_id, route.view)} aria-current={on}
                className={`group relative shrink-0 overflow-hidden rounded-lg ring-2 transition ${on ? 'ring-accept' : 'ring-transparent opacity-60 hover:opacity-100'}`}>
                <img src={`/api/frame/${c.clip_id}_${c.first_frame}.jpg`} alt="" className="h-12 w-[86px] object-cover" />
                <span className="absolute inset-x-0 bottom-0 bg-black/70 px-1 text-[10px] font-bold">{c.video}</span>
                <span className="absolute right-0.5 top-0.5 rounded bg-black/70 px-1 text-[9px]">{Math.round((c.n_accept / Math.max(1, c.n_labels)) * 100)}%</span>
              </button>
            )
          })}
        </nav>
        {clipId && (
          <div className="flex rounded-xl bg-panel p-1 ring-1 ring-line">
            <button onClick={() => go(clipId)} aria-pressed={route.view === 'watch'} className={`rounded-lg px-4 py-1.5 text-sm font-bold ${route.view === 'watch' ? 'bg-text text-ink' : 'text-mute'}`}>▶ Watch</button>
            <button onClick={() => go(clipId, 'report')} aria-pressed={route.view === 'report'} className={`rounded-lg px-4 py-1.5 text-sm font-bold ${route.view === 'report' ? 'bg-text text-ink' : 'text-mute'}`}>Report</button>
          </div>
        )}
      </header>

      {!data ? <p className="p-10 text-mute">Loading {clipId ?? 'videos'}…</p>
        : route.view === 'report'
          ? <Report data={data} onWatch={() => go(data.clip_id)} next={next ? () => go(next.clip_id) : undefined} />
          : <Player key={data.clip_id} data={data} onReport={() => go(data.clip_id, 'report')} />}
    </div>
  )
}
