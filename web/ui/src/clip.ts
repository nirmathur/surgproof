// Types + helpers for the clip player and report. Data: /api/clips, /api/clip/{id} (web/server.py).
export interface ClipSummary { clip_id: string; video: string; split: string; first_frame: number; last_frame: number; n_frames: number; n_labels: number; n_accept: number; n_claims: number; n_spoken: number }
export interface Label {
  label_id: string; frame_id: string; frame: number; key: string; present: number; final_present?: number | null
  conf?: number | null; trust?: number | null; box: number[] | null; evidence?: string | null; critic?: string | null
  status?: string | null; second_pass_agree?: boolean | null; resolved_round?: number | null; final_by?: string | null
  argon_reason?: string | null; p_wrong?: number | null; gt_present?: number | null; gt_correct?: boolean | null
}
export interface ClipEvent { event_id: string; type: string; label: string; t_start: number; t_end: number; status: string; frames: number[]; source: string }
export type Verdict = 'speak' | 'hedge' | 'drop'
export interface ClaimRow {
  claim_id: string; mode: 'free' | 'grounded'; type: string; text: string; statement?: string; t_start: number; t_end: number
  confidence?: number; verdict: Verdict; decided_by: string[] | string; reasons: Record<string, string> | string
  layers: Record<string, string>; gt_correct: boolean | null
}
export interface ClipData { clip_id: string; video: string; frames: number[]; label_source: string | null; labels: Label[]; events: ClipEvent[]; claims: ClaimRow[] }

async function get<T>(p: string): Promise<T> { const r = await fetch(p); if (!r.ok) throw new Error(`${p} returned ${r.status}`); return r.json() }
export const clipApi = { clips: () => get<{ clips: ClipSummary[] }>('/api/clips'), clip: (id: string) => get<ClipData>(`/api/clip/${id}`) }

export const KEYS = ['grasper', 'l_hook', 'liver', 'gallbladder', 'cystic_duct', 'blood'] as const
export const NAME: Record<string, string> = { grasper: 'Grasper', l_hook: 'L-hook', liver: 'Liver', gallbladder: 'Gallbladder', cystic_duct: 'Cystic duct', blood: 'Blood' }
export const COLOR: Record<string, string> = { grasper: '#38BDF8', l_hook: '#A78BFA', liver: '#FB7185', gallbladder: '#A3E635', cystic_duct: '#FACC15', blood: '#F97316' }
export const KIND: Record<string, string> = { grasper: 'instrument', l_hook: 'instrument', liver: 'anatomy', gallbladder: 'anatomy', cystic_duct: 'anatomy', blood: 'finding' }

export type St = 'accept' | 'amber' | 'reject'
export const st = (s?: string | null): St => { const v = (s ?? '').toLowerCase(); return v === 'accept' ? 'accept' : v === 'reject' || v === 'escalate' ? 'reject' : 'amber' }
export const ST_COLOR: Record<St, string> = { accept: '#22C55E', amber: '#F59E0B', reject: '#EF4444' }
export const ST_WORD: Record<St, string> = { accept: 'Verified', amber: 'Unsure', reject: 'Rejected' }
export const V_COLOR: Record<Verdict, string> = { speak: '#22C55E', hedge: '#F59E0B', drop: '#EF4444' }
export const V_WORD: Record<Verdict, string> = { speak: 'Said', hedge: 'Said with caution', drop: 'Withheld' }
export const LAYER_NAME: Record<string, string> = {
  L1: 'Vocabulary', L2: 'Surgical rules', L3: 'Timing', L4: 'Verified labels', P: 'Proportionality',
  L5: 'Blind visual re-check', L5c: 'CLIP re-check', L6: 'Blacked-out frame test', L7: 'Guideline sources',
}
export const isPresent = (l: Label) => (l.final_present ?? l.present) === 1
export const arr = (v: string[] | string) => (Array.isArray(v) ? v : v ? [v] : [])
export const rec = (v: Record<string, string> | string): Record<string, string> => (typeof v === 'object' && v ? v : {})
export const why = (c: ClaimRow) => {
  const by = arr(c.decided_by), r = rec(c.reasons)
  if (c.verdict === 'speak') return `Backed by ${by.map((l) => LAYER_NAME[l] ?? l).join(' + ')}`
  const f = by.find((l) => r[l])
  if (by.includes('no_evidence_layer')) return 'Nothing independent could check it'
  if (by.includes('unsupported')) return 'No evidence supported it'
  return f ? `${LAYER_NAME[f] ?? f}: ${r[f]}` : by.join(', ')
}
export const pct = (v?: number | null, d = 0) => (typeof v === 'number' ? `${(v * 100).toFixed(d)}%` : '–')
export const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null)
