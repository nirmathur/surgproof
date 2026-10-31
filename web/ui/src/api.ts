// Shapes follow docs/DESIGN.md and docs/DATA_CONTRACT.md. Everything is optional: missing = pending.
export type Status = 'accept' | 'AMBER' | 'reject'

export interface LabelRow {
  label_id?: string
  frame_id: string
  video: string
  split: string
  key: string
  present: number
  conf?: number | null
  box: number[] | null // [ymin, xmin, ymax, xmax] in 0..1000
  evidence?: string | null
  pv_match?: boolean | null
  p_wrong?: number | null
  critic?: string | null
  second_pass_present?: number | null
  second_pass_agree?: boolean | null
  status?: string | null
  final_present?: number | null
  final_by?: string | null
  resolved_round?: number | null
  argon_reason?: string | null
}

export interface Curve { curve?: [number, number][]; coverage_at_95?: number; acc_at_60?: number; aurc?: number }

export interface Results {
  n_test_labels?: number
  manual_raw_acc?: number
  blank_raw_acc?: number
  accepted?: { coverage?: number; accuracy?: number; amber_share?: number; rejected_share?: number }
  accepted_5_informative?: { coverage?: number; accuracy?: number; n_test_labels_excl_liver?: number; n_accepted_excl_liver?: number }
  amber_loop?: {
    max_rounds?: number
    ambers_by_round_test?: number[]
    per_round_test?: { round: number; ambers_remaining: number; accepted_count: number; rejected_count: number; resolved_in_round: number }[]
  }
  per_label_accepted_acc?: Record<string, number>
  single_conf?: Curve
  system_zero_label?: Curve
  blank_single_conf?: Curve
  majority_vote?: Curve
  upper_bound_calibrated?: Curve
  tracker_recheck?: { coverage?: number; accuracy?: number }
}

export interface Student {
  mean_acc_by_source?: Record<string, number>
  mean_n_train_by_source?: Record<string, number>
  headline_labels?: string[]
  n_test_frames?: number
  n_test_videos?: number
}

export interface Manual {
  procedure?: string
  sources?: { url?: string; title?: string; source_type?: string; is_real_intraop_footage?: boolean; real_intraop_footage_note?: string }[]
  phases?: string[]
  structures?: { key: string; name?: string; appearance?: string; where_found?: string; when_visible?: string; confused_with?: string[]; how_to_tell_apart?: string }[]
  critical_events?: string[]
}

export interface Overview {
  manual: Manual | null
  results: Results | null
  student: Student | null
  n_verified_labels: number | null
  tracker: { source: string | null; summary: Record<string, unknown> | null; budget_curve: Record<string, number>[] | null }
  atlas: { slug: string; procedure?: string; human_labels_used?: number; audit?: { overall_confidence?: number }; timeline?: { phases?: { name?: string }[] } }[]
  test_videos: string[]
}

export interface Claim { claim_id: string; clip_id: string; mode?: string; t_start?: number; t_end?: number; text: string; verdict?: string; source?: string; tag: 'verified' | 'unsupported' | 'contradicted'; failed_layers: string[] }

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path} returned ${r.status}`)
  return r.json()
}

export const api = {
  overview: () => get<Overview>('/api/overview'),
  labels: () => get<{ source: string | null; has_status: boolean; rows: LabelRow[] }>('/api/labels'),
  narration: () => get<{ source: string | null; claims: Claim[] }>('/api/narration'),
}

export const normStatus = (s?: string | null): Status | null => {
  if (!s) return null
  const v = s.toLowerCase()
  if (v === 'accept') return 'accept'
  if (v === 'reject' || v === 'escalate') return 'reject'
  return 'AMBER'
}

export const pct = (v?: number | null, d = 1) => (typeof v === 'number' ? `${(v * 100).toFixed(d)}%` : 'pending')
