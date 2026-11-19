export interface Protocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  protocols: Protocol[];
  watchlist: string[];
}

export const day = 86_400_000;
export const dollars = (log: number) => Math.max(1, Math.round(10 ** log));
export const logCap = (p: Protocol) => Math.log10(p.mcap_usd!);
export const size = (p: Protocol) => Math.log10(p.tvl_usd) - 8;
export const comparable = (p: Protocol) => p.tvl_usd > 0 && (p.mcap_usd ?? 0) > 0 && p.category !== 'CEX';
export const anchor = (p: Protocol) => comparable(p) && Boolean(p.gecko_id);
export const order = (a: string, b: string) => a < b ? -1 : a > b ? 1 : 0;
export const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
export const labels = ['Baseline', 'TVL', '7-day flow proxy', 'Listing age', 'Chain breadth', 'Sector', 'Category'];

export function quantile(values: number[], p: number) {
  const sorted = [...values].sort((a, b) => a - b);
  const at = p * (sorted.length - 1);
  const lo = Math.floor(at);
  return sorted[lo] + (sorted[Math.ceil(at)] - sorted[lo]) * (at - lo);
}

function sector(category: string) {
  if (/bridge|chain|privacy/i.test(category)) return 'Infrastructure';
  if (/stak|index|allocator|yield aggregat/i.test(category)) return 'Managed exposure';
  if (/dex|derivative|perpetual|option|prediction|basis|liquidity/i.test(category)) return 'Trading';
  if (/lend|cdp|collateral|credit/i.test(category)) return 'Credit';
  return category;
}

type Term = [number, number];
const specs = [
  {name: 'Pooled size', features: [], pooling: false},
  {name: 'Sector and category', features: [], pooling: true},
  {name: 'Sector, flow and age', features: [2, 3], pooling: true},
  {name: 'Sector and all signals', features: [2, 3, 4], pooling: true},
];
type Spec = typeof specs[number];

// Penalized log-cap regression. Coordinate descent updates a shared residual so
// correlated category and sector effects are shrunk jointly, rather than stacked.
export function fit(train: Protocol[], snapshotDate: string, spec: Spec) {
  const category = (p: Protocol) => p.category ?? 'Unclassified';
  const groups = [...new Set(train.map(category))].sort(order);
  const sectors = [...new Set(groups.map(sector))].sort(order);
  const categoryIndex = new Map(groups.map((group, i) => [group, 5 + sectors.length + i]));
  const sectorIndex = new Map(sectors.map((group, i) => [group, 5 + i]));
  const observedAt = Date.parse(snapshotDate);
  function terms(p: Protocol): Term[] {
    const age = (observedAt - Date.parse(p.listed_at ?? '')) / day;
    const x = [
      1, size(p),
      Math.asinh(clamp(p.change_7d ?? 0, -100, 100) / 20),
      Number.isFinite(age) ? Math.log10(1 + Math.max(0, age) / 365) - 0.5 : 0,
      Math.log2(Math.max(1, p.chains.length)),
    ];
    const result: Term[] = [0, 1, ...spec.features].map(i => [i, x[i]]);
    if (spec.pooling) {
      for (const i of [sectorIndex.get(sector(category(p))), categoryIndex.get(category(p))]) {
        if (i !== undefined) result.push([i, 1]);
      }
    }
    return result;
  }
  const columns: Term[][] = Array.from({length: 5 + sectors.length + groups.length}, () => []);
  train.forEach((p, j) => terms(p).forEach(([i, x]) => columns[i].push([j, x])));
  const beta = columns.map(() => 0);
  const residual = train.map(logCap);
  const weights = train.map(() => 1);
  for (let pass = 0; pass < 5; pass++) {
    if (pass > 0) residual.forEach((e, j) => { weights[j] = Math.min(1, 0.6 / Math.max(1e-9, Math.abs(e))); });
    for (let sweep = 0; sweep < 32; sweep++) {
      for (const [i, column] of columns.entries()) {
        if (!column.length) continue;
        const penalty = i === 0 ? 0 : i === 1 ? 1 : i < 5 ? 8 : i < 5 + sectors.length ? 5 : 8;
        let numerator = 0;
        let denominator = penalty;
        for (const [j, x] of column) {
          numerator += weights[j] * x * (residual[j] + beta[i] * x);
          denominator += weights[j] * x * x;
        }
        const next = denominator > 0 ? numerator / denominator : 0;
        const delta = next - beta[i];
        beta[i] = next;
        for (const [j, x] of column) residual[j] -= delta * x;
      }
    }
  }
  return (p: Protocol) => {
    const parts = labels.map(() => 0);
    for (const [i, x] of terms(p)) {
      const part = i < 5 ? i : i < 5 + sectors.length ? 5 : 6;
      parts[part] += beta[i] * x;
    }
    return {parts, log: parts.reduce((a, b) => a + b, 0)};
  };
}

const loss = (e: number) => Math.min(Math.abs(e), 1.5);
export function metrics(errors: number[]) {
  return {
    n: errors.length,
    median: errors.length ? 10 ** quantile(errors.map(Math.abs), 0.5) : null,
    hits: errors.filter(e => Math.abs(e) <= Math.log10(3)).length,
  };
}

function choose(date: string, splits: {train: Protocol[]; valid: Protocol[]}[]) {
  return specs.map(spec => ({
    spec,
    loss: splits.reduce((sum, {train, valid}) => {
      const predict = fit(train, date, spec);
      return sum + valid.reduce((total, p) => total + loss(logCap(p) - predict(p).log), 0);
    }, 0),
  })).sort((a, b) => a.loss - b.loss)[0].spec;
}

function inner(rows: Protocol[]) {
  // Keep tokens that share a known identity together during model selection.
  const ids = [...new Set(rows.map(p => p.gecko_id ?? p.slug))].sort(order);
  const partition = new Map(ids.map((id, i) => [id, i % 5]));
  return Array.from({length: 5}, (_, fold) => ({
    train: rows.filter(p => partition.get(p.gecko_id ?? p.slug) !== fold),
    valid: rows.filter(p => partition.get(p.gecko_id ?? p.slug) === fold),
  }));
}

export function calibrate(snapshot: Snapshot) {
  const date = snapshot.as_of;
  const comparableSet = snapshot.protocols.filter(comparable);
  const anchors = comparableSet.filter(anchor);
  const folds = comparableSet.map(p => {
    const train = comparableSet.filter(q => q !== p);
    const spec = choose(date, inner(train));
    const {log} = fit(train, date, spec)(p);
    return {slug: p.slug, actual: p.mcap_usd, predicted: dollars(log), error: logCap(p) - log, spec: spec.name};
  });
  const spec = choose(date, inner(anchors));
  const errors = folds.map(p => p.error);
  return {
    anchors, spec, folds, summary: metrics(errors),
    residualBand: [quantile(errors, 0.1), quantile(errors, 0.9)],
    predict: fit(anchors, date, spec),
  };
}

export function cohortReplay(snapshot: Snapshot, anchors: Protocol[]) {
  const date = snapshot.as_of;
  const windows = [2023, 2024, 2025, 2026].map(year => {
    const validationStart = Date.UTC(year - 1, 0, 1);
    const testStart = Date.UTC(year, 0, 1);
    const testEnd = Date.UTC(year + 1, 0, 1);
    const gap = 30 * day;
    const listed = (p: Protocol) => Date.parse(p.listed_at ?? '');
    const test = snapshot.protocols.filter(p => listed(p) >= testStart && listed(p) < testEnd);
    const testTokens = new Set(test.map(p => p.gecko_id).filter(Boolean));
    const older = anchors.filter(p => !testTokens.has(p.gecko_id));
    const valid = older.filter(p => listed(p) >= validationStart && listed(p) < testStart - gap);
    const validationTokens = new Set(valid.map(p => p.gecko_id));
    const train = older.filter(p => listed(p) < validationStart - gap && !validationTokens.has(p.gecko_id));
    const canSelect = train.length >= 8 && valid.length >= 3;
    const spec = canSelect ? choose(date, [{train, valid}]) : specs[0];
    const predict = fit([...train, ...valid], date, spec);
    const predictions = test.map(p => {
      const {log} = predict(p);
      return {
        slug: p.slug, year, predicted: dollars(log),
        error: anchor(p) ? logCap(p) - log : null,
      };
    });
    return {
      year, spec: spec.name, canSelect, train: train.map(p => p.slug), valid: valid.map(p => p.slug),
      predictions, gapDays: 30,
      ...metrics(predictions.flatMap(p => p.error === null ? [] : [p.error])),
    };
  });
  return {windows, ...metrics(windows.flatMap(w => w.predictions.flatMap(p => p.error === null ? [] : [p.error])))};
}

export function refitRanges(anchors: Protocol[], candidates: Protocol[], date: string, spec: Spec) {
  const groups = new Map<string, Protocol[]>();
  for (const p of anchors) {
    const id = p.gecko_id ?? p.slug;
    groups.set(id, [...(groups.get(id) ?? []), p]);
  }
  const blocks = [...groups.values()];
  const draws = candidates.map(() => ({ranks: [] as number[], values: [] as number[]}));
  let state = 50501;
  const refits = 256;
  for (let run = 0; run < refits; run++) {
    const sample = blocks.flatMap(() => {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      return blocks[Math.floor(state / 2 ** 32 * blocks.length)];
    });
    const predict = fit(sample, date, spec);
    const book = candidates.map((p, i) => ({i, slug: p.slug, value: dollars(predict(p).log)}))
      .sort((a, b) => b.value - a.value || order(a.slug, b.slug));
    book.forEach(({i, value}, rank) => {
      draws[i].ranks.push(rank + 1);
      draws[i].values.push(value);
    });
  }
  return {refits, bySlug: new Map(candidates.map((p, i) => {
    const {ranks, values} = draws[i];
    return [p.slug, {
      rankMin: Math.min(...ranks), rankMax: Math.max(...ranks),
      rank80: [Math.floor(quantile(ranks, 0.1)), Math.ceil(quantile(ranks, 0.9))],
      value80: [Math.round(quantile(values, 0.1)), Math.round(quantile(values, 0.9))],
    }];
  }))};
}
