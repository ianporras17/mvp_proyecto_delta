export type Box = [number, number, number, number];
export type Detection = { box: Box; score: number; kind?: 'person' | 'car' };

export function iou(a: Box, b: Box) {
  const area = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  return area / (a[2] * a[3] + b[2] * b[3] - area);
}
export function nms(items: Detection[], threshold: number, topK = 0) {
  let sorted = items.map((d, i) => ({ d, i })).sort((a, b) => b.d.score - a.d.score || a.i - b.i);
  if (topK) sorted = sorted.slice(0, topK);
  const kept: Detection[] = [];
  for (const { d } of sorted) if (kept.every(k => iou(k.box, d.box) <= threshold)) kept.push(d);
  return kept;
}

export function makeMask(boxes: Box[], sourceWidth: number, sourceHeight: number, width: number, height: number) {
  const mask = new Uint8Array(width * height);
  for (const [x, y, w, h] of boxes) {
    if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) throw new Error('Geometría del detector inválida.');
    const left = Math.max(0, Math.min(width, Math.floor(x * width / sourceWidth)));
    const right = Math.max(0, Math.min(width, Math.ceil((x + w) * width / sourceWidth)));
    const top = Math.max(0, Math.min(height, Math.floor(y * height / sourceHeight)));
    const bottom = Math.max(0, Math.min(height, Math.ceil((y + h) * height / sourceHeight)));
    for (let row = top; row < bottom; row++) mask.fill(1, row * width + left, row * width + right);
  }
  return mask;
}

type Track = { kind: string; box: Box; last: number; hits: number; counted: boolean; velocity: [number, number] };
export class VideoCounter {
  private tracks = new Map<number, Track>();
  private nextId = 1;
  private index = -1;
  private people = 0;
  private cars = 0;
  update(detections: Detection[], index: number) {
    if (index !== this.index + 1) throw new Error('El conteo requiere índices consecutivos.');
    this.index = index;
    this.tracks.forEach((t, id) => { if (index - t.last > 61) this.tracks.delete(id); });
    const candidates: [number, number, number][] = [];
    for (const [id, t] of this.tracks) {
      const gap = index - t.last;
      const predicted: Box = [t.box[0] + t.velocity[0] * gap, t.box[1] + t.velocity[1] * gap, t.box[2], t.box[3]];
      detections.forEach((d, i) => { if (d.kind === t.kind) { const score = Math.max(iou(predicted, d.box), iou(t.box, d.box)); if (score >= .3) candidates.push([score, id, i]); } });
    }
    const matchedTracks = new Set<number>(), matchedDetections = new Set<number>();
    candidates.sort((a, b) => b[0] - a[0] || b[1] - a[1] || b[2] - a[2]);
    for (const [, id, i] of candidates) {
      if (matchedTracks.has(id) || matchedDetections.has(i)) continue;
      const t = this.tracks.get(id)!, d = detections[i], gap = index - t.last;
      t.velocity = [(d.box[0] - t.box[0]) / gap, (d.box[1] - t.box[1]) / gap];
      t.box = d.box; t.last = index; t.hits++;
      matchedTracks.add(id); matchedDetections.add(i);
    }
    detections.forEach((d, i) => { if (!matchedDetections.has(i)) this.tracks.set(this.nextId++, { kind: d.kind!, box: d.box, last: index, hits: 1, counted: false, velocity: [0, 0] }); });
    for (const t of this.tracks.values()) if (!t.counted && t.hits >= 3) { t.counted = true; if (t.kind === 'person') this.people++; else this.cars++; }
  }
  summary() { return { people: this.people, cars: this.cars, frames_analyzed: this.index + 1, limitation: 'Trayectorias estimadas; oclusiones, cruces o reingresos pueden omitir o duplicar objetos.' }; }
}
