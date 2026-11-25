import * as ort from 'onnxruntime-web/wasm';
import { nms, type Box, type Detection } from './geometry';

ort.env.wasm.wasmPaths = `${self.location.origin}/vendor/ort/`;
ort.env.wasm.numThreads = Math.min(4, self.navigator.hardwareConcurrency || 1);
type Outputs = Record<string, ort.Tensor>;

function floats(output: ort.Tensor | undefined, count?: number) {
  if (!output || !(output.data instanceof Float32Array) || (count !== undefined && output.data.length !== count) || output.data.some(v => !Number.isFinite(v))) throw new Error('El modelo devolvió tensores incompatibles o no finitos.');
  return output.data;
}

function bgrTensor(rgba: Uint8ClampedArray, width: number, height: number, paddedWidth = width, paddedHeight = height) {
  const plane = paddedWidth * paddedHeight, data = new Float32Array(plane * 3);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const source = (y * width + x) * 4, target = y * paddedWidth + x;
    data[target] = rgba[source + 2]; data[plane + target] = rgba[source + 1]; data[plane * 2 + target] = rgba[source];
  }
  return new ort.Tensor('float32', data, [1, 3, paddedHeight, paddedWidth]);
}

export class BrowserModels {
  private sessions: ort.InferenceSession[] = [];
  static async load() {
    const models = new BrowserModels();
    try {
      const response = await fetch('/models/manifest.json');
      if (!response.ok) throw new Error('No se encontró el manifiesto de modelos.');
      const manifest = await response.json();
      for (const name of ['face', 'plate', 'objects']) {
        const resource = await fetch(`/models/${name}.onnx`);
        if (!resource.ok) throw new Error(`No se encontró el modelo ${name}.`);
        const bytes = await resource.arrayBuffer();
        const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
        if (digest !== manifest[name].sha256) throw new Error(`Modelo ${name} dañado.`);
        models.sessions.push(await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }));
      }
      return models;
    } catch (e) { await models.close(); throw e; }
  }
  async detect(image: ImageData) {
    const { width, height, data } = image;
    const pw = Math.ceil(width / 32) * 32, ph = Math.ceil(height / 32) * 32;
    const faceInput = bgrTensor(data, width, height, pw, ph);
    const faces = await this.sessions[0].run({ input: faceInput });
    const faceCandidates: Detection[] = [];
    for (const stride of [8, 16, 32]) {
      const columns = pw / stride, count = columns * ph / stride;
      const cls = floats(faces[`cls_${stride}`], count), obj = floats(faces[`obj_${stride}`], count), boxes = floats(faces[`bbox_${stride}`], count * 4);
      for (let i = 0; i < count; i++) {
        const score = Math.fround(Math.sqrt(Math.fround(Math.max(0, Math.min(1, cls[i])) * Math.max(0, Math.min(1, obj[i])))));
        if (score <= Math.fround(.6)) continue;
        const cx = Math.fround((i % columns + boxes[i * 4]) * stride), cy = Math.fround((Math.floor(i / columns) + boxes[i * 4 + 1]) * stride);
        const w = Math.fround(Math.exp(boxes[i * 4 + 2]) * stride), h = Math.fround(Math.exp(boxes[i * 4 + 3]) * stride);
        faceCandidates.push({ box: [Math.fround(cx - w / 2), Math.fround(cy - h / 2), w, h], score });
      }
    }
    // OpenCV FaceDetectorYN uses integer boxes for NMS, float boxes for output.
    const faceSorted = faceCandidates.sort((a, b) => b.score - a.score).slice(0, 5000);
    const intCandidates = faceSorted.map((d, i) => ({ ...d, box: d.box.map(Math.trunc) as Box, original: i }));
    const faceKept = nms(intCandidates, .3).map(d => faceSorted[intCandidates.indexOf(d as typeof intCandidates[number])]);
    const plates = this.decodePlates(await this.sessions[1].run({ input: bgrTensor(data, width, height) }), width, height);
    const objects = await this.detectObjects(image);
    return { regions: [...faceKept, ...plates].map(d => d.box), faces: faceKept, plates, objects };
  }
  private decodePlates(outputs: Outputs, width: number, height: number) {
    let rows = Math.floor(Math.floor((height + 1) / 2) / 2), cols = Math.floor(Math.floor((width + 1) / 2) / 2);
    const priors: number[][] = [];
    const sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]];
    [8, 16, 32, 64].forEach((step, level) => {
      rows = Math.floor(rows / 2); cols = Math.floor(cols / 2);
      for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) for (const size of sizes[level]) priors.push([Math.fround((x + .5) * step / width), Math.fround((y + .5) * step / height), Math.fround(size / width), Math.fround(size / height)]);
    });
    const loc = floats(outputs.loc, priors.length * 14), conf = floats(outputs.conf, priors.length * 2), quality = floats(outputs.iou, priors.length);
    if (conf.some(v => v < 0 || v > 1)) throw new Error('Probabilidades de placas inválidas.');
    const candidates: Detection[] = [];
    priors.forEach((p, i) => {
      const score = Math.fround(Math.sqrt(Math.fround(conf[i * 2 + 1] * Math.max(0, Math.min(1, quality[i])))));
      if (score <= Math.fround(.6)) return;
      const points = [4, 6, 10, 12].map(k => [Math.fround(Math.fround(p[0] + Math.fround(Math.fround(loc[i * 14 + k] * Math.fround(.1)) * p[2])) * width), Math.fround(Math.fround(p[1] + Math.fround(Math.fround(loc[i * 14 + k + 1] * Math.fround(.1)) * p[3])) * height)]);
      const left = Math.max(0, Math.min(...points.map(p => p[0]))), right = Math.min(width, Math.max(...points.map(p => p[0])));
      const top = Math.max(0, Math.min(...points.map(p => p[1]))), bottom = Math.min(height, Math.max(...points.map(p => p[1])));
      if (right > left && bottom > top) candidates.push({ box: [left, top, right - left, bottom - top], score });
    });
    return nms(candidates, .3, 5000).slice(0, 750);
  }
  private async detectObjects(image: ImageData) {
    const { width, height } = image, ratio = Math.min(640 / width, 640 / height);
    const rw = Math.max(1, Math.floor(width * ratio)), rh = Math.max(1, Math.floor(height * ratio));
    const data = new Float32Array(640 * 640 * 3).fill(114);
    // OpenCV INTER_LINEAR's half-pixel coordinates, top/left letterbox, RGB.
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
      const sx = Math.max(0, Math.min(width - 1, (x + .5) * width / rw - .5)), sy = Math.max(0, Math.min(height - 1, (y + .5) * height / rh - .5));
      const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
      for (let c = 0; c < 3; c++) {
        const at = (xx: number, yy: number) => image.data[(yy * width + xx) * 4 + c];
        data[c * 640 * 640 + y * 640 + x] = Math.round((at(x0, y0) * (1 - fx) + at(Math.min(width - 1, x0 + 1), y0) * fx) * (1 - fy) + (at(x0, Math.min(height - 1, y0 + 1)) * (1 - fx) + at(Math.min(width - 1, x0 + 1), Math.min(height - 1, y0 + 1)) * fx) * fy);
      }
    }
    const output = await this.sessions[2].run({ images: new ort.Tensor('float32', data, [1, 3, 640, 640]) });
    const values = floats(output.output, 8400 * 85), candidates: Detection[] = [];
    let index = 0;
    for (const stride of [8, 16, 32]) for (let y = 0; y < 640 / stride; y++) for (let x = 0; x < 640 / stride; x++, index++) {
      const offset = index * 85;
      let cls = 0; for (let c = 1; c < 80; c++) if (values[offset + 5 + c] > values[offset + 5 + cls]) cls = c;
      const score = Math.fround(values[offset + 4] * values[offset + 5 + cls]);
      if ((cls !== 0 && cls !== 2) || score <= Math.fround(.4)) continue;
      const cx = (values[offset] + x) * stride, cy = (values[offset + 1] + y) * stride;
      const w = Math.exp(Math.max(-20, Math.min(20, values[offset + 2]))) * stride, h = Math.exp(Math.max(-20, Math.min(20, values[offset + 3]))) * stride;
      const left = Math.max(0, (cx - w / 2) / ratio), right = Math.min(width, (cx + w / 2) / ratio);
      const top = Math.max(0, (cy - h / 2) / ratio), bottom = Math.min(height, (cy + h / 2) / ratio);
      if (right > left && bottom > top) candidates.push({ box: [left, top, right - left, bottom - top], score, kind: cls === 0 ? 'person' : 'car' });
    }
    return [...nms(candidates.filter(d => d.kind === 'person'), .45), ...nms(candidates.filter(d => d.kind === 'car'), .45)];
  }
  async close() { for (const s of this.sessions) await s.release(); }
}
