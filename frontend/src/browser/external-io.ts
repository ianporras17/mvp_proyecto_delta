/** MP-SPDZ ExternalIO, replicated Z2(64), semi-honest. All sums stay here. */
export const CHUNK_SIZE = 262144;
const MASK64 = (1n << 64n) - 1n;

export function encodeWords(values: readonly bigint[]): ArrayBuffer {
  const buffer = new ArrayBuffer(values.length * 8), view = new DataView(buffer);
  values.forEach((value, i) => view.setBigUint64(i * 8, value & MASK64, true));
  return buffer;
}

export function sumShares(messages: ArrayBuffer[], count: number): BigUint64Array {
  const words = sumShareWords(messages, count), result = new BigUint64Array(count);
  for (let i = 0; i < count; i++) result[i] = BigInt(words[i * 2]) | BigInt(words[i * 2 + 1]) << 32n;
  return result;
}

/** Two uint32 limbs avoid allocating a BigInt per pixel. Each intermediate is
 * below 2^34 (exact in a JS number); the carry preserves the full Z2(64) sum. */
export function sumShareWords(messages: ArrayBuffer[], count: number): Uint32Array {
  if (messages.length !== 3 || messages.some(m => m.byteLength !== count * 8)) throw new Error('Longitud de contribuciones MPC inválida.');
  const views = messages.map(m => new DataView(m));
  const sum = new Uint32Array(count * 2);
  for (let i = 0; i < count; i++) {
    const low = views[0].getUint32(i * 8, true) + views[1].getUint32(i * 8, true) + views[2].getUint32(i * 8, true);
    sum[i * 2] = low >>> 0;
    sum[i * 2 + 1] = (views[0].getUint32(i * 8 + 4, true) + views[1].getUint32(i * 8 + 4, true) + views[2].getUint32(i * 8 + 4, true) + Math.floor(low / 4294967296)) >>> 0;
  }
  return sum;
}

class PartyChannel {
  private messages: ArrayBuffer[] = [];
  private pending?: { resolve: (body: ArrayBuffer) => void; reject: (error: Error) => void };
  private error?: Error;
  readonly socket: WebSocket;
  readonly opened: Promise<void>;
  constructor(url: string) {
    this.socket = new WebSocket(url);
    this.socket.binaryType = 'arraybuffer';
    this.opened = new Promise((resolve, reject) => {
      this.socket.onopen = () => resolve();
      this.socket.onerror = () => { const e = new Error('No se pudo conectar con una party MPC.'); reject(e); this.fail(e); };
    });
    this.socket.onclose = () => this.fail(new Error('Una party MPC cerró la conexión.'));
    this.socket.onmessage = event => {
      if (!(event.data instanceof ArrayBuffer) || event.data.byteLength > CHUNK_SIZE * 8 || this.messages.length >= 4) {
        this.fail(new Error('Mensaje MPC inválido.')); this.socket.close(); return;
      }
      if (this.pending) { const { resolve } = this.pending; this.pending = undefined; resolve(event.data); }
      else this.messages.push(event.data);
    };
  }
  private fail(error: Error) { this.error = error; this.pending?.reject(error); this.pending = undefined; }
  async read(): Promise<ArrayBuffer> {
    const buffered = this.messages.shift();
    if (buffered) return buffered;
    if (this.error) throw this.error;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending = undefined; reject(new Error('MPC no respondió a tiempo.')); }, 120_000);
      this.pending = { resolve: body => { clearTimeout(timer); resolve(body); }, reject: e => { clearTimeout(timer); reject(e); } };
    });
  }
}

export class BrowserMpc {
  private channels: PartyChannel[] = [];
  private remaining = 0;
  private total = 0;
  async connect(partyUrls: string[], videoId: string, length: number) {
    if (!Number.isSafeInteger(length) || length <= 0 || partyUrls.length !== 3) throw new Error('Perfil MPC inválido.');
    this.remaining = this.total = length;
    try {
      this.channels = partyUrls.map(url => new PartyChannel(`${url.replace(/^http/, 'ws')}/api/v1/jobs/${videoId}/external-io`));
      await Promise.all(this.channels.map(c => c.opened));
      const specs = await Promise.all(this.channels.map(c => c.read()));
      if (specs.some(s => s.byteLength !== 12 || new Uint8Array(s).some((b, i) => b !== new Uint8Array(specs[0])[i]))) throw new Error('Especificaciones MPC inconsistentes.');
      const spec = new DataView(specs[0]);
      if (spec.getInt32(0, true) !== 82 || spec.getInt32(4, true) !== 64 || spec.getInt32(8, true) !== 64) throw new Error('Se requiere el ring MPC de 64 bits.');
      this.send(encodeWords([-BigInt(length), BigInt(Math.ceil(length / CHUNK_SIZE))]));
      const acknowledgements = await Promise.all(this.channels.map(c => c.read()));
      for (const ack of acknowledgements) {
        const view = new DataView(ack);
        if (ack.byteLength !== 16 || view.getBigUint64(0, true) !== 1145852993n || view.getBigUint64(8, true) !== 1n) throw new Error('Una party no confirmó anonimización.');
      }
    } catch (e) { this.close(); throw e; }
  }
  private send(buffer: ArrayBuffer) { for (const channel of this.channels) channel.socket.send(buffer); }
  async exchange(pixels: Uint8Array, mask: Uint8Array): Promise<Uint8Array> {
    const expected = Math.min(CHUNK_SIZE, this.remaining);
    if (!expected || pixels.length !== expected || mask.length !== expected || mask.some(v => v !== 0 && v !== 1)) throw new Error('Bloque o máscara MPC inválidos.');
    const publicMask = new ArrayBuffer(CHUNK_SIZE * 8), mv = new DataView(publicMask);
    for (let i = 0; i < mask.length; i++) mv.setUint32(i * 8, mask[i], true);
    this.send(publicMask);
    const random = sumShareWords(await Promise.all(this.channels.map(c => c.read())), CHUNK_SIZE);
    const privateInput = new ArrayBuffer(CHUNK_SIZE * 8), pv = new DataView(privateInput);
    for (let i = 0; i < CHUNK_SIZE; i++) {
      const low = random[i * 2] + (pixels[i] ?? 0);
      pv.setUint32(i * 8, low >>> 0, true);
      pv.setUint32(i * 8 + 4, (random[i * 2 + 1] + Math.floor(low / 4294967296)) >>> 0, true);
    }
    this.send(privateInput);
    const output = sumShareWords(await Promise.all(this.channels.map(c => c.read())), CHUNK_SIZE);
    const result = new Uint8Array(pixels.length);
    for (let i = 0; i < CHUNK_SIZE; i++) {
      const expectedPixel = i < pixels.length && !mask[i] ? pixels[i] : 0;
      if (output[i * 2] !== expectedPixel || output[i * 2 + 1] !== 0) throw new Error(`Resultado MPC inválido en el píxel ${this.total - this.remaining + i}.`);
      if (i < result.length) result[i] = output[i * 2];
    }
    this.remaining -= pixels.length;
    return result;
  }
  finish() { if (this.remaining) throw new Error('Salida MPC incompleta.'); this.close(); }
  close() { this.channels.forEach(c => c.socket.close()); }
}
