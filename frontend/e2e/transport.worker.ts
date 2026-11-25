import { BrowserMpc, CHUNK_SIZE } from '../src/browser/external-io';

self.onmessage = async () => {
  const root = 'http://localhost:8000/api/v1/browser-jobs';
  const mpc = new BrowserMpc();
  let ticket: string | undefined, heartbeat: ReturnType<typeof setInterval> | undefined;
  const channels: { privatePackets: number; publicMasks: number; receivedShares: number }[] = [];
  const OriginalSocket = WebSocket;
  // Inspect shape/range in the trusted browser; never save wire bodies.
  self.WebSocket = class extends OriginalSocket {
    private packets = 0;
    private received = 0;
    private audit = { privatePackets: 0, publicMasks: 0, receivedShares: 0 };
    constructor(url: string | URL) {
      super(url); channels.push(this.audit);
      this.addEventListener('message', event => {
        this.received++;
        if (this.received > 2) {
          const data = event.data as ArrayBuffer, view = new DataView(data);
          if (data.byteLength !== CHUNK_SIZE * 8 || view.getUint32(4, true) === 0) throw new Error('A party sent clear pixels instead of an individual contribution.');
          this.audit.receivedShares++;
        }
      });
    }
    send(body: ArrayBuffer) {
      this.packets++;
      if (this.packets > 1) {
        const view = new DataView(body);
        if (body.byteLength !== CHUNK_SIZE * 8) throw new Error('Wrong wire length');
        if (this.packets % 2 === 0) {
          for (let i = 0; i < CHUNK_SIZE; i++) if (view.getUint32(i * 8, true) > 1 || view.getUint32(i * 8 + 4, true)) throw new Error('Wrong public mask');
          this.audit.publicMasks++;
        } else {
          if (view.getUint32(4, true) === 0) throw new Error('Private input is not masked');
          this.audit.privatePackets++;
        }
      }
      super.send(body);
    }
  };
  const api = async (path: string, method = 'GET', body?: unknown) => {
    const response = await fetch(`${root}${path}`, { method, ...(body ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  };
  try {
    let job = await api('', 'POST', { filename: 'T49-transport-verification.mp4', format: 'mp4' });
    ticket = job.id;
    heartbeat = setInterval(() => { void api(`/${ticket}`).catch(() => {}); }, 10_000);
    while (job.status === 'QUEUED') { await new Promise(r => setTimeout(r, 1000)); job = await api(`/${ticket}`); }
    const length = 2 * CHUNK_SIZE + 16;
    job = await api(`/${ticket}/start`, 'POST', { metadata: { width: 2, height: 2, frame_count: length / 4, fps: 30, dtype: 'uint8', channels: 1, version: 2 } });
    await mpc.connect(['http://localhost:8100', 'http://localhost:8101', 'http://localhost:8102'], job.video_id, length);
    for (let offset = 0; offset < length; offset += CHUNK_SIZE) {
      const pixels = Uint8Array.from({ length: Math.min(CHUNK_SIZE, length - offset) }, (_, i) => (offset + i) % 256);
      const mask = Uint8Array.from(pixels, (_, i) => (offset + i) % 5 === 0 ? 1 : 0);
      await mpc.exchange(pixels, mask);
    }
    mpc.finish();
    await api(`/${ticket}/storing`, 'POST'); await api(`/${ticket}/complete`, 'POST');
    postMessage({ status: 'complete', channels, chunks: 3, verifiedPixels: length, padding: CHUNK_SIZE - 16 });
  } catch (error) {
    if (ticket) await api(`/${ticket}`, 'DELETE').catch(() => {});
    postMessage({ status: 'error', message: String(error) });
  } finally { if (heartbeat) clearInterval(heartbeat); mpc.close(); }
};
