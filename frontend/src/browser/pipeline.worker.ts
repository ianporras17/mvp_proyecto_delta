import { BrowserMpc, CHUNK_SIZE } from './external-io';
import { BrowserModels } from './models';
import { makeMask, VideoCounter } from './geometry';
import { openVideo, LocalVideo, grayscale } from './video';

type Settings = { coordinatorUrl: string; partyUrls: string[]; storageUrl: string };
let ticket: string | undefined;
let heartbeat: ReturnType<typeof setInterval> | undefined;
const progress = (stage: number, message: string, extra = {}) => postMessage({ type: 'progress', stage, message, ...extra });

async function run(file: File, settings: Settings) {
  const local = new LocalVideo(), mpc = new BrowserMpc(), counter = new VideoCounter();
  let models: BrowserModels | undefined, encoder: Awaited<ReturnType<LocalVideo['encoder']>> | undefined;
  let heartbeatError: Error | undefined;
  const api = async (path: string, method = 'GET', body?: unknown) => {
    const response = await fetch(`${settings.coordinatorUrl}/api/v1/browser-jobs${path}`, { method, ...(body ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}), signal: AbortSignal.timeout(60_000) });
    if (!response.ok) throw new Error(`Coordinator: HTTP ${response.status}.`);
    return response.json();
  };
  try {
    progress(0, 'Leyendo el video en este navegador.');
    const video = await openVideo(file);
    const job = await api('', 'POST', { filename: file.name, format: 'mp4' });
    ticket = job.id;
    postMessage({ type: 'ticket', id: ticket });
    heartbeat = setInterval(() => { void api(`/${ticket}`).then(state => { if (state.status === 'FAILED') throw new Error('El trabajo fue abortado.'); }).catch(e => { heartbeatError = e; mpc.close(); }); }, 10_000);
    let state = job;
    while (state.status === 'QUEUED') {
      progress(0, `Esperando turno. Posición en la cola: ${state.position}.`);
      await new Promise(r => setTimeout(r, 1000));
      state = await api(`/${ticket}`);
    }
    if (state.status !== 'READY') throw new Error('El turno de procesamiento no está disponible.');
    progress(1, 'Preparando los tres modelos en este navegador.');
    models = await BrowserModels.load();
    const { width, height, fps, frame_count } = video.metadata;
    encoder = await local.encoder(width, height, fps);
    // Decoder is opened lazily. Prepare all runtimes only once per video.
    state = await api(`/${ticket}/start`, 'POST', { metadata: video.metadata });
    await mpc.connect(settings.partyUrls, state.video_id, width * height * frame_count);
    let pendingPixels = new Uint8Array(CHUNK_SIZE), pendingMask = new Uint8Array(CHUNK_SIZE), filled = 0;
    const returned: Uint8Array[] = [];
    let resultSize = 0, index = 0, encoded = 0, faceCount = 0, plateCount = 0;
    const started = performance.now();
    let detectionMs = 0, exchangeMs = 0;
    const consume = async (length: number) => {
      const exchangeStarted = performance.now();
      const output = await mpc.exchange(pendingPixels.slice(0, length), pendingMask.slice(0, length));
      exchangeMs += performance.now() - exchangeStarted;
      returned.push(output); resultSize += output.length;
      while (resultSize >= width * height) {
        const frame = new Uint8Array(width * height);
        let copied = 0;
        while (copied < frame.length) {
          const head = returned[0], n = Math.min(head.length, frame.length - copied);
          frame.set(head.subarray(0, n), copied); copied += n;
          if (n === head.length) returned.shift(); else returned[0] = head.slice(n);
        }
        resultSize -= frame.length;
        await encoder!.add(frame, encoded++);
      }
      filled = 0; pendingPixels = new Uint8Array(CHUNK_SIZE); pendingMask = new Uint8Array(CHUNK_SIZE);
    };
    for await (const image of local.frames(video)) {
      if (heartbeatError) throw heartbeatError;
      const detectionStarted = performance.now();
      const detections = await models.detect(image);
      detectionMs += performance.now() - detectionStarted;
      faceCount += detections.faces.length; plateCount += detections.plates.length;
      counter.update(detections.objects, index);
      const pixels = grayscale(image, width, height), mask = makeMask(detections.regions, image.width, image.height, width, height);
      let offset = 0;
      while (offset < pixels.length) {
        const n = Math.min(CHUNK_SIZE - filled, pixels.length - offset);
        pendingPixels.set(pixels.subarray(offset, offset + n), filled); pendingMask.set(mask.subarray(offset, offset + n), filled);
        filled += n; offset += n;
        if (filled === CHUNK_SIZE) await consume(filled);
      }
      index++;
      progress(2, `Procesando cuadro ${index} de ${frame_count}.`, { counts: counter.summary(), frame: index, totalFrames: frame_count, faceCount, plateCount });
    }
    if (filled) await consume(filled);
    mpc.finish();
    if (index !== frame_count || encoded !== frame_count || resultSize) throw new Error('La secuencia de salida está incompleta.');
    progress(3, 'Guardando el video anonimizado.');
    const result = await encoder.finish();
    await api(`/${ticket}/storing`, 'POST');
    const form = new FormData(); form.append('video', result, 'video-anonimizado.mp4'); form.append('counts', JSON.stringify(counter.summary()));
    const response = await fetch(`${settings.storageUrl}/api/v1/results`, { method: 'POST', body: form });
    if (!response.ok) throw new Error(`No se pudo guardar el resultado (HTTP ${response.status}).`);
    const stored = await response.json();
    await api(`/${ticket}/complete`, 'POST');
    const published = await fetch(`${settings.storageUrl}/api/v1/results/${stored.id}/publish`, { method: 'POST' });
    if (!published.ok) throw new Error('El resultado se recibió, pero no se pudo publicar en el catálogo.');
    const timings = { detection_seconds: detectionMs / 1000, exchange_seconds: exchangeMs / 1000, processing_seconds: (performance.now() - started) / 1000 };
    postMessage({ type: 'complete', counts: counter.summary(), resultId: stored.id, faceCount, plateCount, metadata: video.metadata, timings,
      logs: [`Cuadros validados antes del códec: ${encoded}/${frame_count}.`, `Detecciones de rostros: ${faceCount}; placas: ${plateCount}.`, `Inferencia: ${timings.detection_seconds.toFixed(1)} s; intercambio MPC: ${timings.exchange_seconds.toFixed(1)} s.`, `Resultado almacenado: ${stored.id}.`] });
  } catch (e) {
    if (ticket) await api(`/${ticket}`, 'DELETE').catch(() => {});
    postMessage({ type: 'error', message: e instanceof Error ? e.message : String(e) });
  } finally {
    if (heartbeat) clearInterval(heartbeat);
    mpc.close(); encoder?.close(); local.close(); await models?.close();
  }
}
self.onmessage = event => { void run(event.data.file, event.data.settings); };
