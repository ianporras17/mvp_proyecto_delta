import { config } from './config';
export type PipelineEvent = { type: string; stage?: number; message?: string; counts?: { people: number; cars: number; frames_analyzed: number; limitation?: string }; id?: string; resultId?: string; logs?: string[] };
export function processVideo(file: File, onProgress: (event: PipelineEvent) => void, signal: AbortSignal) {
  const worker = new Worker(new URL('./pipeline.worker.ts', import.meta.url), { type: 'module' });
  let ticket: string | undefined;
  return new Promise<PipelineEvent>((resolve, reject) => {
    const cancelRemote = () => { if (ticket) void fetch(`${config.coordinatorUrl}/api/v1/browser-jobs/${ticket}`, { method: 'DELETE', keepalive: true }).catch(() => {}); };
    const close = () => { worker.terminate(); window.removeEventListener('pagehide', cancelRemote); signal.removeEventListener('abort', abort); };
    const abort = () => { cancelRemote(); close(); reject(new Error('Procesamiento cancelado.')); };
    window.addEventListener('pagehide', cancelRemote);
    signal.addEventListener('abort', abort, { once: true });
    worker.onmessage = ({ data }: MessageEvent<PipelineEvent>) => {
      if (data.type === 'ticket') ticket = data.id;
      else if (data.type === 'error') { close(); reject(new Error(data.message)); }
      else if (data.type === 'complete') { close(); resolve(data); }
      else onProgress(data);
    };
    worker.onerror = event => { cancelRemote(); close(); reject(new Error(event.message || 'Falló el procesamiento en el navegador.')); };
    if (signal.aborted) { abort(); return; }
    worker.postMessage({ file, settings: config });
  });
}
