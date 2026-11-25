export interface VideoDetails {
  name: string;
  size: number;
  duration: number;
  width: number;
  height: number;
}

export function validateVideoFile(file: File) {
  if (!/\.mp4$/i.test(file.name) || (file.type && !['video/mp4', 'application/mp4'].includes(file.type))) {
    throw new Error('Selecciona un video en formato MP4.');
  }
  if (!file.size) throw new Error('El archivo está vacío. Selecciona otro video.');
}

/** Reads browser metadata only. Never uploads, decodes every frame, or transforms pixels. */
export function inspectVideo(file: File, signal?: AbortSignal): Promise<VideoDetails> {
  try { validateVideoFile(file); } catch (error) { return Promise.reject(error); }
  if (signal?.aborted) return Promise.reject(new DOMException('Lectura cancelada.', 'AbortError'));
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const url = URL.createObjectURL(file);
    let finished = false;
    const finish = (error?: Error) => {
      if (finished) return;
      finished = true;
      const details = { name: file.name, size: file.size, duration: video.duration, width: video.videoWidth, height: video.videoHeight };
      clearTimeout(timeout);
      signal?.removeEventListener('abort', abort);
      video.onloadedmetadata = null;
      video.onerror = null;
      video.removeAttribute('src');
      URL.revokeObjectURL(url);
      if (error) reject(error);
      else resolve(details);
    };
    const abort = () => finish(new DOMException('Lectura cancelada.', 'AbortError'));
    const timeout = setTimeout(() => finish(new Error('No pudimos leer el video a tiempo. Prueba otro MP4 compatible con tu navegador.')), 10_000);
    signal?.addEventListener('abort', abort, { once: true });
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      if (!Number.isFinite(video.duration) || video.duration <= 0 || !video.videoWidth || !video.videoHeight) {
        finish(new Error('El archivo no contiene un video válido con duración y dimensiones reconocibles.'));
      } else finish();
    };
    video.onerror = () => finish(new Error('No pudimos abrir este MP4. Puede estar dañado o utilizar un códec no compatible con tu navegador.'));
    video.src = url;
  });
}

export function formatFileSize(bytes: number) {
  return bytes >= 1_000_000
    ? `${(bytes / 1_000_000).toLocaleString('es-CR', { maximumFractionDigits: 1 })} MB`
    : `${Math.max(1, Math.ceil(bytes / 1_000)).toLocaleString('es-CR')} KB`;
}
