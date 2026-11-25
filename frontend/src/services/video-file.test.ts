import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { formatFileSize, inspectVideo, validateVideoFile } from './video-file';

function mp4(name = 'muestra.mp4', type = 'video/mp4') {
  return new File(['contenido del video'], name, { type });
}

let video: HTMLVideoElement;
let createObjectURL: Mock<(file: Blob | MediaSource) => string>;
let revokeObjectURL: Mock<(url: string) => void>;
let createElement: ReturnType<typeof vi.spyOn>;

function setMetadata(duration: number, width = 1920, height = 1080) {
  Object.defineProperties(video, {
    duration: { configurable: true, value: duration },
    videoWidth: { configurable: true, value: width },
    videoHeight: { configurable: true, value: height },
  });
}

function expectReleased() {
  expect(video.onloadedmetadata).toBeNull();
  expect(video.onerror).toBeNull();
  expect(video.hasAttribute('src')).toBe(false);
  expect(revokeObjectURL).toHaveBeenCalledExactlyOnceWith('blob:video-demo');
  expect(vi.getTimerCount()).toBe(0);
}

beforeEach(() => {
  vi.useFakeTimers();
  const originalCreateElement = document.createElement.bind(document);
  video = originalCreateElement('video');
  setMetadata(12.5);
  createElement = vi.spyOn(document, 'createElement').mockImplementation((tagName, options) => (
    tagName.toLowerCase() === 'video' ? video : originalCreateElement(tagName, options)
  ));
  createObjectURL = vi.fn((_file: Blob | MediaSource) => 'blob:video-demo');
  revokeObjectURL = vi.fn((_url: string) => {});
  vi.stubGlobal('URL', class extends URL {
    static createObjectURL = createObjectURL;
    static revokeObjectURL = revokeObjectURL;
  });
});

afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('validación del archivo de video', () => {
  it.each([
    ['muestra.mp4', 'video/mp4'],
    ['MUESTRA.MP4', 'application/mp4'],
    ['muestra.mp4', ''],
  ])('acepta un MP4 no vacío: %s con MIME %s', (name, mime) => {
    expect(() => validateVideoFile(mp4(name, mime))).not.toThrow();
  });

  it.each([
    ['muestra.mov', 'video/mp4'],
    ['muestra.webm', 'video/webm'],
    ['muestra.mp4', 'video/webm'],
    ['muestra.mp4', 'application/octet-stream'],
    ['muestra.mp4.exe', 'video/mp4'],
  ])('rechaza extensiones o MIME incompatibles: %s con MIME %s', (name, mime) => {
    expect(() => validateVideoFile(mp4(name, mime))).toThrow('formato MP4');
  });

  it('rechaza archivos vacíos antes de intentar leer sus metadatos', async () => {
    const empty = new File([], 'vacio.mp4', { type: 'video/mp4' });
    expect(() => validateVideoFile(empty)).toThrow('vacío');
    await expect(inspectVideo(empty)).rejects.toThrow('vacío');
    expect(createElement).not.toHaveBeenCalled();
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('devuelve una promesa rechazada para un formato inválido sin crear recursos', async () => {
    await expect(inspectVideo(mp4('muestra.avi', 'video/avi'))).rejects.toThrow('formato MP4');
    expect(createElement).not.toHaveBeenCalled();
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(revokeObjectURL).not.toHaveBeenCalled();
  });
});

describe('lectura local de los metadatos', () => {
  it('usa la duración y las dimensiones del elemento de video junto al nombre y tamaño originales', async () => {
    const file = mp4('corto.mp4');
    setMetadata(7.25, 1280, 720);
    const result = inspectVideo(file);
    expect(video.preload).toBe('metadata');
    expect(video.src).toBe('blob:video-demo');
    expect(createObjectURL).toHaveBeenCalledExactlyOnceWith(file);
    video.dispatchEvent(new Event('loadedmetadata'));
    await expect(result).resolves.toEqual({
      name: 'corto.mp4', size: file.size, duration: 7.25, width: 1280, height: 720,
    });
    expectReleased();
  });

  it('captura los metadatos antes de quitar src, aunque el navegador los reinicie al liberarlo', async () => {
    setMetadata(16.75, 3840, 2160);
    const originalRemoveAttribute = video.removeAttribute.bind(video);
    vi.spyOn(video, 'removeAttribute').mockImplementation((name) => {
      originalRemoveAttribute(name);
      if (name === 'src') setMetadata(Number.NaN, 0, 0);
    });
    const result = inspectVideo(mp4());
    video.dispatchEvent(new Event('loadedmetadata'));
    await expect(result).resolves.toMatchObject({ duration: 16.75, width: 3840, height: 2160 });
    expect(video.duration).toBeNaN();
    expect(video.videoWidth).toBe(0);
    expectReleased();
  });

  it.each([20, 20.05])('admite el límite de duración y la tolerancia del contenedor: %s s', async (duration) => {
    setMetadata(duration);
    const result = inspectVideo(mp4());
    video.dispatchEvent(new Event('loadedmetadata'));
    await expect(result).resolves.toMatchObject({ duration });
    expectReleased();
  });

  it.each([20.051, 21, 120, 1200, 86400])('admite videos sin un máximo fijo de duración: %s', async (duration) => {
    setMetadata(duration);
    const result = inspectVideo(mp4());
    video.dispatchEvent(new Event('loadedmetadata'));
    await expect(result).resolves.toMatchObject({ duration });
    expectReleased();
  });

  it.each([
    [Number.NaN, 1920, 1080],
    [Number.POSITIVE_INFINITY, 1920, 1080],
    [Number.NEGATIVE_INFINITY, 1920, 1080],
    [0, 1920, 1080],
    [-1, 1920, 1080],
    [10, 0, 1080],
    [10, 1920, 0],
  ])('rechaza metadatos sin duración o dimensiones utilizables: %s / %s × %s', async (duration, width, height) => {
    setMetadata(duration, width, height);
    const result = inspectVideo(mp4());
    video.dispatchEvent(new Event('loadedmetadata'));
    await expect(result).rejects.toThrow('duración y dimensiones reconocibles');
    expectReleased();
  });

  it('informa errores de archivo dañado o códec incompatible y libera los recursos', async () => {
    const result = inspectVideo(mp4());
    video.dispatchEvent(new Event('error'));
    await expect(result).rejects.toThrow('dañado o utilizar un códec no compatible');
    expectReleased();
  });

  it('cancela la espera exactamente a los diez segundos si no llegan metadatos', async () => {
    const result = inspectVideo(mp4());
    const rejection = expect(result).rejects.toThrow('leer el video a tiempo');
    await vi.advanceTimersByTimeAsync(9999);
    expect(video.hasAttribute('src')).toBe(true);
    expect(revokeObjectURL).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    await rejection;
    expectReleased();
  });
});

describe('cancelación y limpieza', () => {
  it('rechaza una señal ya cancelada antes de crear el video o su URL', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(inspectVideo(mp4(), controller.signal)).rejects.toMatchObject({ name: 'AbortError' });
    expect(createElement).not.toHaveBeenCalled();
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(revokeObjectURL).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('cancela una lectura pendiente y retira la suscripción de la señal', async () => {
    const controller = new AbortController();
    const add = vi.spyOn(controller.signal, 'addEventListener');
    const remove = vi.spyOn(controller.signal, 'removeEventListener');
    const result = inspectVideo(mp4(), controller.signal);
    const handler = add.mock.calls[0][1];
    expect(add).toHaveBeenCalledWith('abort', handler, { once: true });
    controller.abort();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    expect(remove).toHaveBeenCalledExactlyOnceWith('abort', handler);
    expectReleased();
  });

  it('retira también la suscripción de cancelación al terminar correctamente', async () => {
    const controller = new AbortController();
    const add = vi.spyOn(controller.signal, 'addEventListener');
    const remove = vi.spyOn(controller.signal, 'removeEventListener');
    const result = inspectVideo(mp4(), controller.signal);
    const handler = add.mock.calls[0][1];
    video.dispatchEvent(new Event('loadedmetadata'));
    await expect(result).resolves.toMatchObject({ duration: 12.5 });
    expect(remove).toHaveBeenCalledExactlyOnceWith('abort', handler);
    controller.abort();
    expectReleased();
  });

  it('ignora eventos tardíos después de cancelar sin liberar dos veces la URL', async () => {
    const controller = new AbortController();
    const result = inspectVideo(mp4(), controller.signal);
    const lateMetadata = video.onloadedmetadata!;
    const lateError = video.onerror!;
    controller.abort();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    lateMetadata.call(video, new Event('loadedmetadata'));
    lateError.call(video, new Event('error'));
    await vi.advanceTimersByTimeAsync(10_000);
    expectReleased();
  });
});

describe('tamaño de archivo legible', () => {
  it.each([
    [0, '1 KB'],
    [1, '1 KB'],
    [999, '1 KB'],
    [1000, '1 KB'],
    [1001, '2 KB'],
    [1_000_000, '1 MB'],
    [1_500_000, '1,5 MB'],
    [1_560_000, '1,6 MB'],
    [10_000_000, '10 MB'],
  ])('muestra %s bytes como %s', (bytes, expected) => {
    expect(formatFileSize(bytes)).toBe(expected);
  });
});
