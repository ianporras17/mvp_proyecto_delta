import { createFile, DataStream, Endianness, type Movie, type Sample, type Track } from 'mp4box';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { Muxer, ArrayBufferTarget } from 'mp4-muxer';

export async function openVideo(file: File) {
  const mp4 = createFile();
  let movie: Movie | undefined;
  let error: unknown;
  const samples: Sample[] = [];
  mp4.onError = message => { error = message; };
  mp4.onReady = info => {
    movie = info;
    if (!info.videoTracks[0]) return;
    mp4.setExtractionOptions(info.videoTracks[0].id, undefined, { nbSamples: 1 });
    mp4.start();
  };
  mp4.onSamples = (_id, _user, batch) => { samples.push(...batch); };
  const bytes = await file.arrayBuffer() as ArrayBuffer & { fileStart: number };
  bytes.fileStart = 0;
  mp4.appendBuffer(bytes); mp4.flush();
  if (error || !movie || !(movie as Movie).videoTracks.length) throw new Error('El MP4 no contiene una pista de video legible.');
  const track: Track = (movie as Movie).videoTracks[0];
  if (samples.length !== track.nb_samples) throw new Error('El MP4 contiene una secuencia incompleta.');
  const fps = track.nb_samples * track.timescale / track.duration;
  if (!Number.isFinite(fps) || fps <= 0) throw new Error('FPS de origen inválido.');
  const outputFps = Math.min(30, fps);
  const sourceWidth = track.video!.width, sourceHeight = track.video!.height;
  const config = { codec: track.codec, codedWidth: sourceWidth, codedHeight: sourceHeight } as VideoDecoderConfig;
  const entry = mp4.getTrackById(track.id).mdia.minf.stbl.stsd.entries[0];
  if ('avcC' in entry && entry.avcC) {
    const stream = new DataStream(undefined, 0, Endianness.BIG_ENDIAN);
    (entry.avcC as { write: (stream: DataStream) => void }).write(stream);
    config.description = stream.buffer.slice(8);
  }
  const selected = new Set<number>();
  let next = 0;
  for (let i = 0; i < samples.length; i++) if (i / fps + 1e-9 >= next) { selected.add(i); next += 1 / outputFps; }
  const scale = Math.min(1, 1920 / sourceWidth, 1080 / sourceHeight);
  const width = Math.floor(sourceWidth * scale + 1e-9) >> 1 << 1, height = Math.floor(sourceHeight * scale + 1e-9) >> 1 << 1;
  if (Math.min(width, height) < 2) throw new Error('Dimensiones de video inválidas.');
  return { file, track, samples, config, selected, sourceWidth, sourceHeight, metadata: { width, height, fps: outputFps, frame_count: selected.size, channels: 1, dtype: 'uint8', version: 2 } };
}
export type OpenVideo = Awaited<ReturnType<typeof openVideo>>;

export class LocalVideo {
  private ffmpeg?: FFmpeg;
  private async ff() {
    if (!this.ffmpeg) {
      this.ffmpeg = new FFmpeg();
      await this.ffmpeg.load({ coreURL: `${self.location.origin}/vendor/ffmpeg/ffmpeg-core.js`, wasmURL: `${self.location.origin}/vendor/ffmpeg/ffmpeg-core.wasm` }, { signal: AbortSignal.timeout(60_000) });
    }
    return this.ffmpeg;
  }
  async *frames(video: OpenVideo): AsyncGenerator<ImageData> {
    if (typeof VideoDecoder !== 'undefined' && (await VideoDecoder.isConfigSupported(video.config).catch(() => ({ supported: false }))).supported) {
      yield* this.webFrames(video);
    } else {
      // No upload/transcode service: MPEG-4 Part 2 (FMP4) stays in local WASM.
      const ff = await this.ff();
      await ff.writeFile('original.mp4', new Uint8Array(await video.file.arrayBuffer()));
      try {
        for (let start = 0; start < video.samples.length; start += 8) {
          const end = Math.min(video.samples.length - 1, start + 7);
          if (await ff.exec(['-i', 'original.mp4', '-an', '-vf', `select=between(n\\,${start}\\,${end})`, '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-y', 'frames.rgba']) !== 0) throw new Error('Falló la decodificación local del MP4.');
          const data = await ff.readFile('frames.rgba') as Uint8Array;
          await ff.deleteFile('frames.rgba');
          const frameSize = video.sourceWidth * video.sourceHeight * 4;
          if (data.length !== (end - start + 1) * frameSize) throw new Error('FFmpeg devolvió cuadros incompletos.');
          for (let i = start; i <= end; i++) if (video.selected.has(i)) yield new ImageData(new Uint8ClampedArray(data.slice((i - start) * frameSize, (i - start + 1) * frameSize)), video.sourceWidth, video.sourceHeight);
        }
      } finally { await ff.deleteFile('original.mp4'); }
    }
  }
  private async *webFrames(video: OpenVideo) {
    const ready: VideoFrame[] = [];
    let failure: Error | undefined, signal: (() => void) | undefined, flushDone = false;
    const notify = () => signal?.();
    const decoder = new VideoDecoder({ output: frame => { ready.push(frame); notify(); }, error: e => { failure = e; notify(); } });
    decoder.ondequeue = notify;
    decoder.configure(video.config);
    const canvas = new OffscreenCanvas(video.sourceWidth, video.sourceHeight), context = canvas.getContext('2d', { willReadFrequently: true })!;
    let cursor = 0, frameIndex = 0, flushing: Promise<void> | undefined;
    try {
      while (!flushDone || ready.length) {
        if (failure) throw failure;
        while (cursor < video.samples.length && ready.length < 4 && decoder.decodeQueueSize < 4) {
          const sample = video.samples[cursor++];
          decoder.decode(new EncodedVideoChunk({ type: sample.is_sync ? 'key' : 'delta', timestamp: Math.round(sample.cts * 1e6 / sample.timescale), duration: Math.round(sample.duration * 1e6 / sample.timescale), data: sample.data! }));
        }
        if (cursor === video.samples.length && !flushing) flushing = decoder.flush().then(() => { flushDone = true; notify(); }).catch(e => { failure = e; notify(); });
        if (ready.length) {
          const frame = ready.shift()!;
          try {
            if (video.selected.has(frameIndex)) { context.drawImage(frame, 0, 0); yield context.getImageData(0, 0, canvas.width, canvas.height); }
          } finally { frame.close(); frameIndex++; }
        } else if (!flushDone) await new Promise<void>(resolve => { signal = resolve; });
      }
      if (frameIndex !== video.samples.length) throw new Error('La decodificación perdió cuadros.');
    } finally { if (decoder.state !== 'closed') decoder.close(); ready.forEach(f => f.close()); }
  }
  async encoder(width: number, height: number, fps: number) {
    const config: VideoEncoderConfig = { codec: 'avc1.420033', width, height, framerate: fps, bitrate: 4_000_000, avc: { format: 'avc' }, hardwareAcceleration: 'prefer-software' };
    if (typeof VideoEncoder !== 'undefined' && (await VideoEncoder.isConfigSupported(config).catch(() => ({ supported: false }))).supported) {
      // mp4-muxer's optional frameRate accepts only integers. For fractional FPS
      // preserve our explicit microsecond timestamps instead of quantizing FPS.
      const muxer = new Muxer({ target: new ArrayBufferTarget(), video: { codec: 'avc', width, height }, fastStart: 'in-memory', firstTimestampBehavior: 'offset' });
      let failure: Error | undefined;
      const encoder = new VideoEncoder({ output: (chunk, metadata) => muxer.addVideoChunk(chunk, metadata), error: e => { failure = e; } });
      encoder.configure(config);
      return {
        add: async (gray: Uint8Array, index: number) => {
          if (failure) throw failure;
          const rgba = new Uint8Array(width * height * 4);
          gray.forEach((v, i) => { rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = v; rgba[i * 4 + 3] = 255; });
          const frame = new VideoFrame(rgba, { format: 'RGBA', codedWidth: width, codedHeight: height, timestamp: Math.round(index * 1e6 / fps), duration: Math.round((index + 1) * 1e6 / fps) - Math.round(index * 1e6 / fps) });
          try { encoder.encode(frame, { keyFrame: index % 60 === 0 }); } finally { frame.close(); }
          if (encoder.encodeQueueSize > 4) await encoder.flush();
        },
        finish: async () => { await encoder.flush(); if (failure) throw failure; muxer.finalize(); encoder.close(); return new Blob([muxer.target.buffer], { type: 'video/mp4' }); },
        close: () => { if (encoder.state !== 'closed') encoder.close(); },
      };
    }
    const ff = await this.ff(), files: string[] = [];
    return {
      add: async (gray: Uint8Array, index: number) => {
        await ff.writeFile('output.gray', gray);
        const name = `encoded-${index}.mp4`;
        if (await ff.exec(['-f', 'rawvideo', '-pixel_format', 'gray', '-video_size', `${width}x${height}`, '-framerate', String(fps), '-i', 'output.gray', '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-video_track_timescale', '1000000', '-y', name]) !== 0) throw new Error('Falló la codificación local.');
        await ff.deleteFile('output.gray'); files.push(name);
      },
      finish: async () => {
        await ff.writeFile('concat.txt', files.map(name => `file '${name}'`).join('\n'));
        if (await ff.exec(['-f', 'concat', '-safe', '0', '-i', 'concat.txt', '-c', 'copy', '-an', '-movflags', '+faststart', '-y', 'result.mp4']) !== 0) throw new Error('No se pudo finalizar el MP4.');
        const bytes = await ff.readFile('result.mp4') as Uint8Array<ArrayBuffer>;
        return new Blob([bytes], { type: 'video/mp4' });
      },
      close: () => {},
    };
  }
  close() { this.ffmpeg?.terminate(); }
}

/** OpenCV BGR2GRAY's integer coefficients, before optional INTER_AREA resize. */
export function grayscale(image: ImageData, width: number, height: number) {
  const source = new Uint8Array(image.width * image.height);
  for (let i = 0; i < source.length; i++) source[i] = (image.data[i * 4] * 4899 + image.data[i * 4 + 1] * 9617 + image.data[i * 4 + 2] * 1868 + 8192) >> 14;
  if (image.width === width && image.height === height) return source;
  const output = new Uint8Array(width * height), sx = image.width / width, sy = image.height / height;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const x0 = x * sx, x1 = (x + 1) * sx, y0 = y * sy, y1 = (y + 1) * sy;
    let sum = 0;
    for (let yy = Math.floor(y0); yy < Math.ceil(y1); yy++) for (let xx = Math.floor(x0); xx < Math.ceil(x1); xx++) sum += source[Math.min(image.height - 1, yy) * image.width + Math.min(image.width - 1, xx)] * (Math.min(x1, xx + 1) - Math.max(x0, xx)) * (Math.min(y1, yy + 1) - Math.max(y0, yy));
    output[y * width + x] = Math.round(sum / (sx * sy));
  }
  return output;
}
