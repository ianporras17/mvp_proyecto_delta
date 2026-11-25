import { openVideo, LocalVideo } from '../src/browser/video';
import { BrowserModels } from '../src/browser/models';
self.onmessage = async ({ data: file }) => {
  const video = new LocalVideo();
  let models: BrowserModels | undefined;
  try {
    if (file.mode === 'encode') {
      const fps = 29.671399594320487;
      const encoder = await video.encoder(64, 48, fps);
      try {
        for (let i = 0; i < 60; i++) await encoder.add(new Uint8Array(64 * 48).fill(i * 4), i);
        const result = await encoder.finish();
        const decoded = await openVideo(new File([result], 'fractional.mp4', { type: 'video/mp4' }));
        postMessage({ status: 'complete', metadata: decoded.metadata, duration: decoded.track.duration / decoded.track.timescale });
      } finally { encoder.close(); }
      return;
    }
    if (file.name.endsWith('.png')) {
      const bitmap = await createImageBitmap(file);
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height), ctx = canvas.getContext('2d')!;
      ctx.drawImage(bitmap, 0, 0); bitmap.close();
      models = await BrowserModels.load();
      const result = await models.detect(ctx.getImageData(0, 0, canvas.width, canvas.height));
      postMessage({ status: 'complete', result });
      return;
    }
    const opened = await openVideo(file);
    postMessage({ status: 'metadata', metadata: opened.metadata, codec: opened.config.codec });
    models = await BrowserModels.load();
    let result;
    for await (const frame of video.frames(opened)) { result = await models.detect(frame); break; }
    postMessage({ status: 'complete', result, metadata: opened.metadata, codec: opened.config.codec });
  } catch (e) { postMessage({ status: 'error', message: String(e) }); }
  finally { video.close(); await models?.close(); }
};
