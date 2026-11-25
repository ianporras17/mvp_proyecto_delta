import { test, expect } from '@playwright/test';
test('T49 fractional FPS survives H264 encoding and MP4 muxing', async ({ page }) => {
  test.skip(process.env.RUN_T49_PROBE !== '1');
  await page.goto('/catalog.html');
  const result = await page.evaluate(() => new Promise<{ status: string; metadata: { frame_count: number; fps: number }; duration: number }>((resolve, reject) => {
    const worker = new Worker('/e2e/probe.worker.ts', { type: 'module' });
    worker.onerror = e => reject(new Error(e.message));
    worker.onmessage = ({ data }) => { worker.terminate(); resolve(data); };
    worker.postMessage({ mode: 'encode' });
  }));
  expect(result.status, JSON.stringify(result)).toBe('complete');
  expect(result.metadata.frame_count).toBe(60);
  expect(Math.abs(result.metadata.fps - 29.671399594320487)).toBeLessThan(.0001);
  expect(Math.abs(result.duration - 60 / 29.671399594320487)).toBeLessThan(.0001);
});
