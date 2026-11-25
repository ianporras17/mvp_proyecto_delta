import { test, expect } from '@playwright/test';
import path from 'node:path';
for (const name of ['video1.mp4', 'video2.mp4', 'video3.mp4']) test(`T49 critical browser probe ${name}`, async ({ page }) => {
  test.skip(process.env.RUN_T49_PROBE !== '1');
  test.setTimeout(180_000);
  page.on('pageerror', error => console.log('BROWSER ERROR', error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('CONSOLE', message.text()); });
  await page.goto('/catalog.html');
  await page.evaluate(() => { document.body.innerHTML = '<input type="file">'; });
  await page.locator('input').setInputFiles(path.resolve('../..', name));
  const result = await page.evaluate(async () => {
    const worker = new Worker('/e2e/probe.worker.ts', { type: 'module' });
    return new Promise<Record<string, unknown>>((resolve, reject) => {
      worker.onerror = e => reject(new Error(e.message));
      worker.onmessage = ({ data }) => { if (data.status !== 'metadata') { worker.terminate(); resolve(data); } };
      worker.postMessage(document.querySelector('input')!.files![0]);
    });
  });
  console.log(name, JSON.stringify(result));
  expect(result.status).toBe('complete');
});
