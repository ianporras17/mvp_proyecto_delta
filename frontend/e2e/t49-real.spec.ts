import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
import { login } from './auth';
test.use({ trace: 'off' });

const sources = process.env.E2E_VIDEO_PATH ? [process.env.E2E_VIDEO_PATH] : process.env.T49_VIDEO ? [process.env.T49_VIDEO] : ['video1.mp4', 'video2.mp4', 'video3.mp4'];
for (const source of sources) test(`T49 browser → MPC → storage → catalog: ${path.basename(source)}`, async ({ page, context }) => {
  const name = path.basename(source);
  test.skip(process.env.RUN_WEB_MPC_E2E !== '1', 'Requires real Docker parties, browser and root videos');
  test.setTimeout(21_600_000);
  const evidence = path.resolve('../.runtime/t49-evidence');
  await fs.mkdir(evidence, { recursive: true });
  const started = Date.now(), http: { method: string; url: string; bytes: number | null }[] = [];
  let ticket: string | undefined, resultId: string | undefined;
  page.on('pageerror', error => console.log('BROWSER ERROR', error.message));
  await page.addInitScript(() => {
    const OriginalWorker = window.Worker;
    window.Worker = class extends OriginalWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener('message', ({ data }) => { if (data.type === 'complete') (window as unknown as Record<string, unknown>).__T49_RESULT = data; });
      }
    };
  });
  page.on('console', message => { if (message.type() === 'error') console.log('BROWSER CONSOLE', message.text()); });
  context.on('request', request => {
    if (request.method() === 'POST') {
      const body = request.postDataBuffer(), url = request.url();
      http.push({ method: request.method(), url, bytes: body?.length ?? null });
      // The only multipart request is the final anonymous result, to storage.
      if (!url.includes(':8200/')) {
        expect(request.headers()['content-type'] ?? '').not.toContain('multipart');
        expect(body?.length ?? 0).toBeLessThan(4096);
      }
    }
    expect(request.url()).not.toContain(':8001/');
  });
  context.on('response', async response => {
    if (response.url().endsWith('/api/v1/browser-jobs') && response.status() === 202) ticket = (await response.json()).id;
    if (response.url().endsWith('/api/v1/results') && response.status() === 201) resultId = (await response.json()).id;
  });
  await login(page);
  await page.getByLabel('Seleccionar archivo MP4', { exact: true }).setInputFiles(path.resolve('../..', source));
  const timer = setInterval(async () => {
    console.log(name, await page.locator('.simulation-panel').innerText().catch(() => 'preparing'));
  }, 60_000);
  try {
    await expect(page.locator('.simulation-complete, .upload-error')).toBeVisible({ timeout: 21_000_000 });
    await expect(page.getByText('Video anonimizado y almacenado correctamente', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Descargar video anonimizado' })).toHaveCount(0);
    await expect(page.getByRole('region', { name: 'Conteo del video' })).toBeVisible();
    expect(ticket).toBeTruthy(); expect(resultId).toBeTruthy();
    const state = await (await context.request.get(`http://localhost:8000/api/v1/browser-jobs/${ticket}`)).json();
    expect(state.status).toBe('COMPLETED');
    await page.screenshot({ path: path.join(evidence, `${name}.completed.png`), fullPage: true });
    const counts = await page.getByRole('region', { name: 'Conteo del video' }).innerText();
    const catalog = await context.newPage();
    await catalog.goto(process.env.T49_CATALOG_URL ?? 'http://localhost:8081');
    const downloadPromise = catalog.waitForEvent('download');
    await catalog.locator(`a[href$="/${resultId}/download"]`).click();
    const download = await downloadPromise;
    await download.saveAs(path.join(evidence, `${name}.anonymous.mp4`));
    expect((await fs.stat(path.join(evidence, `${name}.anonymous.mp4`))).size).toBeGreaterThan(100);
    const summary = { source: name, ticket, resultId, seconds: (Date.now() - started) / 1000, counts, http, browser: await page.evaluate(() => navigator.userAgent), result: await page.evaluate(() => (window as unknown as Record<string, unknown>).__T49_RESULT) };
    await fs.writeFile(path.join(evidence, `${name}.summary.json`), JSON.stringify(summary, null, 2));
    console.log('T49 COMPLETED', JSON.stringify(summary));
    await catalog.close();
  } finally { clearInterval(timer); }
});
