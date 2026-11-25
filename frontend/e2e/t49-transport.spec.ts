import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
test.use({ trace: 'off' });
test('T49 three independent party channels, consecutive chunks, padding and masked inputs', async ({ page }) => {
  test.skip(process.env.RUN_T49_TRANSPORT !== '1');
  test.setTimeout(21_600_000);
  await page.goto('/catalog.html');
  const result = await page.evaluate(() => new Promise<{ status: string; channels: { privatePackets: number; publicMasks: number; receivedShares: number }[] }>((resolve, reject) => {
    const worker = new Worker('/e2e/transport.worker.ts', { type: 'module' });
    worker.onerror = e => { worker.terminate(); reject(new Error(e.message)); };
    worker.onmessage = ({ data }) => { worker.terminate(); resolve(data); };
    worker.postMessage({});
  }));
  expect(result.status, JSON.stringify(result)).toBe('complete');
  expect(result.channels).toHaveLength(3);
  for (const channel of result.channels) expect(channel).toEqual({ privatePackets: 3, publicMasks: 3, receivedShares: 6 });
  await fs.mkdir(path.resolve('../.runtime/t49-evidence'), { recursive: true });
  await fs.writeFile(path.resolve('../.runtime/t49-evidence/transport.json'), JSON.stringify(result, null, 2));
});
