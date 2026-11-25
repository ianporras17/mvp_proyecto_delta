import { expect, test } from '@playwright/test';
import path from 'node:path';
import { login } from './auth';

test('cancelar en la web libera las parties y promueve el siguiente trabajo en cola', async ({ page, request }) => {
  test.skip(process.env.RUN_WEB_MPC_E2E !== '1', 'Requires the real Docker services');
  test.setTimeout(90_000);
  const root = process.env.E2E_COORDINATOR_URL ?? 'http://localhost:8000/api/v1/browser-jobs';
  let firstId: string | undefined, secondId: string | undefined;
  await login(page);
  try {
    const registration = page.waitForResponse(response => response.url().endsWith('/api/v1/browser-jobs') && response.status() === 202);
    await page.getByLabel('Seleccionar archivo MP4', { exact: true }).setInputFiles(path.resolve('e2e/fixtures/sample.mp4'));
    firstId = (await (await registration).json()).id;
    const queued = await request.post(root, { data: { filename: 'queued-after-cancellation.mp4', format: 'mp4' } });
    expect(queued.status()).toBe(202);
    const second = await queued.json();
    secondId = second.id;
    expect(second.status).toBe('QUEUED');
    expect(second.position).toBe(1);
    await page.getByRole('button', { name: 'Cancelar procesamiento', exact: true }).click();
    await expect(page.getByRole('alert')).toHaveText('Procesamiento cancelado.');
    await expect.poll(async () => (await (await request.get(`${root}/${firstId}`)).json()).status, { timeout: 45_000 }).toBe('FAILED');
    await expect.poll(async () => (await (await request.get(`${root}/${secondId}`)).json()).status).toBe('READY');
    await page.getByRole('button', { name: 'Quitar video', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Seleccionar video', exact: true })).toBeEnabled();
  } finally {
    for (const id of [firstId, secondId]) if (id) await request.delete(`${root}/${id}`);
  }
});
