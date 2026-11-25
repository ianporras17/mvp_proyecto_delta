import { expect, type Page } from '@playwright/test';

export function credentials(role: 'admin' | 'user'): [string, string] {
  const prefix = `E2E_${role.toUpperCase()}`;
  const email = process.env[`${prefix}_EMAIL`], password = process.env[`${prefix}_PASSWORD`];
  if (!email || !password) throw new Error(`Configura ${prefix}_EMAIL y ${prefix}_PASSWORD con cuentas reales de PocketBase.`);
  return [email, password];
}

export async function login(page: Page, role: 'admin' | 'user' = 'user') {
  await loginWithCredentials(page, ...credentials(role));
}

export async function loginWithCredentials(page: Page, email: string, password: string) {
  await page.goto('/#/login');
  await page.getByLabel('Correo electrónico', { exact: true }).fill(email);
  await page.getByLabel('Contraseña', { exact: true }).fill(password);
  await page.getByRole('button', { name: 'Iniciar sesión', exact: true }).click();
  await expect(page).toHaveURL(/#\/dashboard$/);
}
