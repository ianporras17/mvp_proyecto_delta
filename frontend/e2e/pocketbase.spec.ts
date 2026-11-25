import { expect, test } from '@playwright/test';
import { credentials } from './auth';

test('PocketBase aplica los permisos en el servidor y rechaza registro público y elevación de rol', async ({ request }) => {
  const url = process.env.E2E_POCKETBASE_URL ?? 'http://127.0.0.1:8090';
  const users = `${url}/api/collections/users`;
  const auth = async (role: 'admin' | 'user') => {
    const [identity, password] = credentials(role);
    const response = await request.post(`${users}/auth-with-password`, { data: { identity, password } });
    expect(response.ok()).toBe(true);
    return response.json();
  };
  const admin = await auth('admin'), user = await auth('user');
  const account = { name: 'Registro no autorizado', email: `denied.${Date.now()}@example.test`, password: 'UnaClave123!', passwordConfirm: 'UnaClave123!', role: 'admin' };
  expect((await request.post(`${users}/records`, { data: account })).ok()).toBe(false);
  expect((await request.post(`${users}/records`, { headers: { Authorization: user.token }, data: account })).ok()).toBe(false);
  expect((await request.patch(`${users}/records/${user.record.id}`, { headers: { Authorization: user.token }, data: { role: 'admin' } })).ok()).toBe(false);
  expect((await request.get(`${users}/records/${admin.record.id}`, { headers: { Authorization: user.token } })).status()).toBe(404);
  const userList = await request.get(`${users}/records`, { headers: { Authorization: user.token } });
  expect((await userList.json()).items).toEqual([]);
  const adminList = await request.get(`${users}/records`, { headers: { Authorization: admin.token } });
  expect(adminList.ok()).toBe(true);
  expect((await adminList.json()).items).toEqual(expect.arrayContaining([expect.objectContaining({ id: user.record.id, email: user.record.email })]));
});
