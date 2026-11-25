import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { RecordModel } from 'pocketbase';

const { pb, collection } = vi.hoisted(() => {
  const collection = {
    authRefresh: vi.fn(),
    authWithPassword: vi.fn(),
    getFullList: vi.fn(),
    create: vi.fn(),
  };
  const pb = {
    authStore: { isValid: false, record: null as RecordModel | null, clear: vi.fn() },
    collection: vi.fn(() => collection),
  };
  return { pb, collection };
});

vi.mock('./pocketbase', () => ({ pb }));
import { pocketbaseAuth, toAppUser } from './pocketbase-auth';

const record = {
  id: 'abc123', collectionId: 'users_collection', collectionName: 'users',
  name: 'Ana Pérez', email: 'ana@example.test', role: 'admin', created: '2026-11-16 10:00:00.000Z',
} as RecordModel;

beforeEach(() => {
  vi.clearAllMocks();
  pb.authStore.isValid = false;
  pb.authStore.record = null;
});

describe('adaptador PocketBase', () => {
  it('convierte el registro auth al usuario que consume la interfaz', () => {
    expect(toAppUser(record)).toEqual({
      id: 'abc123', name: 'Ana Pérez', email: 'ana@example.test', role: 'admin', createdAt: record.created,
    });
    expect(() => toAppUser({ ...record, role: 'superadmin' } as RecordModel)).toThrow('rol válido');
  });

  it('autentica por correo y devuelve el rol guardado por PocketBase', async () => {
    collection.authWithPassword.mockResolvedValue({ record });
    await expect(pocketbaseAuth.login(' ana@example.test ', 'Password123!')).resolves.toMatchObject({
      email: 'ana@example.test', role: 'admin',
    });
    expect(pb.collection).toHaveBeenCalledWith('users');
    expect(collection.authWithPassword).toHaveBeenCalledWith('ana@example.test', 'Password123!');
  });

  it('restaura y actualiza el registro de una sesión válida', async () => {
    pb.authStore.isValid = true;
    pb.authStore.record = record;
    collection.authRefresh.mockResolvedValue({ record });
    await expect(pocketbaseAuth.initialize()).resolves.toMatchObject({ id: record.id, role: 'admin' });
    expect(collection.authRefresh).toHaveBeenCalledOnce();
  });

  it('limpia una sesión expirada', async () => {
    pb.authStore.isValid = true;
    pb.authStore.record = record;
    collection.authRefresh.mockRejectedValue(new Error('expired'));
    await expect(pocketbaseAuth.initialize()).resolves.toBeNull();
    expect(pb.authStore.clear).toHaveBeenCalledOnce();
  });

  it('lista cuentas y crea una cuenta con confirmación de contraseña y rol', async () => {
    collection.getFullList.mockResolvedValue([record]);
    collection.create.mockResolvedValue(record);
    await expect(pocketbaseAuth.listUsers()).resolves.toHaveLength(1);
    await pocketbaseAuth.createUser({
      name: ' Ana Pérez ', email: ' ANA@example.test ', password: 'Password123!', role: 'admin',
    });
    expect(collection.create).toHaveBeenCalledWith({
      name: 'Ana Pérez', email: 'ana@example.test', password: 'Password123!',
      passwordConfirm: 'Password123!', role: 'admin',
    });
  });

  it('valida datos antes de enviarlos al servicio', async () => {
    await expect(pocketbaseAuth.createUser({ name: 'A', email: 'no válido', password: 'corta', role: 'user' }))
      .rejects.toThrow('nombre');
    expect(collection.create).not.toHaveBeenCalled();
  });
});
