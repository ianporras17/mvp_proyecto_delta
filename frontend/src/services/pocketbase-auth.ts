import type { RecordModel } from 'pocketbase';
import { pb } from './pocketbase';
import type { CreateUserInput, AppUser, UserRole } from '../types';

const USERS_COLLECTION = 'users';

function isRole(role: unknown): role is UserRole {
  return role === 'admin' || role === 'user';
}

export function toAppUser(record: RecordModel): AppUser {
  if (typeof record.name !== 'string' || !isRole(record.role)) {
    throw new Error('La cuenta no tiene un nombre o rol válido. Contacta al administrador.');
  }
  return { id: record.id, name: record.name, email: record.email, role: record.role, createdAt: record.created };
}

export const pocketbaseAuth = {
  async initialize(): Promise<AppUser | null> {
    if (!pb.authStore.isValid || !pb.authStore.record) {
      pb.authStore.clear();
      return null;
    }
    try {
      const response = await pb.collection(USERS_COLLECTION).authRefresh();
      return toAppUser(response.record);
    } catch {
      pb.authStore.clear();
      return null;
    }
  },

  async login(email: string, password: string): Promise<AppUser> {
    const response = await pb.collection(USERS_COLLECTION).authWithPassword(email.trim(), password);
    try {
      return toAppUser(response.record);
    } catch (error) {
      pb.authStore.clear();
      throw error;
    }
  },

  logout(): void {
    pb.authStore.clear();
  },

  async listUsers(): Promise<AppUser[]> {
    const records = await pb.collection(USERS_COLLECTION).getFullList({ sort: '-created' });
    return records.map(toAppUser);
  },

  async createUser(input: CreateUserInput): Promise<AppUser> {
    const name = input.name.trim();
    const email = input.email.trim().toLowerCase();
    if (name.length < 2 || name.length > 80) throw new Error('El nombre debe tener entre 2 y 80 caracteres.');
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) throw new Error('Escribe un correo electrónico válido.');
    if (input.password.length < 8 || input.password.length > 128) throw new Error('La contraseña debe tener entre 8 y 128 caracteres.');
    if (!isRole(input.role)) throw new Error('Selecciona un rol válido: administrador o usuario.');

    const record = await pb.collection(USERS_COLLECTION).create({
      name, email, password: input.password, passwordConfirm: input.password, role: input.role,
    });
    return toAppUser(record);
  },
};
