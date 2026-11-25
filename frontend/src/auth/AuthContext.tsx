import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { pocketbaseAuth } from '../services/pocketbase-auth';
import type { CreateUserInput, AppUser } from '../types';

interface AuthState {
  user: AppUser | null;
  users: AppUser[];
  loading: boolean;
  startupError: string | null;
  login: (email: string, password: string) => Promise<AppUser>;
  logout: () => void;
  createUser: (input: CreateUserInput) => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AppUser | null>(null);
  const [users, setUsers] = useState<AppUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [startupError, setStartupError] = useState<string | null>(null);

  const refreshUsers = useCallback(async (current: AppUser | null) => {
    if (current?.role === 'admin') setUsers(await pocketbaseAuth.listUsers());
    else setUsers([]);
  }, []);

  useEffect(() => {
    let active = true;
    pocketbaseAuth.initialize().then(async (current) => {
      if (!active) return;
      setUser(current);
      await refreshUsers(current);
    }).catch((error: unknown) => {
      if (active) setStartupError(error instanceof Error ? error.message : 'No pudimos conectar con PocketBase.');
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [refreshUsers]);

  const login = useCallback(async (email: string, password: string) => {
    const current = await pocketbaseAuth.login(email, password);
    setUser(current);
    await refreshUsers(current);
    return current;
  }, [refreshUsers]);

  const logout = useCallback(() => {
    pocketbaseAuth.logout();
    setUser(null);
    setUsers([]);
  }, []);

  const createUser = useCallback(async (input: CreateUserInput) => {
    await pocketbaseAuth.createUser(input);
    await refreshUsers(user);
  }, [refreshUsers, user]);

  const value = useMemo(() => ({ user, users, loading, startupError, login, logout, createUser }),
    [user, users, loading, startupError, login, logout, createUser]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth debe usarse dentro de AuthProvider.');
  return context;
}
