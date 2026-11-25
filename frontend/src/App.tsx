import { LoaderCircle, ShieldCheck } from 'lucide-react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { useAuth } from './auth/AuthContext';
import { AppLayout } from './components/AppLayout';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import UsersPage from './pages/UsersPage';

export default function App() {
  const { user, users, loading, startupError, createUser } = useAuth();
  if (loading) return <div className="startup-screen" role="status"><LoaderCircle className="spin" size={30} /><p>Preparando tu espacio…</p></div>;
  if (startupError) return <main className="startup-screen"><ShieldCheck size={35} /><h1>No pudimos conectar con Proyecto Delta</h1><p role="alert">{startupError}</p><p>Comprueba que PocketBase esté activo y abre la aplicación desde localhost.</p><button className="button button-primary" onClick={() => window.location.reload()}>Intentar de nuevo</button></main>;
  return <Routes>
    <Route path="/login" element={<LoginPage />} />
    <Route element={<AppLayout />}>
      <Route path="/dashboard" element={<DashboardPage />} />
      <Route path="/admin/users" element={user?.role === 'admin' ? <UsersPage users={users} onCreateUser={createUser} /> : <Navigate to={user ? '/dashboard' : '/login'} replace />} />
    </Route>
    <Route path="*" element={<Navigate to={user ? '/dashboard' : '/login'} replace />} />
  </Routes>;
}
