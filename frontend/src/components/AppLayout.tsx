import { ChevronRight, CircleHelp, LayoutDashboard, LogOut, ShieldCheck, Users } from 'lucide-react';
import { NavLink, Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { Brand } from './Brand';

export function AppLayout() {
  const { user, logout } = useAuth();
  const location = useLocation();
  if (!user) return <Navigate to="/login" replace />;
  const initials = user.name.trim().split(/\s+/).slice(0, 2).map(part => part[0]).join('').toUpperCase();
  return <div className="app-shell">
    <a className="skip-link" href="#contenido" onClick={event => { event.preventDefault(); const main = document.getElementById('contenido'); main?.focus(); main?.scrollIntoView(); }}>Saltar al contenido</a>
    <aside className="sidebar" aria-label="Barra lateral">
      <Brand compact />
      <div className="workspace-label">ESPACIO DE TRABAJO</div>
      <nav className="main-nav" aria-label="Navegación principal">
        <NavLink to="/dashboard"><LayoutDashboard size={19} /><span>Mi espacio</span><ChevronRight size={15} className="nav-chevron" /></NavLink>
        {user.role === 'admin' && <NavLink to="/admin/users"><Users size={19} /><span>Usuarios</span><ChevronRight size={15} className="nav-chevron" /></NavLink>}
      </nav>
      <div className="sidebar-bottom">
        <button className="logout-button" onClick={logout}><LogOut size={17} />Cerrar sesión</button>
      </div>
    </aside>
    <div className="app-body">
      <header className="topbar">
        <div className="breadcrumbs"><span>Área de trabajo</span><ChevronRight size={14} /><strong>{location.pathname.startsWith('/admin') ? 'Usuarios' : 'Video'}</strong></div>
        <div className="topbar-right"><div className="profile"><span className="avatar">{initials}</span><span><strong>{user.name}</strong><small>{user.role === 'admin' ? 'Administrador' : 'Usuario'}</small></span></div></div>
      </header>
      <main id="contenido" className="main-content" tabIndex={-1}><Outlet /></main>
      <footer className="app-footer"><span>© {new Date().getFullYear()} TecPeopleFlow <span className="footer-dot">·</span> Proyecto Delta</span><span><ShieldCheck size={14} />Diseñado con la privacidad en mente <CircleHelp size={14} /></span></footer>
    </div>
  </div>;
}
