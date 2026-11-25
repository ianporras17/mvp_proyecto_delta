import { ArrowRight, Eye, EyeOff, KeyRound, LockKeyhole, Mail, Monitor, ShieldCheck } from 'lucide-react';
import { useState, type FormEvent } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { Brand } from '../components/Brand';

export default function LoginPage() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  if (user) return <Navigate to="/dashboard" replace />;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) return;
    if (!email.trim() || !password) { setError('Ingresa tu correo y contraseña para continuar.'); return; }
    setError(''); setBusy(true);
    try { await login(email, password); navigate('/dashboard', { replace: true }); }
    catch (reason: unknown) { setError(reason instanceof Error ? reason.message : 'No pudimos iniciar sesión. Intenta de nuevo.'); }
    finally { setBusy(false); }
  }

  return <main className="login-page">
    <section className="login-story" aria-labelledby="login-story-title">
      <Brand />
      <div className="login-story-copy"><span className="eyebrow light"><span />PROYECTO DELTA</span></div>
      <div className="privacy-illustration" aria-hidden="true">
        <div className="illustration-grid" />
        <div className="illustration-orbit orbit-one" /><div className="illustration-orbit orbit-two" />
        <div className="video-art"><div className="video-art-toolbar"><span /><span /><span /><small>PRIVACIDAD POR DISEÑO</small></div><div className="art-scene"><div className="art-building building-one" /><div className="art-building building-two" /><div className="art-building building-three" /><div className="art-person person-one"><i /><b /></div><div className="art-person person-two"><i /><b /></div><div className="art-car"><i /><b /></div><div className="art-road" /><span className="art-corner corner-one" /><span className="art-corner corner-two" /></div><div className="art-caption"><ShieldCheck size={17} /><span>Tecnología al servicio de las personas</span><span className="art-led" /></div></div>
        <div className="art-security-badge"><ShieldCheck size={23} /><div><strong>Privacidad en cada cuadro</strong><small>Rostros y placas, en el centro del cuidado.</small></div></div>
        <span className="art-node node-one" /><span className="art-node node-two" /><span className="art-node node-three" />
      </div>
      <div className="login-story-footer"><span>TecPeopleFlow · {new Date().getFullYear()}</span></div>
    </section>
    <section className="login-panel" aria-labelledby="login-title">
      <div className="login-form-wrap"><span className="login-key"><KeyRound size={27} /></span><h2 id="login-title">Ingresa con las credenciales asignadas para acceder a tu espacio de trabajo.</h2>
        <form onSubmit={submit} noValidate>
          <div className="form-field"><label htmlFor="login-email">Correo electrónico</label><div className="input-with-icon"><Mail size={18} /><input id="login-email" type="email" autoComplete="username" placeholder="tu.nombre@ejemplo.com" value={email} onChange={event => setEmail(event.target.value)} required disabled={busy} aria-describedby={error ? 'login-error' : undefined} /></div></div>
          <div className="form-field"><label htmlFor="login-password">Contraseña</label><div className="input-with-icon password-input"><LockKeyhole size={18} /><input id="login-password" type={showPassword ? 'text' : 'password'} autoComplete="current-password" placeholder="Ingresa tu contraseña" value={password} onChange={event => setPassword(event.target.value)} required disabled={busy} aria-describedby={error ? 'login-error' : undefined} /><button type="button" className="icon-button" aria-label={showPassword ? 'Ocultar contraseña' : 'Mostrar contraseña'} aria-pressed={showPassword} onClick={() => setShowPassword(!showPassword)}>{showPassword ? <EyeOff size={18} /> : <Eye size={18} />}</button></div></div>
          {error && <p id="login-error" className="form-error" role="alert">{error}</p>}
          <button type="submit" className="button button-primary login-submit" disabled={busy}>{busy ? 'Verificando acceso…' : 'Iniciar sesión'}<ArrowRight size={18} /></button>
        </form>
        <p className="access-help">¿Necesitas acceso? Contacta a tu administrador.<br />Las cuentas se crean únicamente desde administración.</p>
      </div>
      <footer className="login-panel-footer">Privacidad en movimiento. Personas primero.</footer>
    </section>
  </main>;
}
