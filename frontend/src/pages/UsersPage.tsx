import { useId, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import {
  ArrowUpRight,
  CheckCircle2,
  Eye,
  EyeOff,
  LoaderCircle,
  Plus,
  Search,
  ShieldCheck,
  UserRound,
  UsersRound,
  X,
} from 'lucide-react';
import type { CreateUserInput, AppUser, UserRole } from '../types';
import './users.css';

interface UsersPageProps {
  users: AppUser[];
  onCreateUser: (input: CreateUserInput) => Promise<void>;
}

type FieldName = 'name' | 'email' | 'password';
type FieldErrors = Partial<Record<FieldName, string>>;

const dateFormatter = new Intl.DateTimeFormat('es-CR', {
  day: 'numeric',
  month: 'short',
  year: 'numeric',
});

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : dateFormatter.format(date);
}

function initials(name: string) {
  return name.trim().split(/\s+/).slice(0, 2).map((part) => part[0]).join('').toUpperCase();
}

export default function UsersPage({ users, onCreateUser }: UsersPageProps) {
  const ids = useId();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const createButtonRef = useRef<HTMLButtonElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const emailRef = useRef<HTMLInputElement>(null);
  const passwordRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState('');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<UserRole>('user');
  const [showPassword, setShowPassword] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [submitError, setSubmitError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');

  const normalizedQuery = query.trim().toLocaleLowerCase('es');
  const filteredUsers = users.filter((user) =>
    `${user.name} ${user.email}`.toLocaleLowerCase('es').includes(normalizedQuery),
  );

  function openCreateDialog() {
    previousFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setName('');
    setEmail('');
    setPassword('');
    setRole('user');
    setShowPassword(false);
    setFieldErrors({});
    setSubmitError('');
    dialogRef.current?.showModal();
  }

  function closeDialog() {
    if (!isSubmitting) dialogRef.current?.close();
  }

  function clearFieldError(field: FieldName) {
    setFieldErrors((previous) => ({ ...previous, [field]: undefined }));
    setSubmitError('');
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmitting) return;

    const errors: FieldErrors = {};
    if (name.trim().length < 2) errors.name = 'Escribe un nombre de al menos 2 caracteres.';
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
      errors.email = 'Escribe un correo electrónico válido.';
    }
    if (password.length < 8) errors.password = 'La contraseña debe tener al menos 8 caracteres.';
    setFieldErrors(errors);
    setSubmitError('');

    if (Object.keys(errors).length > 0) {
      const firstInvalidField = errors.name ? nameRef : errors.email ? emailRef : passwordRef;
      firstInvalidField.current?.focus();
      return;
    }

    setIsSubmitting(true);
    try {
      await onCreateUser({ name: name.trim(), email: email.trim(), password, role });
      setSuccessMessage(`La cuenta de ${name.trim()} se creó correctamente.`);
      setPassword('');
      dialogRef.current?.close();
    } catch (error) {
      setSubmitError(error instanceof Error && error.message ? error.message : 'No se pudo crear el usuario. Inténtalo de nuevo.');
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="users-page" aria-labelledby={`${ids}-page-title`}>
      <header className="users-page__header">
        <div>
          <p className="users-page__eyebrow"><span aria-hidden="true" />Administración</p>
          <h1 id={`${ids}-page-title`}>Personas y accesos</h1>
        </div>
        <button ref={createButtonRef} type="button" className="users-page__primary-button" onClick={openCreateDialog}>
          <Plus size={18} aria-hidden="true" />Crear usuario
        </button>
      </header>

      <div className="users-page__notice-region" aria-live="polite" aria-atomic="true">
        {successMessage && (
          <div className="users-page__success">
            <CheckCircle2 size={19} aria-hidden="true" />
            <span>{successMessage}</span>
            <button type="button" aria-label="Cerrar confirmación" onClick={() => setSuccessMessage('')}>
              <X size={17} aria-hidden="true" />
            </button>
          </div>
        )}
      </div>

      <div className="users-page__card">
        <div className="users-page__toolbar">
          <div className="users-page__list-heading">
            <span className="users-page__section-icon"><UsersRound size={20} aria-hidden="true" /></span>
            <div>
              <h2>Usuarios del sistema <span className="users-page__count">{users.length}</span></h2>
              <p>Consulta sus datos y permisos de acceso.</p>
            </div>
          </div>
          <div className="users-page__search">
            <Search size={18} aria-hidden="true" />
            <label className="users-page__sr-only" htmlFor={`${ids}-search`}>Buscar por nombre o correo</label>
            <input
              id={`${ids}-search`}
              type="search"
              placeholder="Buscar nombre o correo…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
        </div>

        {filteredUsers.length > 0 ? (
          <div className="users-page__table-scroll" tabIndex={0} role="region" aria-label="Lista de usuarios; desplaza horizontalmente para ver todas las columnas">
            <table className="users-page__table">
              <caption className="users-page__sr-only">Usuarios registrados en Proyecto Delta</caption>
              <thead>
                <tr><th scope="col">Nombre</th><th scope="col">Correo electrónico</th><th scope="col">Rol</th><th scope="col">Fecha de creación</th></tr>
              </thead>
              <tbody>
                {filteredUsers.map((user) => (
                  <tr key={user.id}>
                    <td>
                      <div className="users-page__person">
                        <span className="users-page__avatar" aria-hidden="true">{initials(user.name)}</span>
                        <span>{user.name}</span>
                      </div>
                    </td>
                    <td className="users-page__email">{user.email}</td>
                    <td>
                      <span className={`users-page__role users-page__role--${user.role}`}>
                        {user.role === 'admin' ? <ShieldCheck size={14} aria-hidden="true" /> : <UserRound size={14} aria-hidden="true" />}
                        {user.role === 'admin' ? 'Administrador' : 'Usuario'}
                      </span>
                    </td>
                    <td className="users-page__date"><time dateTime={user.createdAt}>{formatDate(user.createdAt)}</time></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="users-page__empty">
            <span className="users-page__empty-icon"><UsersRound size={27} aria-hidden="true" /></span>
            <h3>{normalizedQuery ? 'No encontramos coincidencias' : 'Todo comienza con tu equipo'}</h3>
            <p>{normalizedQuery ? 'Prueba con otro nombre o correo electrónico.' : 'Crea el primer usuario para darle acceso a Proyecto Delta.'}</p>
            <button type="button" className="users-page__text-button" onClick={normalizedQuery ? () => setQuery('') : openCreateDialog}>
              {normalizedQuery ? 'Limpiar búsqueda' : 'Crear un usuario'}<ArrowUpRight size={16} aria-hidden="true" />
            </button>
          </div>
        )}

        <footer className="users-page__table-footer" aria-live="polite" aria-atomic="true">
          <span>{normalizedQuery ? `${filteredUsers.length} de ${users.length} usuarios` : `${users.length} ${users.length === 1 ? 'usuario registrado' : 'usuarios registrados'}`}</span>
          <span className="users-page__access-note"><ShieldCheck size={14} aria-hidden="true" />Acceso administrado</span>
        </footer>
      </div>
      <p className="users-page__permissions-note"><ShieldCheck size={16} aria-hidden="true" />Solo los administradores pueden crear usuarios y asignar sus roles.</p>

      <dialog
        ref={dialogRef}
        className="user-dialog"
        aria-labelledby={`${ids}-dialog-title`}
        aria-describedby={`${ids}-dialog-description`}
        onClose={() => (previousFocusRef.current ?? createButtonRef.current)?.focus()}
        onCancel={(event) => { if (isSubmitting) event.preventDefault(); }}
      >
        <div className="user-dialog__heading">
          <span className="user-dialog__icon"><UsersRound size={23} aria-hidden="true" /></span>
          <button type="button" className="user-dialog__close" aria-label="Cerrar ventana de crear usuario" onClick={closeDialog} disabled={isSubmitting}>
            <X size={21} aria-hidden="true" />
          </button>
        </div>
        <h2 id={`${ids}-dialog-title`}>Crear usuario</h2>
        <p id={`${ids}-dialog-description`} className="user-dialog__description">Completa sus datos y define cómo podrá acceder al sistema.</p>

        <form noValidate onSubmit={handleSubmit} aria-busy={isSubmitting}>
          <div className="user-dialog__field">
            <label htmlFor={`${ids}-name`}>Nombre completo</label>
            <input
              ref={nameRef} id={`${ids}-name`} name="name" autoComplete="name" autoFocus required
              placeholder="Ej. Andrea Rodríguez" value={name} disabled={isSubmitting}
              aria-invalid={Boolean(fieldErrors.name)} aria-describedby={fieldErrors.name ? `${ids}-name-error` : undefined}
              onChange={(event) => { setName(event.target.value); clearFieldError('name'); }}
            />
            {fieldErrors.name && <p id={`${ids}-name-error`} className="user-dialog__field-error">{fieldErrors.name}</p>}
          </div>
          <div className="user-dialog__field">
            <label htmlFor={`${ids}-email`}>Correo electrónico</label>
            <input
              ref={emailRef} id={`${ids}-email`} name="email" type="email" autoComplete="email" required
              placeholder="nombre@ejemplo.com" value={email} disabled={isSubmitting}
              aria-invalid={Boolean(fieldErrors.email)} aria-describedby={fieldErrors.email ? `${ids}-email-error` : undefined}
              onChange={(event) => { setEmail(event.target.value); clearFieldError('email'); }}
            />
            {fieldErrors.email && <p id={`${ids}-email-error`} className="user-dialog__field-error">{fieldErrors.email}</p>}
          </div>
          <div className="user-dialog__field">
            <label htmlFor={`${ids}-password`}>Contraseña</label>
            <div className="user-dialog__password">
              <input
                ref={passwordRef} id={`${ids}-password`} name="password" type={showPassword ? 'text' : 'password'}
                autoComplete="new-password" required minLength={8} placeholder="Mínimo 8 caracteres"
                value={password} disabled={isSubmitting} aria-invalid={Boolean(fieldErrors.password)}
                aria-describedby={`${ids}-password-hint${fieldErrors.password ? ` ${ids}-password-error` : ''}`}
                onChange={(event) => { setPassword(event.target.value); clearFieldError('password'); }}
              />
              <button type="button" aria-label={showPassword ? 'Ocultar contraseña' : 'Mostrar contraseña'} aria-pressed={showPassword} onClick={() => setShowPassword(!showPassword)} disabled={isSubmitting}>
                {showPassword ? <EyeOff size={19} aria-hidden="true" /> : <Eye size={19} aria-hidden="true" />}
              </button>
            </div>
            <p id={`${ids}-password-hint`} className="user-dialog__hint">Comparte esta contraseña con la persona de forma privada.</p>
            {fieldErrors.password && <p id={`${ids}-password-error`} className="user-dialog__field-error">{fieldErrors.password}</p>}
          </div>
          <div className="user-dialog__field">
            <label htmlFor={`${ids}-role`}>Rol de acceso</label>
            <select id={`${ids}-role`} name="role" value={role} onChange={(event) => setRole(event.target.value as UserRole)} disabled={isSubmitting} aria-describedby={`${ids}-role-hint`}>
              <option value="user">Usuario</option>
              <option value="admin">Administrador</option>
            </select>
            <p id={`${ids}-role-hint`} className="user-dialog__hint">{role === 'admin' ? 'Podrá crear usuarios y gestionar los accesos del sistema.' : 'Tendrá acceso a su espacio personal en el sistema.'}</p>
          </div>
          {submitError && <p className="user-dialog__submit-error" role="alert">{submitError}</p>}
          <div className="user-dialog__actions">
            <button type="button" className="user-dialog__cancel" onClick={closeDialog} disabled={isSubmitting}>Cancelar</button>
            <button type="submit" className="user-dialog__submit" disabled={isSubmitting}>
              {isSubmitting ? <LoaderCircle className="user-dialog__spinner" size={17} aria-hidden="true" /> : <Plus size={17} aria-hidden="true" />}
              {isSubmitting ? 'Creando usuario…' : 'Crear usuario'}
            </button>
          </div>
        </form>
      </dialog>
    </section>
  );
}
