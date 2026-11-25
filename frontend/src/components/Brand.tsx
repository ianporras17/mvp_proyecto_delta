export function Brand({ compact = false }: { compact?: boolean }) {
  return <div className={`brand${compact ? ' brand--compact' : ''}`}>
    <svg className="brand-mark" viewBox="0 0 48 48" aria-hidden="true">
      <path d="m24 7 15 9v16l-15 9-15-9V16Z" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
      <path d="m9 16 15 9 15-9M24 25v16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
      <circle cx="24" cy="7" r="3.5" fill="currentColor" /><circle cx="9" cy="32" r="3.5" fill="currentColor" /><circle cx="39" cy="32" r="3.5" fill="currentColor" />
    </svg>
    <div><span className="brand-name">TecPeople<span>Flow</span></span><span className="brand-caption">PRIVACIDAD EN MOVIMIENTO</span></div>
  </div>;
}
