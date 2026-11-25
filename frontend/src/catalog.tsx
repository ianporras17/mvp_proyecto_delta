import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { config } from './browser/config';
import './styles.css';
type Result = { id: string; filename: string; size: number; created_at: string; download_url: string; counts: { people?: number; cars?: number } };
function Catalog() {
  const [results, setResults] = useState<Result[]>([]), [error, setError] = useState('');
  async function refresh() {
    try {
      const response = await fetch(`${config.storageUrl}/api/v1/results`);
      if (!response.ok) throw new Error(`Almacenamiento: HTTP ${response.status}`);
      setResults(await response.json()); setError('');
    } catch (e) { setError(e instanceof Error ? e.message : 'No se pudo consultar el almacenamiento.'); }
  }
  useEffect(() => { void refresh(); }, []);
  return <main style={{ maxWidth: 960, margin: '40px auto', padding: 24 }}><h1>Videos anonimizados</h1><p>Resultados disponibles para descargar.</p><button className="button button-primary" onClick={() => void refresh()}>Actualizar listado</button>{error && <p role="alert">{error}</p>}<ul style={{ padding: 0, listStyle: 'none' }}>{results.map(result => <li className="upload-card" key={result.id} style={{ marginTop: 16, padding: 24 }}><strong>{new Date(result.created_at).toLocaleString('es-CR')}</strong><p>{(result.size / 1e6).toFixed(2)} MB · Personas: {result.counts.people ?? 0} · Carros: {result.counts.cars ?? 0}</p><a className="button button-primary" href={`${config.storageUrl}${result.download_url}`}>Descargar video anonimizado</a></li>)}</ul>{!results.length && !error && <p>No hay resultados guardados.</p>}</main>;
}
createRoot(document.getElementById('root')!).render(<Catalog />);
