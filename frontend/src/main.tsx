import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import 'maplibre-gl/dist/maplibre-gl.css';
import './styles.css';
import App from './App';
import { ErrorBoundary } from './ui';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary fallback={(e) => (
      <div className="crash" role="alert">
        <h1>Something broke</h1>
        <p>{e.message}</p>
        <button className="retry" onClick={() => window.location.reload()}>Reload</button>
      </div>
    )}>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
