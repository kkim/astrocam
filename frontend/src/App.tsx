import { useState, useEffect, useRef } from 'react';
import './App.css';
import { Camera, Sliders, Image, Save, Zap, Menu, X, Terminal, Grid, RefreshCw } from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:8000`;

interface Controls {
  brightness: number;
  contrast: number;
  saturation: number;
  gain: number;
  exposure: number;
  sharpness: number;
  average: number;
  auto_exposure: number;
}

function App() {
  const [controls, setControls] = useState<Controls>({
    brightness: 128, contrast: 32, saturation: 64, gain: 0,
    exposure: 156, sharpness: 2, average: 1, auto_exposure: 0
  });
  const [status, setStatus] = useState<string>('Ready');
  const [logs, setLogs] = useState<string[]>([]);
  const [captures, setCaptures] = useState<string[]>([]);
  const [rigMode, setRigMode] = useState<string>('mock');
  const [motorStatus, setMotorStatus] = useState({ duty_cycle: 0, voltage: 0, mock_mode: true });
  const [isAdjustingMotor, setIsAdjustingMotor] = useState(false);
  const [panoramaStatus, setPanoramaStatus] = useState({ active: false, current: 0, total: 0, progress: 0, offset_x: 0, offset_y: 0 });
  const [panoramaConfig, setPanoramaConfig] = useState({ frames: 20, drift_step: 15.0, auto_align: true });
  const logEndRef = useRef<HTMLDivElement>(null);
  const [health, setHealth] = useState({
    connected: true,
    mean_brightness: 0,
    width: 1920,
    height: 1080,
    fps: 0
  });
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const hRes = await fetch(`${API_BASE}/status`);
        if (hRes.ok) setHealth(await hRes.json());

        const cRes = await fetch(`${API_BASE}/controls`);
        if (cRes.ok) setControls(await cRes.json());

        const mRes = await fetch(`${API_BASE}/motor/status`);
        if (mRes.ok && !isAdjustingMotor) setMotorStatus(await mRes.json());

        const pRes = await fetch(`${API_BASE}/panorama/status`);
        if (pRes.ok) setPanoramaStatus(await pRes.json());

        const lRes = await fetch(`${API_BASE}/logs`);
        if (lRes.ok) setLogs(await lRes.json());

        const capRes = await fetch(`${API_BASE}/captures/list`);
        if (capRes.ok) setCaptures(await capRes.json());

        const rRes = await fetch(`${API_BASE}/rig`);
        if (rRes.ok) {
            const data = await rRes.json();
            setRigMode(data.mode);
        }
      } catch (err) {
        console.error("Fetch error:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [isAdjustingMotor]);

  useEffect(() => {
    if (logEndRef.current) logEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const updateControl = (prop: string, val: number) => {
    setControls(prev => ({ ...prev, [prop]: val }));
    fetch(`${API_BASE}/controls`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ property: prop, value: val })
    }).catch(e => console.error(e));
  };

  const updateMotorSpeed = (speed: number) => {
    setIsAdjustingMotor(true);
    setMotorStatus(prev => ({ ...prev, duty_cycle: speed, voltage: (3.3 * speed) / 100 }));
    fetch(`${API_BASE}/motor/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed })
    }).catch(e => console.error(e));

    const timeoutId = (window as any).motorTimeout;
    if (timeoutId) clearTimeout(timeoutId);
    (window as any).motorTimeout = setTimeout(() => setIsAdjustingMotor(false), 2000);
  };

  const handleSwitchRig = (mode: string) => {
    setStatus(`Switching to ${mode}...`);
    fetch(`${API_BASE}/rig`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    }).then(res => res.json()).then(data => {
      if (data.success) {
        setRigMode(data.mode);
        setStatus(`Rig set to ${data.mode}`);
      }
    }).catch(e => setStatus(`Error: ${e.message}`));
  };

  const handleCapture = () => {
    setStatus('Capturing frame...');
    fetch(`${API_BASE}/capture`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setStatus(`Saved: ${data.filename}`);
          setTimeout(() => setStatus('Ready'), 3000);
        }
      }).catch(e => setStatus(`Error: ${e.message}`));
  };

  const handleStartPanorama = () => {
    setStatus('Starting panorama...');
    fetch(`${API_BASE}/panorama/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(panoramaConfig)
    }).catch(e => setStatus(`Error: ${e.message}`));
  };

  return (
    <div className={`app-container ${isSidebarOpen ? 'sidebar-open' : ''}`}>
      <button className="mobile-toggle" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
        {isSidebarOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      <div className="main-view">
        <header className="main-header">
          <h1><Camera size={24} /> AstroCam Rig</h1>
          <div className="status-badge" style={{ color: health.connected ? '#238636' : '#da3633' }}>
            ● {health.connected ? 'Connected' : 'Disconnected'}
          </div>
        </header>

        <div className="stream-container">
          <img src={`${API_BASE}/stream`} alt="Live Stream" className="video-preview" />
          <div className="stream-overlay">
            <span>{health.width || 0}x{health.height || 0} @ {(health.fps || 0).toFixed(1)} FPS</span>
            <span>Luminance: {(health.mean_brightness || 0).toFixed(1)}</span>
          </div>
        </div>

        <div className="layout-grid">
          <section className="log-container">
            <div className="panel-header"><Terminal size={14} /> System Logs</div>
            <div className="log-window">
              {Array.isArray(logs) && logs.map((log, i) => <div key={i} className="log-entry">{log}</div>)}
              <div ref={logEndRef} />
            </div>
          </section>

          <section className="captures-container">
            <div className="panel-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Grid size={14} /> Gallery</div>
              <span className="count-badge">{Array.isArray(captures) ? captures.length : 0}</span>
            </div>
            <div className="captures-grid">
              {!Array.isArray(captures) || captures.length === 0 ? (
                <div className="empty-msg">No images captured</div>
              ) : (
                captures.map(file => (
                  <div key={file} className="capture-item" onClick={() => window.open(`${API_BASE}/captures/${file}`, '_blank')}>
                    <img src={`${API_BASE}/captures/${file}`} alt={file} loading="lazy" />
                    <div className="capture-label">{file}</div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>

        {status !== 'Ready' && <div className="status-toast">{status}</div>}
      </div>

      <aside className={`sidebar ${isSidebarOpen ? 'active' : ''}`}>
        <div className="sidebar-header">
          <h2><Sliders size={20} /> Controls</h2>
        </div>

        <div className="sidebar-scroll">
          <div className="control-section">
            <div className="section-header"><Zap size={16} /> Mount</div>
            <div className="control-group">
              <label>Duty Cycle: {(motorStatus.duty_cycle || 0).toFixed(1)}%</label>
              <input type="range" min="0" max="100" step="0.2" value={motorStatus.duty_cycle || 0} onChange={(e) => updateMotorSpeed(parseFloat(e.target.value))} />
              <div className="preset-row">
                <button onClick={() => updateMotorSpeed(85.0)}>Sidereal</button>
                <button onClick={() => updateMotorSpeed(0)}>Stop</button>
              </div>
            </div>
          </div>

          <div className="control-section">
            <div className="section-header"><Image size={16} /> Panorama</div>
            {panoramaStatus.active ? (
              <div className="progress-container">
                <div className="progress-info">
                  <span>{panoramaStatus.current || 0}/{panoramaStatus.total || 0} frames</span>
                  <span>Shift: {(panoramaStatus.offset_x || 0).toFixed(0)}px</span>
                </div>
                <div className="progress-bar-bg">
                  <div className="progress-bar-fill" style={{ width: `${panoramaStatus.progress || 0}%` }}></div>
                </div>
              </div>
            ) : (
              <div className="panorama-config">
                <div className="config-row">
                  <div className="field">
                    <label>Frames</label>
                    <input type="number" value={panoramaConfig.frames} onChange={e => setPanoramaConfig(p => ({...p, frames: parseInt(e.target.value) || 1}))} />
                  </div>
                  <div className="field">
                    <label>Auto-Align</label>
                    <input type="checkbox" checked={panoramaConfig.auto_align} onChange={e => setPanoramaConfig(p => ({...p, auto_align: e.target.checked}))} />
                  </div>
                </div>
                <button className="btn-primary purple" onClick={handleStartPanorama}>Start Panorama</button>
              </div>
            )}
          </div>

          <div className="control-section">
            <div className="section-header"><RefreshCw size={16} /> Camera Settings</div>
            {Object.entries(controls || {}).map(([key, value]) => (
              <div key={key} className="control-group">
                <label>{key}: {value}</label>
                <input 
                  type="range" 
                  min={key === 'average' ? 1 : 0} 
                  max={key === 'exposure' ? 1000 : 255} 
                  value={value || 0} 
                  onChange={e => updateControl(key, parseInt(e.target.value))} 
                />
              </div>
            ))}
          </div>

          <div className="control-section">
            <div className="section-header"><Camera size={16} /> Engine</div>
            <div className="rig-toggle">
              <button className={rigMode === 'mock' ? 'active' : ''} onClick={() => handleSwitchRig('mock')}>Mock</button>
              <button className={rigMode === 'real' ? 'active' : ''} onClick={() => handleSwitchRig('real')}>Real</button>
            </div>
          </div>
        </div>
        
        <div className="sidebar-footer">
          <button className="btn-capture" onClick={handleCapture}><Save size={18} /> Take Photo</button>
        </div>
      </aside>
    </div>
  );
}

export default App;
