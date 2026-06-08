import React, { useState, useEffect } from 'react';
import './App.css';
import { Camera, Sliders, Image, Save, Zap, Menu, X, Clock, Terminal, Grid } from 'lucide-react';

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
    brightness: 128,
    contrast: 32,
    saturation: 64,
    gain: 0,
    exposure: 156,
    sharpness: 2,
    average: 1,
    auto_exposure: 0
  });
  const [status, setStatus] = useState<string>('Ready');
  const [logs, setLogs] = useState<string[]>([]);
  const [captures, setCaptures] = useState<string[]>([]);
  const [rigMode, setRigMode] = useState<string>('mock');
  const [motorStatus, setMotorStatus] = useState({ duty_cycle: 0, voltage: 0, mock_mode: true });
  const [isAdjustingMotor, setIsAdjustingMotor] = useState(false);
  const [sequenceStatus, setSequenceStatus] = useState({ active: false, count: 0, total: 0 });
  const [sequenceConfig, setSequenceConfig] = useState({ count: 10, interval: 2 });
  const [panoramaStatus, setPanoramaStatus] = useState({ active: false, current: 0, total: 0, progress: 0, offset_x: 0, offset_y: 0 });
  const [panoramaConfig, setPanoramaConfig] = useState({ frames: 20, drift_step: 15.0, auto_align: true });
  const logEndRef = React.useRef<HTMLDivElement>(null);
  const [health, setHealth] = useState<{connected: boolean, mean_brightness: number, last_frame_time: number, width: number, height: number, fps: number}>({
    connected: true,
    mean_brightness: 0,
    last_frame_time: 0,
    width: 1920,
    height: 1080,
    fps: 0
  });
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      fetch(`${API_BASE}/status`).then(res => res.json()).then(data => setHealth(data)).catch(() => {});
      fetch(`${API_BASE}/controls`).then(res => res.json()).then(data => setControls(data)).catch(() => {});
      fetch(`${API_BASE}/motor/status`).then(res => res.json()).then(data => { if (!isAdjustingMotor) setMotorStatus(data); }).catch(() => {});
      fetch(`${API_BASE}/sequence/status`).then(res => res.json()).then(data => setSequenceStatus(data)).catch(() => {});
      fetch(`${API_BASE}/panorama/status`).then(res => res.json()).then(data => setPanoramaStatus(data)).catch(() => {});
      fetch(`${API_BASE}/logs`).then(res => res.json()).then(data => setLogs(data)).catch(() => {});
      fetch(`${API_BASE}/captures/list`).then(res => res.json()).then(data => setCaptures(data)).catch(() => {});
      fetch(`${API_BASE}/rig`).then(res => res.json()).then(data => setRigMode(data.mode)).catch(() => {});
    }, 2000);

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
    });
  };

  const updateMotorSpeed = (speed: number) => {
    setIsAdjustingMotor(true);
    setMotorStatus(prev => ({ ...prev, duty_cycle: speed, voltage: (3.3 * speed) / 100 }));
    fetch(`${API_BASE}/motor/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed })
    });
    const timeoutId = (window as any).motorTimeout;
    if (timeoutId) clearTimeout(timeoutId);
    (window as any).motorTimeout = setTimeout(() => setIsAdjustingMotor(false), 2000);
  };

  const handleSwitchRig = (mode: string) => {
    fetch(`${API_BASE}/rig`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    }).then(res => res.json()).then(data => {
      if (data.success) setRigMode(data.mode);
    });
  };

  const handleCapture = () => {
    fetch(`${API_BASE}/capture`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setStatus(`Saved: ${data.filename}`);
          setTimeout(() => setStatus('Ready'), 3000);
        }
      });
  };

  const handleStartPanorama = () => {
    fetch(`${API_BASE}/panorama/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(panoramaConfig)
    });
  };

  return (
    <div className={`app-container ${isSidebarOpen ? 'sidebar-open' : ''}`}>
      <button className="mobile-toggle" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
        {isSidebarOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      <div className="main-view">
        <h1><Camera size={24} /> AstroCam</h1>
        <div className="stream-container">
          <img src={`${API_BASE}/stream`} alt="Live Stream" className="video-preview" />
        </div>
        <div className="health-bar">
          <div className="health-item" style={{ color: health.connected ? '#238636' : '#da3633' }}>
            ● {health.connected ? 'Connected' : 'Disconnected'}
          </div>
          <div className="health-item">Luminance: {health.mean_brightness.toFixed(1)}</div>
          <div className="health-item">FPS: {health.fps.toFixed(1)}</div>
        </div>

        <div className="layout-grid">
          <div className="log-container">
            <div className="log-header"><Terminal size={14} /> System Logs</div>
            <div className="log-window">
              {logs.map((log, i) => <div key={i} className="log-entry">{log}</div>)}
              <div ref={logEndRef} />
            </div>
          </div>

          <div className="captures-container">
            <div className="log-header"><Grid size={14} /> Recent Captures</div>
            <div className="captures-grid">
              {captures.length === 0 ? (
                <div className="empty-msg">No captures yet</div>
              ) : (
                captures.map(file => (
                  <div key={file} className="capture-item" onClick={() => window.open(`${API_BASE}/captures/${file}`, '_blank')}>
                    <img src={`${API_BASE}/captures/${file}`} alt={file} />
                    <div className="capture-label">{file.substring(0, 15)}...</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      <div className={`sidebar ${isSidebarOpen ? 'active' : ''}`}>
        <div className="sidebar-header"><h1><Sliders size={20} /> Controls</h1></div>

        <div className="control-section">
          <div className="section-header"><Zap size={16} /> Mount Control</div>
          <div className="control-group">
            <label>Speed (85% = Sidereal)</label>
            <div className="slider-container">
              <input type="range" min="0" max="100" step="0.2" value={motorStatus.duty_cycle} onChange={(e) => updateMotorSpeed(parseFloat(e.target.value))} />
              <div className="value-display">{motorStatus.duty_cycle.toFixed(1)}%</div>
            </div>
          </div>
        </div>

        <div className="control-section">
          <div className="section-header"><Image size={16} /> Panorama</div>
          {panoramaStatus.active ? (
            <div className="sequence-progress">
              <div className="progress-text">{panoramaStatus.current} / {panoramaStatus.total} (X:{panoramaStatus.offset_x})</div>
              <div className="progress-bar-bg">
                <div className="progress-bar-fill" style={{ width: `${panoramaStatus.progress}%`, backgroundColor: '#aa3bff' }}></div>
              </div>
            </div>
          ) : (
            <div className="sequence-form">
              <div className="input-row">
                <div className="input-group">
                  <label>Frames</label>
                  <input type="number" value={panoramaConfig.frames} onChange={(e) => setPanoramaConfig(prev => ({ ...prev, frames: parseInt(e.target.value) || 1 }))} />
                </div>
                <div className="input-group">
                  <label>Auto Align</label>
                  <input type="checkbox" checked={panoramaConfig.auto_align} onChange={(e) => setPanoramaConfig(prev => ({ ...prev, auto_align: e.target.checked }))} />
                </div>
              </div>
              <button className="start-seq-btn" style={{ backgroundColor: '#aa3bff' }} onClick={handleStartPanorama}><Zap size={16} /> Start Panorama</button>
            </div>
          )}
        </div>

        <div className="control-group" style={{ marginTop: '24px' }}>
          <label>Rig Engine</label>
          <div className="rig-toggle">
            <button className={rigMode === 'mock' ? 'active' : ''} onClick={() => handleSwitchRig('mock')}>Mock</button>
            <button className={rigMode === 'real' ? 'active' : ''} onClick={() => handleSwitchRig('real')}>Real</button>
          </div>
        </div>
        
        <div className="actions">
          <button onClick={handleCapture}><Save size={18} /> Capture Frame</button>
        </div>
      </div>
    </div>
  );
}

export default App;
