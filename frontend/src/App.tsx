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

interface TrackingStatus {
  active: boolean;
  drift_x: number;
  drift_y: number;
  drift_speed: number;
  camera_pa: number;
  sim_drift_speed: number | null;
  sim_drift_angle: number | null;
  sim_camera_angle: number | null;
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
  const [prevDuty, setPrevDuty] = useState<number>(85.0);
  const [trackingStatus, setTrackingStatus] = useState<TrackingStatus>({
    active: false,
    drift_x: 0,
    drift_y: 0,
    drift_speed: 0,
    camera_pa: 0,
    sim_drift_speed: null,
    sim_drift_angle: null,
    sim_camera_angle: null
  });
  const [panoramaStatus, setPanoramaStatus] = useState({ active: false, current: 0, total: 0, progress: 0, offset_x: 0, offset_y: 0, offset_angle: 0 });
  const [panoramaConfig, setPanoramaConfig] = useState({ frames: 20, drift_step: 15.0, auto_align: true });
  const [health, setHealth] = useState({
    connected: true,
    mean_brightness: 0,
    width: 1920,
    height: 1080,
    fps: 0
  });
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const logWindowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const hRes = await fetch(`${API_BASE}/status`);
        if (hRes.ok) setHealth(await hRes.json());

        const cRes = await fetch(`${API_BASE}/controls`);
        if (cRes.ok) setControls(await cRes.json());

        const mRes = await fetch(`${API_BASE}/motor/status`);
        if (mRes.ok && !isAdjustingMotor) {
          const data = await mRes.json();
          setMotorStatus(data);
          if (data.duty_cycle > 0) {
            setPrevDuty(data.duty_cycle);
          }
        }

        const pRes = await fetch(`${API_BASE}/panorama/status`);
        if (pRes.ok) setPanoramaStatus(await pRes.json());

        const lRes = await fetch(`${API_BASE}/logs`);
        if (lRes.ok) setLogs(await lRes.json());

        const capRes = await fetch(`${API_BASE}/gallery`);
        if (capRes.ok) setCaptures(await capRes.json());

        const rRes = await fetch(`${API_BASE}/rig`);
        if (rRes.ok) {
            const data = await rRes.json();
            setRigMode(data.mode);
        }

        const tRes = await fetch(`${API_BASE}/tracking/status`);
        if (tRes.ok) setTrackingStatus(await tRes.json());
      } catch (err) {
        console.error("Fetch error:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [isAdjustingMotor]);

  useEffect(() => {
    if (logWindowRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logWindowRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      if (isNearBottom) {
        logWindowRef.current.scrollTop = scrollHeight;
      }
    }
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
    if (speed > 0) {
      setPrevDuty(speed);
    }
    fetch(`${API_BASE}/motor/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed })
    }).catch(e => console.error(e));

    const timeoutId = (window as any).motorTimeout;
    if (timeoutId) clearTimeout(timeoutId);
    (window as any).motorTimeout = setTimeout(() => setIsAdjustingMotor(false), 2000);
  };

  const updateCameraAngle = (angle: number) => {
    setTrackingStatus(prev => ({ ...prev, sim_camera_angle: angle }));
    fetch(`${API_BASE}/mock/camera_angle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ angle })
    }).catch(e => console.error("Error setting camera angle:", e));
  };

  const updateSimDrift = (speed: number, angle: number) => {
    setTrackingStatus(prev => ({ ...prev, sim_drift_speed: speed, sim_drift_angle: angle }));
    fetch(`${API_BASE}/mock/sim_drift`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed, angle })
    }).catch(e => console.error("Error setting sim drift:", e));
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

  const handleStopPanorama = () => {
    setStatus('Stopping panorama...');
    fetch(`${API_BASE}/panorama/stop`, {
      method: 'POST'
    }).then(() => setStatus('Ready'))
      .catch(e => setStatus(`Error: ${e.message}`));
  };

  const setMountMode = (mode: 'off' | 'on' | 'auto') => {
    if (mode === 'auto') {
      const isAlreadyActive = trackingStatus.active;
      setStatus(isAlreadyActive ? 'Relocking tracking reference...' : 'Enabling auto-tracking...');
      fetch(`${API_BASE}/tracking/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable: true })
      }).then(res => res.json()).then(data => {
        if (data.success) {
          setTrackingStatus(prev => ({ ...prev, active: true }));
          setStatus(isAlreadyActive ? 'Reference frame relocked' : 'Auto-tracking enabled');
          setTimeout(() => setStatus('Ready'), 2000);
        }
      }).catch(e => setStatus(`Error: ${e.message}`));
    } else {
      const targetSpeed = mode === 'on' ? prevDuty : 0.0;
      if (trackingStatus.active) {
        setStatus('Disabling auto-tracking...');
        fetch(`${API_BASE}/tracking/toggle`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enable: false })
        }).then(res => res.json()).then(data => {
          if (data.success) {
            setTrackingStatus(prev => ({ ...prev, active: false }));
            updateMotorSpeed(targetSpeed);
            setStatus(mode === 'on' ? `Tracking restored to ${targetSpeed.toFixed(1)}%` : 'Mount stopped');
            setTimeout(() => setStatus('Ready'), 2000);
          }
        }).catch(e => setStatus(`Error: ${e.message}`));
      } else {
        updateMotorSpeed(targetSpeed);
      }
    }
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
            <div className="log-window" ref={logWindowRef}>
              {Array.isArray(logs) && logs.map((log, i) => <div key={i} className="log-entry">{log}</div>)}
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
                  <div key={file} className="capture-item" title={file} onClick={() => window.open(`${API_BASE}/captures/${file}`, '_blank')}>
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
            <div className="section-header"><Camera size={16} /> Engine</div>
            <div className="rig-toggle">
              <button className={rigMode === 'mock' ? 'active' : ''} onClick={() => handleSwitchRig('mock')}>Mock</button>
              <button className={rigMode === 'real' ? 'active' : ''} onClick={() => handleSwitchRig('real')}>Real</button>
            </div>
          </div>

          {rigMode === 'mock' && (
            <div className="control-section" style={{ marginTop: '-12px' }}>
              <div className="tracking-telemetry" style={{ marginTop: '0px', background: 'rgba(88, 166, 255, 0.05)', borderColor: 'rgba(88, 166, 255, 0.15)' }}>
                {trackingStatus.sim_drift_speed !== null && (
                  <div className="control-group" style={{ marginBottom: '8px' }}>
                    <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                      <span>Sim Drift Speed:</span>
                      <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--text-primary)' }}>{trackingStatus.sim_drift_speed.toFixed(1)} px/s</span>
                    </label>
                    <input 
                      type="range" 
                      min="0" 
                      max="100" 
                      step="0.5" 
                      value={trackingStatus.sim_drift_speed} 
                      onChange={e => updateSimDrift(parseFloat(e.target.value), trackingStatus.sim_drift_angle || 0)}
                      style={{ marginTop: '4px' }}
                    />
                  </div>
                )}
                {trackingStatus.sim_camera_angle !== null && (
                  <div className="control-group" style={{ marginBottom: '0px' }}>
                    <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                      <span>Diurnal Rot Angle (Camera PA):</span>
                      <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--text-primary)' }}>{trackingStatus.sim_camera_angle.toFixed(0)}°</span>
                    </label>
                    <input 
                      type="range" 
                      min="0" 
                      max="359" 
                      step="1" 
                      value={trackingStatus.sim_camera_angle} 
                      onChange={e => updateCameraAngle(parseInt(e.target.value))}
                      style={{ marginTop: '4px' }}
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="control-section">
            <div className="section-header"><Zap size={16} /> Mount</div>
            <div className="control-group">
              <label>Duty Cycle: {(motorStatus.duty_cycle || 0).toFixed(1)}%</label>
              <input type="range" min="0" max="100" step="0.2" value={motorStatus.duty_cycle || 0} onChange={(e) => updateMotorSpeed(parseFloat(e.target.value))} />
              <div className="preset-row">
                <button 
                  className={(!trackingStatus.active && (motorStatus.duty_cycle || 0) === 0) ? 'active' : ''} 
                  onClick={() => setMountMode('off')}
                >
                  OFF
                </button>
                <button 
                  className={(!trackingStatus.active && (motorStatus.duty_cycle || 0) > 0) ? 'active' : ''} 
                  onClick={() => setMountMode('on')}
                >
                  ON
                </button>
                <button 
                  className={trackingStatus.active ? 'active' : ''} 
                  onClick={() => setMountMode('auto')}
                >
                  AUTO
                </button>
              </div>

              <div className="tracking-telemetry">
                <div className="tracking-telemetry-row">
                  <span>Drift XY:</span>
                  <span>{trackingStatus.drift_x.toFixed(1)}px, {trackingStatus.drift_y.toFixed(1)}px</span>
                </div>
                <div className="tracking-telemetry-row">
                  <span>Drift Speed:</span>
                  <span>{trackingStatus.drift_speed.toFixed(3)} px/s</span>
                </div>
                <div className="tracking-telemetry-row">
                  <span>Camera PA:</span>
                  <span>{trackingStatus.camera_pa.toFixed(1)}°</span>
                </div>
              </div>
            </div>
          </div>

          <div className="control-section">
            <div className="section-header"><Image size={16} /> Panorama</div>
            {panoramaStatus.active ? (
              <div className="progress-container">
                <div className="progress-info">
                  <span>{panoramaStatus.current || 0}/{panoramaStatus.total || 0} frames</span>
                  <span>X: {(panoramaStatus.offset_x || 0).toFixed(0)}px, Rot: {(panoramaStatus.offset_angle || 0).toFixed(2)}°</span>
                </div>
                <div className="progress-bar-bg">
                  <div className="progress-bar-fill" style={{ width: `${panoramaStatus.progress || 0}%` }}></div>
                </div>
                <button 
                  className="btn-primary danger" 
                  onClick={handleStopPanorama} 
                  style={{ marginTop: '10px' }}
                >
                  Stop Panorama
                </button>
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

        </div>
        
        <div className="sidebar-footer">
          <button className="btn-capture" onClick={handleCapture}><Save size={18} /> Take Photo</button>
        </div>
      </aside>
    </div>
  );
}

export default App;
