import React, { useState, useEffect } from 'react';
import './App.css';
import { Camera, Sliders, Image, Save, Zap, Menu, X, Clock, Terminal } from 'lucide-react';

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
  const [rigMode, setRigMode] = useState<string>('mock');
  const [motorStatus, setMotorStatus] = useState({ duty_cycle: 0, voltage: 0, mock_mode: true });
  const [isAdjustingMotor, setIsAdjustingMotor] = useState(false);
  const [sequenceStatus, setSequenceStatus] = useState({ active: false, count: 0, total: 0 });
  const [sequenceConfig, setSequenceConfig] = useState({ count: 10, interval: 2 });
  const [panoramaStatus, setPanoramaStatus] = useState({ active: false, current: 0, total: 0, progress: 0 });
  const [panoramaConfig, setPanoramaConfig] = useState({ frames: 20, drift_step: 15.0 });
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
    // Single polling loop for all camera data
    const interval = setInterval(() => {
      fetch(`${API_BASE}/status`)
        .then(res => res.json())
        .then(data => setHealth(data))
        .catch(() => setHealth(prev => ({ ...prev, connected: false, fps: 0 })));

      fetch(`${API_BASE}/controls`)
        .then(res => res.json())
        .then(data => setControls(data))
        .catch(() => {});

      fetch(`${API_BASE}/motor/status`)
        .then(res => res.json())
        .then(data => {
          if (!isAdjustingMotor) {
            setMotorStatus(data);
          }
        })
        .catch(() => {});

      fetch(`${API_BASE}/sequence/status`)
        .then(res => res.json())
        .then(data => setSequenceStatus(data))
        .catch(() => {});

      fetch(`${API_BASE}/panorama/status`)
        .then(res => res.json())
        .then(data => setPanoramaStatus(data))
        .catch(() => {});

      fetch(`${API_BASE}/logs`)
        .then(res => res.json())
        .then(data => setLogs(data))
        .catch(() => {});

      fetch(`${API_BASE}/rig`)
        .then(res => res.json())
        .then(data => setRigMode(data.mode))
        .catch(() => {});
    }, 2000);

    // Initial fetch
    fetch(`${API_BASE}/controls`)
      .then(res => res.json())
      .then(data => setControls(data))
      .catch(() => setStatus('Backend not reachable'));

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
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

    // Re-enable polling after 2 seconds of inactivity
    const timeoutId = (window as any).motorTimeout;
    if (timeoutId) clearTimeout(timeoutId);
    (window as any).motorTimeout = setTimeout(() => {
      setIsAdjustingMotor(false);
    }, 2000);
  };

  const handleSwitchRig = (mode: string) => {
    setStatus(`Switching to ${mode} rig...`);
    fetch(`${API_BASE}/rig`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    }).then(res => res.json())
    .then(data => {
      if (data.success) {
        setRigMode(data.mode);
        setStatus(`Rig set to ${data.mode}`);
      }
    });
  };

  const handleResolutionChange = (width: number, height: number) => {
    setStatus(`Switching to ${width}x${height}...`);
    fetch(`${API_BASE}/resolution`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ width, height })
    }).then(() => {
      setTimeout(() => setStatus('Ready'), 2000);
    });
  };

  const handleCapture = () => {
    setStatus('Capturing...');
    fetch(`${API_BASE}/capture`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setStatus(`Saved: ${data.filename}`);
          setTimeout(() => setStatus('Ready'), 3000);
        } else {
          setStatus(`Error: ${data.error}`);
        }
      });
  };

  const handleStartSequence = () => {
    setStatus(`Starting sequence (${sequenceConfig.count} frames)...`);
    fetch(`${API_BASE}/sequence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sequenceConfig)
    }).then(res => res.json())
    .then(data => {
      if (data.success) {
        setStatus('Sequence active');
      } else {
        setStatus('Failed to start sequence');
      }
    });
  };

  const handleStartPanorama = () => {
    setStatus(`Starting Panorama (${panoramaConfig.frames} frames)...`);
    fetch(`${API_BASE}/panorama/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(panoramaConfig)
    }).then(res => res.json())
    .then(data => {
      if (data.success) {
        setStatus('Panorama construction active');
      } else {
        setStatus('Failed to start panorama');
      }
    });
  };

  const handleStopPanorama = () => {
    fetch(`${API_BASE}/panorama/stop`, { method: 'POST' });
  };

  return (
    <div className={`app-container ${isSidebarOpen ? 'sidebar-open' : ''}`}>
      <button className="mobile-toggle" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
        {isSidebarOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      <div className="main-view">
        <h1><Camera size={24} /> AstroCam SV205</h1>
        <div className="stream-container">
          <img 
            src={`${API_BASE}/stream`} 
            alt="Live Stream" 
            className="video-preview" 
          />
        </div>
        <div className="health-bar">
          <div className="health-item" style={{ color: health.connected ? '#238636' : '#da3633' }}>
            ● {health.connected ? 'Connected' : 'Disconnected'}
          </div>
          <div className="health-item">
            Luminance: {health.mean_brightness.toFixed(1)}
          </div>
          <div className="health-item">
            FPS: {health.fps.toFixed(1)}
          </div>
        </div>

        <div className="log-container">
          <div className="log-header">
            <Terminal size={14} /> System Logs
          </div>
          <div className="log-window">
            {logs.length === 0 ? (
              <div className="log-entry empty">System ready. No logs yet.</div>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="log-entry">{log}</div>
              ))
            )}
            <div ref={logEndRef} />
          </div>
        </div>

        {status && <div className="status-message">{status}</div>}
      </div>

      <div className={`sidebar ${isSidebarOpen ? 'active' : ''}`}>
        <div className="sidebar-header">
          <h1><Sliders size={20} /> Controls</h1>
        </div>

        <div className="control-section">
          <div className="section-header">
            <Zap size={16} /> Mount Control {motorStatus.mock_mode && <span className="badge">MOCK</span>}
          </div>
          
          <div className="control-group">
            <label>Tracking Speed (0.2% steps)</label>
            <div className="slider-container">
              <input 
                type="range" 
                min="0" 
                max="100" 
                step="0.2"
                value={motorStatus.duty_cycle} 
                onChange={(e) => updateMotorSpeed(parseFloat(e.target.value))}
              />
              <div className="value-display">{motorStatus.duty_cycle.toFixed(1)}%</div>
            </div>
            <div className="voltage-info">Approx. {motorStatus.voltage.toFixed(2)}V</div>
          </div>

          <div className="preset-row">
            <button className="preset-btn" onClick={() => updateMotorSpeed(85)}>Sidereal</button>
            <button className="preset-btn" onClick={() => updateMotorSpeed(Math.min(100, motorStatus.duty_cycle + 0.2))}>Drift +</button>
            <button className="preset-btn" onClick={() => updateMotorSpeed(Math.max(0, motorStatus.duty_cycle - 0.2))}>Drift -</button>
          </div>
        </div>

        <div className="control-section">
          <div className="section-header">
            <Clock size={16} /> Sequence Capture
          </div>

          {sequenceStatus.active ? (
            <div className="sequence-progress">
              <div className="progress-text">Capturing: {sequenceStatus.count} / {sequenceStatus.total}</div>
              <div className="progress-bar-bg">
                <div 
                  className="progress-bar-fill" 
                  style={{ width: `${(sequenceStatus.count / sequenceStatus.total) * 100}%` }}
                ></div>
              </div>
            </div>
          ) : (
            <div className="sequence-form">
              <div className="input-row">
                <div className="input-group">
                  <label>Frames</label>
                  <input 
                    type="number" 
                    value={sequenceConfig.count} 
                    onChange={(e) => setSequenceConfig(prev => ({ ...prev, count: parseInt(e.target.value) || 1 }))}
                  />
                </div>
                <div className="input-group">
                  <label>Interval (s)</label>
                  <input 
                    type="number" 
                    value={sequenceConfig.interval} 
                    onChange={(e) => setSequenceConfig(prev => ({ ...prev, interval: parseFloat(e.target.value) || 1 }))}
                  />
                </div>
              </div>
              <button className="start-seq-btn" onClick={handleStartSequence}>
                <Image size={16} /> Start Sequence
              </button>
            </div>
          )}
        </div>

        <div className="control-section">
          <div className="section-header">
            <Image size={16} /> Panorama Construction
          </div>

          {panoramaStatus.active ? (
            <div className="sequence-progress">
              <div className="progress-text">Processing: {panoramaStatus.current} / {panoramaStatus.total}</div>
              <div className="progress-bar-bg">
                <div 
                  className="progress-bar-fill" 
                  style={{ width: `${panoramaStatus.progress}%`, backgroundColor: '#aa3bff' }}
                ></div>
              </div>
              <button className="preset-btn" style={{ marginTop: '12px', width: '100%' }} onClick={handleStopPanorama}>
                Stop Panorama
              </button>
            </div>
          ) : (
            <div className="sequence-form">
              <div className="input-row">
                <div className="input-group">
                  <label>Total Frames</label>
                  <input 
                    type="number" 
                    value={panoramaConfig.frames} 
                    onChange={(e) => setPanoramaConfig(prev => ({ ...prev, frames: parseInt(e.target.value) || 1 }))}
                  />
                </div>
                <div className="input-group">
                  <label>Drift Step (px)</label>
                  <input 
                    type="number" 
                    value={panoramaConfig.drift_step} 
                    onChange={(e) => setPanoramaConfig(prev => ({ ...prev, drift_step: parseFloat(e.target.value) || 1 }))}
                  />
                </div>
              </div>
              <button className="start-seq-btn" style={{ backgroundColor: '#aa3bff' }} onClick={handleStartPanorama}>
                <Zap size={16} /> Start Panorama
              </button>
            </div>
          )}
        </div>
        
        {Object.entries(controls)
          .filter(([key]) => key !== 'auto_exposure' && (key !== 'exposure' || controls.auto_exposure === 0))
          .map(([key, value]) => (
            <div key={key} className="control-group">
              <label>
                {key === 'average' ? 'Average (N frames)' : key.charAt(0).toUpperCase() + key.slice(1)}
              </label>
              <div className="slider-container">
                <input 
                  type="range" 
                  min={key === 'average' ? '1' : '0'} 
                  max={key === 'exposure' ? '1000' : (key === 'average' ? '100' : '255')} 
                  value={value} 
                  onChange={(e) => updateControl(key, parseInt(e.target.value))}
                />
                <div className="value-display">{Math.round(value)}</div>
              </div>
            </div>
          ))}

        <div className="actions">
          <button onClick={handleCapture}>
            <Image size={18} /> Capture Frame
          </button>

          <button 
            className="secondary" 
            onClick={() => updateControl('auto_exposure', controls.auto_exposure === 1 ? 0 : 1)}
          >
            <Zap size={18} /> {controls.auto_exposure === 1 ? 'Switch to Manual Exp' : 'Switch to Auto Exp'}
          </button>
        </div>

        <div className="control-group" style={{ marginTop: '24px', borderTop: '1px solid #30363d', paddingTop: '16px' }}>
          <label>Resolution</label>
          <select 
            className="resolution-select"
            value={`${health.width}x${health.height}`}
            onChange={(e) => {
              const [w, h] = e.target.value.split('x').map(Number);
              handleResolutionChange(w, h);
            }}
          >
            {![ '3264x2448', '1920x1080', '1280x720', '640x480' ].includes(`${health.width}x${health.height}`) && (
              <option value={`${health.width}x${health.height}`}>{health.width}x{health.height} (Detected)</option>
            )}
            <option value="3264x2448">3264x2448 (8MP)</option>
            <option value="1920x1080">1920x1080 (1080p)</option>
            <option value="1280x720">1280x720 (720p)</option>
            <option value="640x480">640x480 (VGA)</option>
          </select>
          </div>

          <div className="control-group" style={{ marginTop: '12px' }}>
          <label>Rig Engine</label>
          <div className="rig-toggle">
            <button 
              className={rigMode === 'mock' ? 'active' : ''} 
              onClick={() => handleSwitchRig('mock')}
            >Mock</button>
            <button 
              className={rigMode === 'real' ? 'active' : ''} 
              onClick={() => handleSwitchRig('real')}
            >Real</button>
          </div>
          </div>
          </div>
          </div>
  );
}

export default App;
