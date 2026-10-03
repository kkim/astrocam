import { useState, useEffect, useRef } from 'react';
import './App.css';
import { Camera, Sliders, Zap, Menu, X, Terminal, Grid, RefreshCw, Compass, Circle, Square } from 'lucide-react';

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
  drift_speed_x: number;
  drift_speed_y: number;
  drift_speed: number;
  camera_pa: number;
  inlier_ratio: number;
  sim_drift_speed: number | null;
  sim_drift_angle: number | null;
  sim_camera_angle: number | null;
}

interface RecordingStatus {
  is_recording: boolean;
  frames: number;
  raw_frames?: number;
  accumulated_frames?: number;
  accumulation_ms?: number;
  format?: string;
  bytes_stored: number;
  mb_stored: number;
  directory: string;
  duration_sec: number;
  fps: number;
  dropped_frames?: number;
}

function App() {
  const [controls, setControls] = useState<Controls>({
    brightness: 128, contrast: 32, saturation: 64, gain: 0,
    exposure: 156, sharpness: 2, average: 1, auto_exposure: 0
  });
  const [status, setStatus] = useState<string>('Ready');
  const [logs, setLogs] = useState<string[]>([]);
  const [captures, setCaptures] = useState<string[]>([]);
  const [cameraEngine, setCameraEngine] = useState<string>('mock');
  const [mountEngine, setMountEngine] = useState<string>('mock');
  const [isReconnecting, setIsReconnecting] = useState<boolean>(false);
  const [accumulationMs, setAccumulationMs] = useState<number>(0);
  const [recordFormat, setRecordFormat] = useState<'tif' | 'jpg'>('tif');
  const [recordingStatus, setRecordingStatus] = useState<RecordingStatus>({
    is_recording: false,
    frames: 0,
    raw_frames: 0,
    accumulated_frames: 1,
    accumulation_ms: 0,
    format: 'tif',
    bytes_stored: 0,
    mb_stored: 0,
    directory: '',
    duration_sec: 0,
    fps: 0
  });
  const [motorStatus, setMotorStatus] = useState({ duty_cycle: 0, voltage: 0, mock_mode: true });
  const [isAdjustingMotor, setIsAdjustingMotor] = useState(false);
  const [prevDuty, setPrevDuty] = useState<number>(85.0);
  const [trackingStatus, setTrackingStatus] = useState<TrackingStatus>({
    active: false,
    drift_speed_x: 0,
    drift_speed_y: 0,
    drift_speed: 0,
    camera_pa: 0,
    inlier_ratio: 0,
    sim_drift_speed: null,
    sim_drift_angle: null,
    sim_camera_angle: null
  });
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
        if (hRes.ok) {
          const hData = await hRes.json();
          setHealth(hData);
          if (hData.camera_mode) setCameraEngine(hData.camera_mode);
          if (hData.mount_mode) setMountEngine(hData.mount_mode);
          if (hData.recording) setRecordingStatus(hData.recording);
        }

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

        const lRes = await fetch(`${API_BASE}/logs`);
        if (lRes.ok) setLogs(await lRes.json());

        const capRes = await fetch(`${API_BASE}/gallery`);
        if (capRes.ok) setCaptures(await capRes.json());

        const rRes = await fetch(`${API_BASE}/rig`);
        if (rRes.ok) {
            const data = await rRes.json();
            if (data.camera_mode) setCameraEngine(data.camera_mode);
            if (data.mount_mode) setMountEngine(data.mount_mode);
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
    if (!recordingStatus.is_recording) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/recording/status`);
        if (res.ok) {
          const data = await res.json();
          setRecordingStatus(data);
        }
      } catch (err) {
        console.error("Recording status poll error:", err);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [recordingStatus.is_recording]);

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

  const showToast = (msg: string, durationMs: number = 2500) => {
    setStatus(msg);
    setTimeout(() => {
      setStatus(prev => (prev === msg ? 'Ready' : prev));
    }, durationMs);
  };

  const handleSwitchCamera = (mode: 'mock' | 'real') => {
    setStatus(`Switching camera to ${mode}...`);
    fetch(`${API_BASE}/rig`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_mode: mode })
    }).then(res => res.json()).then(data => {
      if (data.success) {
        setCameraEngine(data.camera_mode);
        showToast(`Camera set to ${data.camera_mode}`);
      }
    }).catch(e => showToast(`Error: ${e.message}`, 4000));
  };

  const handleSwitchMount = (mode: 'mock' | 'real') => {
    setStatus(`Switching mount to ${mode}...`);
    fetch(`${API_BASE}/rig`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mount_mode: mode })
    }).then(res => res.json()).then(data => {
      if (data.success) {
        setMountEngine(data.mount_mode);
        showToast(`Mount set to ${data.mount_mode}`);
      }
    }).catch(e => showToast(`Error: ${e.message}`, 4000));
  };

  const handleReconnectCamera = () => {
    setIsReconnecting(true);
    setStatus('Reconnecting camera hardware...');
    fetch(`${API_BASE}/camera/reconnect`, {
      method: 'POST'
    }).then(res => res.json()).then(data => {
      if (data.success) {
        showToast('Camera reconnected successfully!');
      } else {
        showToast('Camera reconnect failed: check hardware connection', 4000);
      }
    }).catch(e => {
      showToast(`Reconnect error: ${e.message}`, 4000);
    }).finally(() => {
      setIsReconnecting(false);
    });
  };

  const handleToggleRecording = () => {
    if (recordingStatus.is_recording) {
      setStatus('Stopping recording...');
      fetch(`${API_BASE}/recording/stop`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
          if (data.success) {
            setRecordingStatus(prev => ({
              ...prev,
              is_recording: false,
              frames: data.frames,
              raw_frames: data.raw_frames,
              accumulated_frames: data.accumulated_frames,
              accumulation_ms: data.accumulation_ms,
              format: data.format,
              mb_stored: data.mb_stored,
              directory: data.directory
            }));
            const accMsg = data.accumulation_ms > 0 ? ` (~${data.accumulated_frames} frames/stack)` : '';
            showToast(`Recording stopped: ${data.frames} files saved${accMsg} (${data.mb_stored} MB)`, 3500);
          } else {
            showToast(`Error: ${data.error || 'Failed to stop recording'}`, 4000);
          }
        })
        .catch(e => showToast(`Error: ${e.message}`, 4000));
    } else {
      setStatus('Starting stream recording...');
      fetch(`${API_BASE}/recording/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          accumulation_ms: accumulationMs,
          format: recordFormat
        })
      })
        .then(res => res.json())
        .then(data => {
          if (data.success) {
            setRecordingStatus(prev => ({
              ...prev,
              is_recording: true,
              frames: 0,
              raw_frames: 0,
              accumulated_frames: 1,
              accumulation_ms: data.accumulation_ms ?? accumulationMs,
              format: data.format ?? recordFormat,
              mb_stored: 0,
              directory: data.directory,
              duration_sec: 0
            }));
            showToast(`Recording started in captures/${data.directory}/`, 3000);
          } else {
            showToast(`Error: ${data.error || 'Failed to start recording'}`, 4000);
          }
        })
        .catch(e => showToast(`Error: ${e.message}`, 4000));
    }
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
          <h1><Camera size={24} /> AstroCam Rig <span style={{ fontSize: '10px', color: '#8b949e', verticalAlign: 'middle' }}>v0.1.2</span></h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div className="status-badge" style={{ color: health.connected ? '#238636' : '#da3633' }}>
              ● {health.connected ? 'Connected' : 'Disconnected'}
            </div>
            {cameraEngine === 'real' && (
              <button 
                className="reconnect-btn" 
                onClick={handleReconnectCamera} 
                disabled={isReconnecting}
                title="Scan and reconnect camera hardware"
              >
                <RefreshCw size={12} className={isReconnecting ? 'spin' : ''} />
                Reconnect
              </button>
            )}
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
            <div className="section-header"><Camera size={16} /> Camera Engine</div>
            <div className="rig-toggle">
              <button className={cameraEngine === 'mock' ? 'active' : ''} onClick={() => handleSwitchCamera('mock')}>Mock</button>
              <button className={cameraEngine === 'real' ? 'active' : ''} onClick={() => handleSwitchCamera('real')}>Real</button>
            </div>
          </div>

          <div className="control-section" style={{ marginTop: '-12px' }}>
            <div className="section-header"><Compass size={16} /> Mount Engine</div>
            <div className="rig-toggle">
              <button className={mountEngine === 'mock' ? 'active' : ''} onClick={() => handleSwitchMount('mock')}>Mock</button>
              <button className={mountEngine === 'real' ? 'active' : ''} onClick={() => handleSwitchMount('real')}>Real</button>
            </div>
          </div>

          {cameraEngine === 'mock' && (
            <div className="control-section" style={{ marginTop: '-12px' }}>
              <div className="tracking-telemetry" style={{ marginTop: '0px', background: 'rgba(88, 166, 255, 0.05)', borderColor: 'rgba(88, 166, 255, 0.15)' }}>
                {trackingStatus.sim_drift_speed != null && (
                  <div className="control-group" style={{ marginBottom: '8px' }}>
                    <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                      <span>Sim Drift Speed:</span>
                      <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--text-primary)' }}>{(trackingStatus.sim_drift_speed ?? 0).toFixed(1)} px/s</span>
                    </label>
                    <input 
                      type="range" 
                      min="0" 
                      max="100" 
                      step="0.5" 
                      value={trackingStatus.sim_drift_speed ?? 0} 
                      onChange={e => updateSimDrift(parseFloat(e.target.value), trackingStatus.sim_drift_angle || 0)}
                      style={{ marginTop: '4px' }}
                    />
                  </div>
                )}
                {trackingStatus.sim_camera_angle != null && (
                  <div className="control-group" style={{ marginBottom: '0px' }}>
                    <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                      <span>Diurnal Rot Angle (Camera PA):</span>
                      <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--text-primary)' }}>{(trackingStatus.sim_camera_angle ?? 0).toFixed(0)}°</span>
                    </label>
                    <input 
                      type="range" 
                      min="0" 
                      max="359" 
                      step="1" 
                      value={trackingStatus.sim_camera_angle ?? 0} 
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
                  <span>Drift Speed XY:</span>
                  <span>{(trackingStatus.drift_speed_x ?? 0).toFixed(1)}, {(trackingStatus.drift_speed_y ?? 0).toFixed(1)}px/s</span>
                </div>
                <div className="tracking-telemetry-row">
                  <span>Drift Speed:</span>
                  <span>{(trackingStatus.drift_speed ?? 0).toFixed(3)} px/s</span>
                </div>
                <div className="tracking-telemetry-row">
                  <span>Camera PA:</span>
                  <span>{(trackingStatus.camera_pa ?? 0).toFixed(1)}°</span>
                </div>
                <div className="tracking-telemetry-row">
                  <span>Match Quality:</span>
                  <span>{((trackingStatus.inlier_ratio ?? 0) * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="control-section">
            <div className="section-header"><Zap size={16} /> Stream Recording</div>
            <div className="control-group">
              <label>Accumulation Time</label>
              <div className="preset-row">
                <button 
                  className={accumulationMs === 0 ? 'active' : ''} 
                  onClick={() => setAccumulationMs(0)}
                  disabled={recordingStatus.is_recording}
                  title="Raw: Record every raw frame at full camera frame rate"
                >
                  0ms (Raw)
                </button>
                <button 
                  className={accumulationMs === 200 ? 'active' : ''} 
                  onClick={() => setAccumulationMs(200)}
                  disabled={recordingStatus.is_recording}
                  title="5 FPS: Accumulate frames over 200ms in RAM (~5 FPS stacked output)"
                >
                  200ms (5 FPS)
                </button>
                <button 
                  className={accumulationMs === 1000 ? 'active' : ''} 
                  onClick={() => setAccumulationMs(1000)}
                  disabled={recordingStatus.is_recording}
                  title="1 FPS: Accumulate frames over 1000ms in RAM (~1 FPS stacked output)"
                >
                  1000ms (1 FPS)
                </button>
              </div>
            </div>

            <div className="control-group" style={{ marginTop: '8px' }}>
              <label>Format</label>
              <div className="preset-row">
                <button 
                  className={recordFormat === 'tif' ? 'active' : ''} 
                  onClick={() => setRecordFormat('tif')}
                  disabled={recordingStatus.is_recording}
                  title="Lossless uncompressed TIFF (ideal for stacking in Siril/DSS)"
                >
                  TIFF (Lossless)
                </button>
                <button 
                  className={recordFormat === 'jpg' ? 'active' : ''} 
                  onClick={() => setRecordFormat('jpg')}
                  disabled={recordingStatus.is_recording}
                  title="High quality JPEG"
                >
                  JPEG
                </button>
              </div>
            </div>
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
          {recordingStatus.is_recording ? (
            <div className="recording-panel">
              <div className="recording-header">
                <span className="rec-badge">
                  <span className="rec-dot"></span> REC {(recordingStatus.accumulation_ms ?? accumulationMs) > 0 ? `${recordingStatus.accumulation_ms ?? accumulationMs}ms` : 'RAW'}
                </span>
                <span className="rec-dir" title={recordingStatus.directory}>
                  {recordingStatus.directory}
                </span>
              </div>
              <div className="recording-stats">
                <div className="rec-stat">
                  <span className="stat-label">Saved</span>
                  <span className="stat-val">{recordingStatus.frames}</span>
                </div>
                <div className="rec-stat">
                  <span className="stat-label">Accumulated</span>
                  <span className="stat-val">
                    {(recordingStatus.accumulation_ms ?? accumulationMs) > 0 
                      ? `${recordingStatus.accumulated_frames || 1} f/stack` 
                      : '1 (raw)'}
                  </span>
                </div>
                <div className="rec-stat">
                  <span className="stat-label">Storage</span>
                  <span className="stat-val">{(recordingStatus.mb_stored || 0).toFixed(1)} <small>MB</small></span>
                </div>
                <div className="rec-stat">
                  <span className="stat-label">Time</span>
                  <span className="stat-val">{Math.floor(recordingStatus.duration_sec || 0)}s</span>
                </div>
              </div>
              <button className="btn-record stop" onClick={handleToggleRecording}>
                <Square size={16} fill="currentColor" /> Stop Recording
              </button>
            </div>
          ) : (
            <div className="recording-panel stopped">
              {recordingStatus.frames > 0 && (
                <div className="last-rec-info">
                  Last: {recordingStatus.frames} files ({(recordingStatus.mb_stored || 0).toFixed(1)} MB)
                  {(recordingStatus.accumulation_ms ?? 0) > 0 && ` • ~${recordingStatus.accumulated_frames || 1} frames/stack`}
                </div>
              )}
              <button className="btn-record start" onClick={handleToggleRecording}>
                <Circle size={14} fill="#ff4d4f" color="#ff4d4f" /> Start Recording ({accumulationMs === 0 ? 'Raw' : accumulationMs === 200 ? '200ms / 5 FPS' : '1000ms / 1 FPS'})
              </button>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

export default App;
