import { useState, useEffect, useRef } from 'react';
import './App.css';
import { Camera, Sliders, Zap, Menu, X, Terminal, Grid, RefreshCw, Compass, Circle, Square, Layers, Sparkles } from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:8000`;

const EXPOSURE_PRESETS = [
  { label: '10ms', value: 100 },
  { label: '20ms', value: 200 },
  { label: '30ms', value: 300 },
  { label: '50ms', value: 500 },
  { label: '100ms', value: 1000 },
  { label: '250ms', value: 2500 },
  { label: '500ms', value: 5000 },
  { label: '1s', value: 10000 },
];

const CAMERA_SETTING_KEYS: (keyof Controls)[] = [
  'exposure',
  'gain',
  'brightness',
  'contrast',
  'saturation',
  'sharpness',
  'auto_exposure'
];

interface Controls {
  brightness: number;
  contrast: number;
  saturation: number;
  gain: number;
  exposure: number;
  sharpness: number;
  average?: number;
  accumulation_count?: number;
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

interface SkyConfig {
  backend: string;
  backends: string[];
  latitude: number;
  longitude: number;
  ra: number;
  dec: number;
  focal_length_mm: number;
  pixel_size_um: number;
  arcsec_per_pixel: number;
  fov_deg: [number, number];
  altitude: number;
  azimuth: number;
  target: string;
  drift_px_per_sec: number;
  history: { name: string; ra: number; dec: number }[];
  status: { state: string; stars: number; message: string };
}

interface TargetItem {
  name: string;
  alt: string;
  type: string;
  ra: number;
  dec: number;
}

interface RecordingStatus {
  is_recording: boolean;
  frames: number;
  raw_frames?: number;
  accumulated_frames?: number;
  accumulation_count?: number;
  accumulation_ms?: number;
  format?: string;
  bytes_stored: number;
  mb_stored: number;
  directory: string;
  duration_sec: number;
  fps: number;
  dropped_frames?: number;
}

interface MotorStatus {
  duty_cycle?: number;
  current_duty?: number;
  target_duty?: number;
  voltage?: number;
  mock_mode?: boolean;
  mount_mode?: string;
  is_ramping?: boolean;
}

interface GalleryItem {
  id: string;
  type?: 'sequence' | 'image';
  name?: string;
  target?: string;
  time?: string;
  date?: string;
  datetime?: string;
  frames?: number;
  first_frame?: string;
  thumbnail: string;
  url: string;
  first_frame_url?: string;
  format?: string;
  has_mosaic?: boolean;
  mosaic_status?: 'ready' | 'processing' | 'none';
  mosaic_url?: string;
  mosaic_mtime?: number;
}

const formatGalleryItem = (item: any): GalleryItem => {
  if (typeof item === 'string') {
    return {
      id: item,
      type: 'image',
      time: '',
      date: '',
      datetime: item,
      frames: 1,
      thumbnail: `${API_BASE}/captures/${item}`,
      url: `${API_BASE}/captures/${item}`,
      has_mosaic: false,
      mosaic_status: 'none'
    };
  }
  const thumbUrl = item.thumbnail?.startsWith('http')
    ? item.thumbnail
    : `${API_BASE}${item.thumbnail?.startsWith('/') ? '' : '/'}${item.thumbnail || ''}`;
  const fullUrl = item.url?.startsWith('http')
    ? item.url
    : `${API_BASE}${item.url?.startsWith('/') ? '' : '/'}${item.url || ''}`;
  const mosaicUrl = item.mosaic_url
    ? (item.mosaic_url.startsWith('http') ? item.mosaic_url : `${API_BASE}${item.mosaic_url.startsWith('/') ? '' : '/'}${item.mosaic_url}`)
    : undefined;
  return {
    ...item,
    thumbnail: thumbUrl,
    url: fullUrl,
    mosaic_url: mosaicUrl,
    mosaic_status: item.mosaic_status || 'none',
    has_mosaic: !!item.has_mosaic
  };
};

function App() {
  const [controls, setControls] = useState<Controls>({
    brightness: 128, contrast: 32, saturation: 64, gain: 0,
    exposure: 300, sharpness: 0, average: 1, auto_exposure: 0
  });
  const [status, setStatus] = useState<string>('Ready');
  const [logs, setLogs] = useState<string[]>([]);
  const [captures, setCaptures] = useState<GalleryItem[]>([]);
  const [cameraEngine, setCameraEngine] = useState<string>('mock');
  const [mountEngine, setMountEngine] = useState<string>('mock');
  const [isReconnecting, setIsReconnecting] = useState<boolean>(false);
  const [accumulationCount, setAccumulationCount] = useState<number>(10);
  const [recordFormat, setRecordFormat] = useState<'tif' | 'jpg'>('tif');
  const [recordingStatus, setRecordingStatus] = useState<RecordingStatus>({
    is_recording: false,
    frames: 0,
    raw_frames: 0,
    accumulated_frames: 1,
    accumulation_count: 1,
    accumulation_ms: 1,
    format: 'tif',
    bytes_stored: 0,
    mb_stored: 0,
    directory: '',
    duration_sec: 0,
    fps: 0
  });
  const [motorStatus, setMotorStatus] = useState<MotorStatus>({ duty_cycle: 0, current_duty: 0, voltage: 0, mock_mode: true });
  const [isAdjustingMotor, setIsAdjustingMotor] = useState(false);
  const [prevDuty, setPrevDuty] = useState<number>(85.0);
  const [sky, setSky] = useState<SkyConfig | null>(null);
  const [targets, setTargets] = useState<TargetItem[]>([]);
  const [searchText, setSearchText] = useState('');
  const [searchMsg, setSearchMsg] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchIsOpen, setSearchIsOpen] = useState(false);
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
  const [health, setHealth] = useState<{
    connected: boolean;
    mean_brightness: number;
    width: number;
    height: number;
    fps: number;
    accumulation_count?: number;
    acc_count?: number;
    camera_mode?: string;
    mount_mode?: string;
    recording?: RecordingStatus;
  }>({
    connected: true,
    mean_brightness: 0,
    width: 1920,
    height: 1080,
    fps: 0,
    accumulation_count: 1,
    acc_count: 1
  });
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const logWindowRef = useRef<HTMLDivElement>(null);

  const handleSetAccumulationCount = (count: number) => {
    setAccumulationCount(count);
    fetch(`${API_BASE}/pipeline/accumulation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ count })
    }).catch(e => console.error("Error setting accumulation:", e));
  };

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
          const duty = data.duty_cycle !== undefined ? data.duty_cycle : (data.current_duty !== undefined ? data.current_duty : 0);
          const voltage = data.voltage !== undefined ? data.voltage : (3.3 * duty) / 100;
          setMotorStatus({
            ...data,
            duty_cycle: duty,
            current_duty: duty,
            voltage
          });
          if (duty > 0.05) {
            setPrevDuty(duty);
          }
        }

        const lRes = await fetch(`${API_BASE}/logs`);
        if (lRes.ok) setLogs(await lRes.json());

        const capRes = await fetch(`${API_BASE}/gallery`);
        if (capRes.ok) {
          const data = await capRes.json();
          if (Array.isArray(data)) {
            setCaptures(data.map(formatGalleryItem));
          }
        }

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
    setMotorStatus(prev => ({
      ...prev,
      duty_cycle: speed,
      current_duty: speed,
      voltage: (3.3 * speed) / 100
    }));
    if (speed > 0.05) {
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

  const loadSky = () =>
    fetch(`${API_BASE}/mock/sky`).then(r => r.json()).then(d => {
      if (d.success) setSky(d);
    }).catch(e => console.error("Error loading sky:", e));

  useEffect(() => {
    loadSky();
    fetch(`${API_BASE}/mock/sky/targets`)
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setTargets(data); })
      .catch(e => console.error("Error loading targets:", e));
    // Poll while the catalog is loading so the status line updates
    const id = setInterval(() => { if (sky?.status?.state === 'loading') loadSky(); }, 1500);
    return () => clearInterval(id);
  }, [sky?.status?.state]);

  // Az/Alt change continuously as the sky rotates, so refresh them periodically
  useEffect(() => {
    const id = setInterval(loadSky, 5000);
    return () => clearInterval(id);
  }, []);

  const searchSky = (text: string = searchText) => {
    const query = text.trim();
    if (!query || searching) return;
    // 1. Check recent history (instant, offline)
    const knownHistory = sky?.history.find(h => h.name.toLowerCase() === query.toLowerCase());
    if (knownHistory) {
      setSearchText('');
      setSearchMsg(`Pointing at ${knownHistory.name}`);
      applySky({ ra: knownHistory.ra, dec: knownHistory.dec, target: knownHistory.name });
      return;
    }
    // 2. Check local catalog of ~250 Messier & named stars (instant, offline)
    const knownTarget = targets.find(t => 
      t.name.toLowerCase() === query.toLowerCase() || 
      (t.alt && t.alt.toLowerCase() === query.toLowerCase())
    );
    if (knownTarget) {
      setSearchText('');
      const displayName = knownTarget.alt ? `${knownTarget.name} (${knownTarget.alt})` : knownTarget.name;
      setSearchMsg(`Pointing at ${displayName}`);
      applySky({ ra: knownTarget.ra, dec: knownTarget.dec, target: displayName });
      return;
    }
    // 3. Fallback to SIMBAD query
    setSearching(true);
    setSearchMsg('');
    fetch(`${API_BASE}/mock/sky/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    }).then(r => r.json()).then(d => {
      if (d.sky) setSky(d.sky);
      if (d.success) { setSearchText(''); setSearchMsg(`Pointing at ${d.name}`); }
      else setSearchMsg(d.error || 'Search failed');
    }).catch(() => setSearchMsg('Search failed (backend unreachable)'))
      .finally(() => setSearching(false));
  };

  const applySky = (changes: Partial<SkyConfig>) => {
    // Only the edited fields are sent; values are persisted server-side in config.json
    setSky(prev => prev ? { ...prev, ...changes, status: { ...prev.status, state: 'loading', message: 'Loading...' } } : prev);
    fetch(`${API_BASE}/mock/sky`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(changes)
    }).then(r => r.json()).then(d => setSky(d)).catch(e => console.error("Error setting sky:", e));
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
              accumulation_count: data.accumulation_count,
              accumulation_ms: data.accumulation_ms,
              format: data.format,
              mb_stored: data.mb_stored,
              directory: data.directory
            }));
            const count = data.accumulation_count ?? data.accumulated_frames ?? 1;
            const accMsg = count > 1 ? ` (~${count}x stack 16-bit)` : '';
            showToast(`Recording stopped: ${data.frames} files saved${accMsg} (${data.mb_stored} MB)`, 3500);
            fetch(`${API_BASE}/gallery`)
              .then(res => res.json())
              .then(gData => {
                if (Array.isArray(gData)) {
                  setCaptures(gData.map(formatGalleryItem));
                }
              })
              .catch(e => console.error("Gallery refresh error:", e));
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
          accumulation_count: accumulationCount,
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
              accumulation_count: data.accumulation_count ?? accumulationCount,
              accumulation_ms: data.accumulation_ms ?? accumulationCount,
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
      const targetSpeed = mode === 'on' ? (prevDuty > 0.05 ? prevDuty : 85.0) : 0.0;
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

  const handleThumbnailClick = async (item: GalleryItem) => {
    if (item.type !== 'sequence') {
      window.open(item.url, '_blank');
      return;
    }

    if (item.mosaic_status === 'ready' && item.mosaic_url) {
      const cacheBust = item.mosaic_mtime ? `?t=${Math.floor(item.mosaic_mtime)}` : `?t=${Date.now()}`;
      window.open(`${item.mosaic_url}${cacheBust}`, '_blank');
      return;
    }

    if (item.mosaic_status === 'processing') {
      showToast('Mosaic is currently being processed. Opening preview...', 2500);
      const targetUrl = item.mosaic_url || item.url;
      window.open(`${targetUrl}?t=${Date.now()}`, '_blank');
      return;
    }

    showToast(`Generating mosaic for ${item.id}...`, 3000);
    setCaptures(prev => prev.map(c => c.id === item.id ? { ...c, mosaic_status: 'processing' } : c));

    try {
      const res = await fetch(`${API_BASE}/mosaic/generate/${item.id}`, { method: 'POST' });
      const data = await res.json();
      if (data.success && data.url) {
        const fullUrl = data.url.startsWith('http')
          ? data.url
          : `${API_BASE}${data.url.startsWith('/') ? '' : '/'}${data.url}`;
        window.open(`${fullUrl}?t=${Date.now()}`, '_blank');
        setTimeout(() => {
          fetch(`${API_BASE}/gallery`)
            .then(r => r.json())
            .then(gData => {
              if (Array.isArray(gData)) {
                setCaptures(gData.map(formatGalleryItem));
              }
            })
            .catch(() => {});
        }, 800);
      } else {
        showToast(`Error: ${data.error || 'Failed to start mosaic'}`, 4000);
        setCaptures(prev => prev.map(c => c.id === item.id ? { ...c, mosaic_status: 'none' } : c));
      }
    } catch (err: any) {
      showToast(`Mosaic request failed: ${err.message}`, 4000);
      setCaptures(prev => prev.map(c => c.id === item.id ? { ...c, mosaic_status: 'none' } : c));
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
            {accumulationCount > 1 ? (
              <span style={{ color: '#58a6ff', fontWeight: 600 }}>
                Stack: {health.acc_count ?? 1}/{accumulationCount} ({accumulationCount}x)
              </span>
            ) : (
              <span>Stack: 1x (Raw)</span>
            )}
            <span>Luminance: {(health.mean_brightness || 0).toFixed(1)}</span>
            {sky && (
              <span title="Azimuth from north / altitude above the horizon">
                Az {sky.azimuth.toFixed(1)}° · Alt {sky.altitude.toFixed(1)}°
              </span>
            )}
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
                <div className="empty-msg">No sequences recorded yet</div>
              ) : (
                captures.map(item => {
                  const isSeq = item.type === 'sequence';
                  const titleTip = isSeq
                    ? `${item.datetime || item.id} • ${item.frames || 1} frames • Mosaic: ${item.mosaic_status === 'ready' ? 'Ready (Click to View)' : item.mosaic_status === 'processing' ? 'Processing (Click to View Preview)' : 'Not Created (Click to Stitch Mosaic)'}`
                    : `${item.datetime || item.id} • Single frame`;

                  return (
                    <div 
                      key={item.id} 
                      className="capture-item" 
                      title={titleTip} 
                      onClick={() => handleThumbnailClick(item)}
                    >
                      <img src={item.thumbnail} alt={item.datetime || item.id} loading="lazy" />
                      
                      {isSeq && (
                        <>
                          {item.mosaic_status === 'ready' && (
                            <div className="mosaic-badge ready" title="16-bit Mosaic Ready (Click to View)">
                              <Sparkles size={11} />
                              <span>Mosaic Ready</span>
                            </div>
                          )}
                          {item.mosaic_status === 'processing' && (
                            <div className="mosaic-badge processing" title="Mosaic is being processed...">
                              <RefreshCw size={11} className="spin" />
                              <span>Processing...</span>
                            </div>
                          )}
                          {item.mosaic_status === 'none' && (
                            <div className="mosaic-badge none" title="Click thumbnail to stitch mosaic">
                              <Layers size={11} />
                              <span>Stitch Mosaic</span>
                            </div>
                          )}
                        </>
                      )}

                      {item.frames !== undefined && item.frames > 0 && (
                        <div className="capture-badge">
                          <Layers size={11} />
                          <span>{item.frames} {item.frames === 1 ? 'frame' : 'frames'}</span>
                        </div>
                      )}

                      <div className="capture-label">
                        {item.target && (
                          <div style={{ fontWeight: 600, color: 'var(--accent)', fontSize: '11px', marginBottom: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            ✨ {item.target}
                          </div>
                        )}
                        <div className="capture-time-row">
                          <span className="capture-time">{item.time || item.id}</span>
                          {item.format && <span className="capture-fmt-tag">{item.format.toUpperCase()}</span>}
                        </div>
                        {item.date && <div className="capture-date-row">{item.date}</div>}
                      </div>
                    </div>
                  );
                })
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

          <div className="control-section" style={{ marginTop: '-12px' }}>
            <div className="section-header"><Compass size={16} /> Navigation & Targets</div>
            <div className="tracking-telemetry" style={{ marginTop: '0px', background: 'rgba(88, 166, 255, 0.05)', borderColor: 'rgba(88, 166, 255, 0.15)' }}>
              {sky && (
                <div style={{ marginTop: '4px', fontSize: '11px', color: 'var(--text-muted)' }}>
                  <div style={{ position: 'relative', marginBottom: '4px' }}>
                      <div style={{ display: 'flex', gap: '6px' }}>
                        <input
                          type="text"
                          placeholder={sky.history.length ? 'Find object or pick recent...' : 'Find object (Polaris, M31...)'}
                          value={searchText}
                          onFocus={() => setSearchIsOpen(true)}
                          onChange={e => {
                            setSearchText(e.target.value);
                            setSearchIsOpen(true);
                          }}
                          onKeyDown={e => {
                            if (e.key === 'Enter') {
                              setSearchIsOpen(false);
                              searchSky();
                            } else if (e.key === 'Escape') {
                              setSearchIsOpen(false);
                            }
                          }}
                          style={{ flex: 1, minWidth: 0, fontSize: '11px' }}
                        />
                        <button
                          onClick={() => {
                            setSearchIsOpen(false);
                            searchSky();
                          }}
                          disabled={searching || !searchText.trim()}
                          style={{ fontSize: '11px' }}
                        >
                          {searching ? '...' : 'Go'}
                        </button>
                      </div>

                      {searchIsOpen && (
                        <>
                          <div
                            style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 998 }}
                            onClick={() => setSearchIsOpen(false)}
                          />
                          <div
                            style={{
                              position: 'absolute',
                              top: '100%',
                              left: 0,
                              right: 0,
                              marginTop: '2px',
                              maxHeight: '180px',
                              overflowY: 'auto',
                              background: 'var(--bg-panel)',
                              border: '1px solid var(--border-color)',
                              borderRadius: '4px',
                              zIndex: 999,
                              boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                            }}
                          >
                            {/* Filter suggestions */}
                            {(() => {
                              const q = searchText.trim().toLowerCase();
                              const matchedHistory = sky.history.filter(h => !q || h.name.toLowerCase().includes(q));
                              const matchedTargets = targets.filter(t => 
                                !q || t.name.toLowerCase().includes(q) || (t.alt && t.alt.toLowerCase().includes(q))
                              ).slice(0, 30); // show top 30 matches

                              if (matchedHistory.length === 0 && matchedTargets.length === 0) {
                                return (
                                  <div style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: '11px' }}>
                                    No matches found. Press 'Go' to search SIMBAD.
                                  </div>
                                );
                              }

                              return (
                                <>
                                  {matchedHistory.length > 0 && (
                                    <div>
                                      <div style={{ padding: '3px 8px', fontSize: '9px', fontWeight: 600, color: 'var(--text-muted)', background: 'rgba(255,255,255,0.03)' }}>
                                        RECENT
                                      </div>
                                      {matchedHistory.map(h => (
                                        <div
                                          key={`h-${h.name}`}
                                          onClick={() => {
                                            setSearchIsOpen(false);
                                            setSearchText('');
                                            setSearchMsg(`Pointing at ${h.name}`);
                                            applySky({ ra: h.ra, dec: h.dec, target: h.name });
                                          }}
                                          style={{
                                            padding: '4px 8px',
                                            cursor: 'pointer',
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            alignItems: 'center',
                                            fontSize: '11px',
                                            borderBottom: '1px solid rgba(255,255,255,0.03)'
                                          }}
                                          onMouseEnter={e => (e.currentTarget.style.background = 'rgba(88, 166, 255, 0.15)')}
                                          onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                                        >
                                          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{h.name}</span>
                                          <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>🕒 recent</span>
                                        </div>
                                      ))}
                                    </div>
                                  )}

                                  {matchedTargets.length > 0 && (
                                    <div>
                                      {matchedHistory.length > 0 && (
                                        <div style={{ padding: '3px 8px', fontSize: '9px', fontWeight: 600, color: 'var(--text-muted)', background: 'rgba(255,255,255,0.03)' }}>
                                          CATALOG
                                        </div>
                                      )}
                                      {matchedTargets.map(t => {
                                        const label = t.alt ? `${t.name} (${t.alt})` : t.name;
                                        return (
                                          <div
                                            key={`t-${t.name}`}
                                            onClick={() => {
                                              setSearchIsOpen(false);
                                              setSearchText('');
                                              setSearchMsg(`Pointing at ${label}`);
                                              applySky({ ra: t.ra, dec: t.dec, target: label });
                                            }}
                                            style={{
                                              padding: '4px 8px',
                                              cursor: 'pointer',
                                              display: 'flex',
                                              justifyContent: 'space-between',
                                              alignItems: 'center',
                                              fontSize: '11px',
                                              borderBottom: '1px solid rgba(255,255,255,0.03)'
                                            }}
                                            onMouseEnter={e => (e.currentTarget.style.background = 'rgba(88, 166, 255, 0.15)')}
                                            onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                                          >
                                            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{t.name}</span>
                                            <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                                              {t.alt ? t.alt : t.type}
                                            </span>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  )}
                                </>
                              );
                            })()}
                          </div>
                        </>
                      )}
                    </div>
                    <div style={{ marginBottom: '8px', minHeight: '14px', color: searchMsg.startsWith('Could') || searchMsg.startsWith('Search') ? 'var(--danger, #f85149)' : undefined }}>
                      {searchMsg || (sky.target ? `Target: ${sky.target}` : '')}
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span>Star Catalog:</span>
                      <select
                        value={sky.backend}
                        onChange={e => applySky({ backend: e.target.value })}
                        style={{ fontSize: '11px' }}
                      >
                        {sky.backends.map(b => <option key={b} value={b}>{b}</option>)}
                      </select>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 8px' }}>
                      {([
                        ['latitude', 'Lat (°)'],
                        ['longitude', 'Lon (°)'],
                        ['ra', 'RA (°)'],
                        ['dec', 'Dec (°)'],
                        ['focal_length_mm', 'Focal (mm)'],
                        ['pixel_size_um', 'Pixel (µm)'],
                      ] as [keyof SkyConfig, string][]).map(([key, label]) => (
                        <label key={key} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                          {label}
                          <input
                            type="number"
                            step="any"
                            key={`${key}-${sky[key]}`}
                            defaultValue={sky[key] as number}
                            onBlur={e => {
                              const v = parseFloat(e.target.value);
                              if (!isNaN(v) && v !== sky[key]) applySky({ [key]: v } as Partial<SkyConfig>);
                            }}
                            onKeyDown={e => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
                            style={{ width: '100%', fontFamily: 'monospace', fontSize: '11px' }}
                          />
                        </label>
                      ))}
                    </div>
                    <div style={{ marginTop: '8px' }}>
                      {sky.arcsec_per_pixel.toFixed(2)}″/px · FOV {sky.fov_deg[0].toFixed(2)}° × {sky.fov_deg[1].toFixed(2)}° · drift {sky.drift_px_per_sec.toFixed(1)} px/s
                    </div>
                    <div style={{ marginTop: '2px', color: sky.status.state === 'error' ? 'var(--danger, #f85149)' : undefined }}>
                      {sky.status.message}
                    </div>
                  </div>
                )}
              </div>
            </div>

          <div className="control-section">
            <div className="section-header"><Zap size={16} /> Mount</div>
            <div className="control-group">
              <label>Duty Cycle: {(motorStatus.duty_cycle || 0).toFixed(1)}%</label>
              <input type="range" min="0" max="100" step="0.2" value={motorStatus.duty_cycle || 0} onChange={(e) => updateMotorSpeed(parseFloat(e.target.value))} />
              <div className="preset-row">
                <button 
                  className={(!trackingStatus.active && (motorStatus.duty_cycle || 0) <= 0.05) ? 'active' : ''} 
                  onClick={() => setMountMode('off')}
                >
                  OFF
                </button>
                <button 
                  className={(!trackingStatus.active && (motorStatus.duty_cycle || 0) > 0.05) ? 'active' : ''} 
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

              {trackingStatus.active && (
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
              )}
            </div>
          </div>

          <div className="control-section">
            <div className="section-header"><Zap size={16} /> Live Stacking & Recording</div>
            <div className="control-group">
              <label>Stacking Window (Canvas & Recording)</label>
              <div className="preset-row">
                <button 
                  className={accumulationCount === 1 ? 'active' : ''} 
                  onClick={() => handleSetAccumulationCount(1)}
                  disabled={recordingStatus.is_recording}
                  title="Raw (1x): Live view & recording of each raw sub-exposure directly"
                >
                  1x (Raw)
                </button>
                <button 
                  className={accumulationCount === 10 ? 'active' : ''} 
                  onClick={() => handleSetAccumulationCount(10)}
                  disabled={recordingStatus.is_recording}
                  title="10x stack (3.2x SNR, ~1.6 bits extra dynamic range)"
                >
                  10x
                </button>
                <button 
                  className={accumulationCount === 25 ? 'active' : ''} 
                  onClick={() => handleSetAccumulationCount(25)}
                  disabled={recordingStatus.is_recording}
                  title="25x stack (5.0x SNR, ~2.3 bits extra dynamic range)"
                >
                  25x
                </button>
                <button 
                  className={accumulationCount === 100 ? 'active' : ''} 
                  onClick={() => handleSetAccumulationCount(100)}
                  disabled={recordingStatus.is_recording}
                  title="100x deep stack (10.0x SNR, ~3.3 bits extra dynamic range)"
                >
                  100x
                </button>
              </div>
            </div>

            <div className="control-group" style={{ marginTop: '8px' }}>
              <label>Capture Format</label>
              <div className="preset-row">
                <button 
                  className={recordFormat === 'tif' ? 'active' : ''} 
                  onClick={() => setRecordFormat('tif')}
                  disabled={recordingStatus.is_recording}
                  title="Lossless uncompressed 16-bit TIFF (ideal for stacking in Siril/DSS)"
                >
                  TIFF (16-bit)
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
            {CAMERA_SETTING_KEYS.map((key) => {
              const value = controls ? controls[key] : 0;
              let minVal = 0;
              let maxVal = 255;
              let stepVal = 1;
              let labelText = `${key}: ${value ?? 0}`;

              if (key === 'exposure') {
                minVal = 1;
                maxVal = 10000; // Up to 1.0s (10000 x 100µs)
                stepVal = 10;
                const ms = (value || 0) * 0.1;
                const timeStr = ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(1)}ms`;
                labelText = `Exposure: ${value} (~${timeStr})`;
              } else if (key === 'brightness') {
                minVal = 0;
                maxVal = 255;
                const note = value === 128 ? ' (128 = Neutral Black Level)' : (value || 0) < 128 ? ' (Dark Offset - Clips Blacks)' : ' (Bright Offset)';
                labelText = `Brightness: ${value}${note}`;
              } else if (key === 'gain') {
                minVal = 0;
                maxVal = 255;
                const note = value === 0 ? ' (0 = Unity 1x, Max Dynamic Range)' : ` (+${(((value || 0) / 255) * 100).toFixed(0)}% Boost)`;
                labelText = `Gain: ${value}${note}`;
              } else if (key === 'contrast') {
                minVal = 0;
                maxVal = 255;
                const note = value === 32 ? ' (32 = Neutral Linear)' : ' (Non-linear S-Curve)';
                labelText = `Contrast: ${value}${note}`;
              } else if (key === 'saturation') {
                minVal = 0;
                maxVal = 255;
                const note = value === 64 ? ' (64 = Neutral Color)' : value === 0 ? ' (0 = Monochrome)' : '';
                labelText = `Saturation: ${value}${note}`;
              } else if (key === 'sharpness') {
                minVal = 0;
                maxVal = 6;
                const note = value === 0 ? ' (0 = Off, Recommended for Astro)' : ' (Warning: Ringing Halos)';
                labelText = `Sharpness: ${value}${note}`;
              } else if (key === 'auto_exposure') {
                minVal = 0;
                maxVal = 1;
                labelText = `Auto Exposure: ${value ? 'Auto (1)' : 'Manual (0)'}`;
              }

              return (
                <div key={key} className="control-group">
                  <label>{labelText}</label>
                  <input 
                    type="range" 
                    min={minVal} 
                    max={maxVal} 
                    step={stepVal}
                    value={value ?? 0} 
                    onChange={e => updateControl(key, parseInt(e.target.value))} 
                  />
                  {key === 'exposure' && (
                    <div className="preset-row wrap">
                      {EXPOSURE_PRESETS.map(p => (
                        <button
                          key={p.value}
                          type="button"
                          className={controls.exposure === p.value ? 'active' : ''}
                          onClick={() => updateControl('exposure', p.value)}
                          title={`Set exposure to ${p.label} (${p.value} units)`}
                        >
                          {p.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

        </div>
        
        <div className="sidebar-footer">
          {recordingStatus.is_recording ? (
            <div className="recording-panel">
              <div className="recording-header">
                <span className="rec-badge">
                  <span className="rec-dot"></span> REC ({recordFormat.toUpperCase()}{(recordingStatus.accumulation_count ?? 1) > 1 ? ` • ${recordingStatus.accumulation_count}x` : ''})
                </span>
                <span className="rec-dir" title={recordingStatus.directory}>
                  {recordingStatus.directory}
                </span>
              </div>
              <div className="recording-stats">
                <div className="rec-stat" title={(recordingStatus.accumulation_count ?? 1) > 1 ? `${recordingStatus.raw_frames ?? recordingStatus.frames} raw sub-exposures integrated into ${recordingStatus.frames} stacks` : `${recordingStatus.frames} saved frames`}>
                  <span className="stat-label">{(recordingStatus.accumulation_count ?? 1) > 1 ? 'Stacks' : 'Saved'}</span>
                  <span className="stat-val">{recordingStatus.frames}</span>
                </div>
                <div className="rec-stat" title="Saved frame write rate">
                  <span className="stat-label">Rate</span>
                  <span className="stat-val">{(recordingStatus.fps || 0).toFixed(1)} <small>fps</small></span>
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
                  Last: {recordingStatus.frames} {(recordingStatus.accumulation_count ?? 1) > 1 ? 'stacks' : 'files'}{(recordingStatus.accumulation_count ?? 1) > 1 && (recordingStatus.accumulated_frames ?? 0) > 1 ? ` (~${recordingStatus.accumulated_frames} subs/stack)` : ''} ({(recordingStatus.mb_stored || 0).toFixed(1)} MB)
                </div>
              )}
              <button className="btn-record start" onClick={handleToggleRecording}>
                <Circle size={14} fill="#ff4d4f" color="#ff4d4f" /> {accumulationCount > 1 ? `Record ${accumulationCount}x Stack (${recordFormat.toUpperCase()})` : `Record Raw (${recordFormat.toUpperCase()})`}
              </button>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

export default App;
