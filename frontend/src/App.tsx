import { useState, useEffect, useRef } from 'react';
import './App.css';
import { Camera, Sliders, Zap, Menu, X, Terminal, Grid, RefreshCw, RotateCcw, Compass, Circle, Square, Layers, Sparkles, Save } from 'lucide-react';

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
  'gamma'
];

interface Controls {
  brightness: number;
  contrast: number;
  saturation: number;
  gain: number;
  exposure: number;
  sharpness: number;
  gamma?: number;
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

interface ProcessingConfig {
  histogram: boolean;
  corrupted_filter: boolean;
  luminance_check: boolean;
  roi: [number, number, number, number];
  stride?: number;
}

interface CalibrationMeta {
  timestamp: string;
  frames_collected: number;
  defect_pixel_count: number;
  median_bg: number;
  sigma_bg: number;
  threshold: number;
}

interface CalibrationStatus {
  is_calibrating: boolean;
  frames_collected: number;
  target_frames: number;
  calibrated: boolean;
  meta?: CalibrationMeta | null;
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
    brightness: 0, contrast: 50, saturation: 0, gain: 64,
    exposure: 1000, sharpness: 50, gamma: 100, average: 1, auto_exposure: 0
  });
  const [status, setStatus] = useState<string>('Ready');
  const [logs, setLogs] = useState<string[]>([]);
  const [captures, setCaptures] = useState<GalleryItem[]>([]);
  const [cameraEngine, setCameraEngine] = useState<string>('mock');
  const [mountEngine, setMountEngine] = useState<string>('mock');
  const [cameraFormat, setCameraFormat] = useState<'YUYV' | 'MJPG'>('YUYV');
  const [isReconnecting, setIsReconnecting] = useState<boolean>(false);
  const [isRestarting, setIsRestarting] = useState<boolean>(false);
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
    camera_format?: string;
    recording?: RecordingStatus;
    corrupted_frames_dropped?: number;
    histogram?: number[];
    processing?: ProcessingConfig;
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
  const histCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [liveProcessing, setLiveProcessing] = useState<ProcessingConfig>({
    histogram: true,
    corrupted_filter: true,
    luminance_check: true,
    roi: [0, 0, 1920, 1080],
    stride: 4
  });
  const [roiInputs, setRoiInputs] = useState({ x: 0, y: 0, w: 1920, h: 1080 });
  const [showRoiBox, setShowRoiBox] = useState(true);

  const [calibrationStatus, setCalibrationStatus] = useState<CalibrationStatus>({
    is_calibrating: false,
    frames_collected: 0,
    target_frames: 15,
    calibrated: false,
    meta: null
  });

  const [streamEpoch, setStreamEpoch] = useState<number>(Date.now());

  // Interactive Astrophotography Display Contrast / Stretch (Black & White point handles on Histogram)
  const [blackPoint, setBlackPoint] = useState<number>(0);
  const [whitePoint, setWhitePoint] = useState<number>(255);
  const [activeHandle, setActiveHandle] = useState<'black' | 'white' | null>(null);
  const blackPointRef = useRef(blackPoint);
  const whitePointRef = useRef(whitePoint);
  const lastHistDataRef = useRef<number[]>([]);
  const drawHistogramRef = useRef<(data: number[]) => void>(() => {});

  useEffect(() => {
    blackPointRef.current = blackPoint;
    whitePointRef.current = whitePoint;
  }, [blackPoint, whitePoint]);

  const stretchSpan = Math.max(1, whitePoint - blackPoint);
  const stretchSlope = (255 / stretchSpan).toFixed(4);
  const stretchIntercept = (-blackPoint / stretchSpan).toFixed(4);
  const isDisplayStretched = blackPoint > 0 || whitePoint < 255;

  const handleHistPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const bin = Math.max(0, Math.min(255, Math.round((x / rect.width) * 255)));

    const distB = Math.abs(bin - blackPointRef.current);
    const distW = Math.abs(bin - whitePointRef.current);

    if (distB <= distW) {
      const nextB = Math.min(bin, whitePointRef.current - 2);
      setBlackPoint(nextB);
      blackPointRef.current = nextB;
      setActiveHandle('black');
    } else {
      const nextW = Math.max(bin, blackPointRef.current + 2);
      setWhitePoint(nextW);
      whitePointRef.current = nextW;
      setActiveHandle('white');
    }
    if (lastHistDataRef.current.length > 0) {
      drawHistogramRef.current(lastHistDataRef.current);
    }
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handleHistPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.stopPropagation();
    if (!activeHandle) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const bin = Math.max(0, Math.min(255, Math.round((x / rect.width) * 255)));

    if (activeHandle === 'black') {
      const nextB = Math.min(bin, whitePointRef.current - 2);
      setBlackPoint(nextB);
      blackPointRef.current = nextB;
    } else if (activeHandle === 'white') {
      const nextW = Math.max(bin, blackPointRef.current + 2);
      setWhitePoint(nextW);
      whitePointRef.current = nextW;
    }
    if (lastHistDataRef.current.length > 0) {
      drawHistogramRef.current(lastHistDataRef.current);
    }
  };

  const handleHistPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.stopPropagation();
    setActiveHandle(null);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  };

  const handleHistDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.stopPropagation();
    setBlackPoint(0);
    setWhitePoint(255);
    blackPointRef.current = 0;
    whitePointRef.current = 255;
    if (lastHistDataRef.current.length > 0) {
      drawHistogramRef.current(lastHistDataRef.current);
    }
    showToast('Display stretch reset (0 – 255)');
  };

  const handleStartCalibration = async () => {
    try {
      const res = await fetch(`${API_BASE}/calibration/black`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames: 15 })
      });
      if (res.ok) {
        setCalibrationStatus(prev => ({ ...prev, is_calibrating: true, frames_collected: 0 }));
        showToast('Black calibration started. Keep lens covered...');
      }
    } catch (e: any) {
      showToast(`Calibration error: ${e.message}`, 4000);
    }
  };

  const handleClearCalibration = async () => {
    try {
      const res = await fetch(`${API_BASE}/calibration/black`, { method: 'DELETE' });
      if (res.ok) {
        setCalibrationStatus({
          is_calibrating: false,
          frames_collected: 0,
          target_frames: 15,
          calibrated: false,
          meta: null
        });
        showToast('Black calibration cleared');
      }
    } catch (e: any) {
      showToast(`Error clearing calibration: ${e.message}`, 4000);
    }
  };

  // Setting-Indexed Baseline Calibration State
  const [baselineStatus, setBaselineStatus] = useState<{
    exists: boolean;
    subtract: boolean;
    pedestal: number;
    is_capturing: boolean;
  }>({
    exists: false,
    subtract: false,
    pedestal: 10,
    is_capturing: false
  });
  const [pedestalInput, setPedestalInput] = useState<number>(10);

  const baselineKey = `E${controls.exposure ?? 1000}-G${controls.gain ?? 64}-B${controls.brightness ?? 0}-C${controls.contrast ?? 50}-S${controls.saturation ?? 0}-S${controls.sharpness ?? 50}-G${controls.gamma ?? 100}`;

  const fetchBaselineStatus = async (key: string) => {
    try {
      const res = await fetch(`${API_BASE}/calibration/baseline/status?key=${encodeURIComponent(key)}`);
      if (res.ok) {
        const data = await res.json();
        setBaselineStatus(data);
        if (data.pedestal !== undefined) {
          setPedestalInput(data.pedestal);
        }
      }
    } catch {
      // ignore
    }
  };

  const handleCaptureBaseline = async () => {
    try {
      setBaselineStatus(prev => ({ ...prev, is_capturing: true }));
      const res = await fetch(`${API_BASE}/calibration/baseline/capture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: baselineKey, frames: 10 })
      });
      if (res.ok) {
        showToast(`Baseline capturing for ${baselineKey} (10 frames, cap on)...`);
        setTimeout(() => fetchBaselineStatus(baselineKey), 1200);
      }
    } catch (e: any) {
      showToast(`Error capturing baseline: ${e.message}`, 4000);
      setBaselineStatus(prev => ({ ...prev, is_capturing: false }));
    }
  };

  const handleToggleSubtract = async () => {
    const nextVal = !baselineStatus.subtract;
    try {
      const res = await fetch(`${API_BASE}/calibration/baseline/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subtract: nextVal, key: baselineKey })
      });
      if (res.ok) {
        setBaselineStatus(prev => ({ ...prev, subtract: nextVal }));
        showToast(`Baseline subtract: ${nextVal ? 'ON' : 'OFF'}`);
      }
    } catch (e: any) {
      showToast(`Error toggling subtract: ${e.message}`, 4000);
    }
  };

  const handleSetPedestal = async (val: number) => {
    const p = Math.max(0, Math.min(100, val));
    setPedestalInput(p);
    try {
      await fetch(`${API_BASE}/calibration/baseline/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pedestal: p, key: baselineKey })
      });
      setBaselineStatus(prev => ({ ...prev, pedestal: p }));
    } catch (e: any) {
      console.error("Error setting pedestal:", e);
    }
  };

  const handleToggleProcessing = async (stage: 'histogram' | 'corrupted_filter' | 'luminance_check') => {
    const newVal = !liveProcessing[stage];
    setLiveProcessing(prev => ({ ...prev, [stage]: newVal }));
    try {
      await fetch(`${API_BASE}/pipeline/processing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [stage]: newVal })
      });
      showToast(`${stage.replace('_', ' ')} ${newVal ? 'ON' : 'OFF'}`);
    } catch (e: any) {
      showToast(`Error: ${e.message}`, 4000);
    }
  };

  const handleSetStride = async (stride: number) => {
    setLiveProcessing(prev => ({ ...prev, stride }));
    try {
      await fetch(`${API_BASE}/pipeline/processing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stride })
      });
      showToast(`Histogram stride set to ${stride}x`);
    } catch (e: any) {
      showToast(`Error: ${e.message}`, 4000);
    }
  };

  const applyRoiCoords = async (rawX: number, rawY: number, rawW: number, rawH: number) => {
    const rx = Math.max(0, Math.min(1870, Math.round(rawX / 50) * 50));
    const ry = Math.max(0, Math.min(1030, Math.round(rawY / 50) * 50));
    const rw = Math.max(50, Math.min(1920 - rx, Math.round(rawW / 50) * 50));
    const rh = Math.max(50, Math.min(1080 - ry, Math.round(rawH / 50) * 50));
    setRoiInputs({ x: rx, y: ry, w: rw, h: rh });
    const roiArray = [rx, ry, rw, rh] as [number, number, number, number];
    setLiveProcessing(prev => ({ ...prev, roi: roiArray, histogram: true }));
    try {
      await fetch(`${API_BASE}/pipeline/processing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roi: roiArray, histogram: true })
      });
      showToast(`ROI set to ${rw}x${rh} at (${rx}, ${ry})`);
    } catch (e: any) {
      showToast(`Error: ${e.message}`, 4000);
    }
  };

  const handleApplyRoi = async () => {
    await applyRoiCoords(roiInputs.x, roiInputs.y, roiInputs.w, roiInputs.h);
  };

  const isDraggingRoiRef = useRef(false);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  const handlePointerDownStream = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    if ((e.target as HTMLElement).closest('button, input, select, textarea')) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const fw = health.width || 1920;
    const fh = health.height || 1080;
    const clickX = Math.max(0, Math.min(fw, ((e.clientX - rect.left) / rect.width) * fw));
    const clickY = Math.max(0, Math.min(fh, ((e.clientY - rect.top) / rect.height) * fh));

    dragStartRef.current = { x: clickX, y: clickY };
    isDraggingRoiRef.current = true;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch (_) {}
  };

  const handlePointerMoveStream = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRoiRef.current || !dragStartRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const fw = health.width || 1920;
    const fh = health.height || 1080;
    const curX = Math.max(0, Math.min(fw, ((e.clientX - rect.left) / rect.width) * fw));
    const curY = Math.max(0, Math.min(fh, ((e.clientY - rect.top) / rect.height) * fh));

    const start = dragStartRef.current;
    const minX = Math.min(start.x, curX);
    const minY = Math.min(start.y, curY);
    const width = Math.abs(curX - start.x);
    const height = Math.abs(curY - start.y);

    const snapX = Math.round(minX / 50) * 50;
    const snapY = Math.round(minY / 50) * 50;
    const snapW = Math.max(50, Math.round(width / 50) * 50);
    const snapH = Math.max(50, Math.round(height / 50) * 50);

    setRoiInputs({
      x: Math.min(fw - 50, snapX),
      y: Math.min(fh - 50, snapY),
      w: Math.min(fw - snapX, snapW),
      h: Math.min(fh - snapY, snapH)
    });
  };

  const handlePointerUpStream = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRoiRef.current || !dragStartRef.current) return;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch (_) {}

    const start = dragStartRef.current;
    const rect = e.currentTarget.getBoundingClientRect();
    const fw = health.width || 1920;
    const fh = health.height || 1080;
    const endX = Math.max(0, Math.min(fw, ((e.clientX - rect.left) / rect.width) * fw));
    const endY = Math.max(0, Math.min(fh, ((e.clientY - rect.top) / rect.height) * fh));

    isDraggingRoiRef.current = false;
    dragStartRef.current = null;

    const dragDist = Math.hypot(endX - start.x, endY - start.y);
    if (dragDist > 25) {
      const minX = Math.min(start.x, endX);
      const minY = Math.min(start.y, endY);
      const width = Math.abs(endX - start.x);
      const height = Math.abs(endY - start.y);
      applyRoiCoords(minX, minY, width, height);
    }
  };

  const handleResetRoi = async () => {
    const fullRoi = [0, 0, 1920, 1080] as [number, number, number, number];
    setRoiInputs({ x: 0, y: 0, w: 1920, h: 1080 });
    setLiveProcessing(prev => ({ ...prev, roi: fullRoi }));
    try {
      await fetch(`${API_BASE}/pipeline/processing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roi: fullRoi })
      });
      showToast('ROI reset to Full Frame (1920x1080)');
    } catch (e: any) {
      showToast(`Error: ${e.message}`, 4000);
    }
  };

  useEffect(() => {
    let active = true;
    const canvas = histCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawHistogram = (data: number[]) => {
      lastHistDataRef.current = data;
      drawHistogramRef.current = drawHistogram;
      const cvs = histCanvasRef.current;
      if (!cvs) return;
      const w = 512;
      const h = 54;
      if (cvs.width !== w) cvs.width = w;
      if (cvs.height !== h) cvs.height = h;
      const ctx = cvs.getContext('2d');
      if (!ctx) return;

      const yAxis = 44; // Baseline axis position: bars above, ticks below
      ctx.clearRect(0, 0, w, h);

      const n = data.length;
      if (n === 0) return;

      // Hardcoded log scaling: log10(count + 1)
      const logCounts = data.map(c => (c > 0 ? Math.log10(c + 1) : 0));
      const maxLog = Math.max(...logCounts, 1.0);

      // Discrete bar graph: every bin is an independent 2px vertical bar
      const barW = w / n; // 512 / 256 = 2.0px per bin
      ctx.fillStyle = '#58a6ff';

      for (let i = 0; i < n; i++) {
        if (logCounts[i] <= 0) continue;
        const barH = Math.max(1, Math.round((logCounts[i] / maxLog) * (yAxis - 2)));
        const x = i * barW;
        ctx.fillRect(x, yAxis - barH, barW, barH);
      }

      // Horizontal baseline x-axis line
      ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.fillRect(0, yAxis, w, 1);

      // Ticks below x-axis: %100=0 -> thickness 2 ticks below x axis, %10=0 -> thickness 1 ticks
      for (let i = 0; i < n; i++) {
        const x = Math.round(i * barW);
        if (i % 100 === 0) {
          ctx.fillStyle = '#e6edf3';
          ctx.fillRect(x - 1, yAxis + 1, 2, 7);
        } else if (i % 10 === 0) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
          ctx.fillRect(x, yAxis + 1, 1, 4);
        }
      }

      // Draw Black Point handle (Sky Blue cursor & pointer)
      const curB = blackPointRef.current;
      const bx = Math.round(curB * barW);
      ctx.fillStyle = '#38bdf8';
      ctx.fillRect(bx - 1, 0, 2, yAxis);
      ctx.beginPath();
      ctx.moveTo(bx - 4, yAxis + 9);
      ctx.lineTo(bx + 4, yAxis + 9);
      ctx.lineTo(bx, yAxis + 1);
      ctx.fill();

      // Draw White Point handle (Amber cursor & pointer)
      const curW = whitePointRef.current;
      const wx = Math.round(curW * barW);
      ctx.fillStyle = '#fbbf24';
      ctx.fillRect(wx - 1, 0, 2, yAxis);
      ctx.beginPath();
      ctx.moveTo(wx - 4, yAxis + 9);
      ctx.lineTo(wx + 4, yAxis + 9);
      ctx.lineTo(wx, yAxis + 1);
      ctx.fill();
    };

    const fetchHistogram = async () => {
      try {
        const res = await fetch(`${API_BASE}/histogram`);
        if (res.ok && active) {
          const json = await res.json();
          if (Array.isArray(json.histogram) && json.histogram.length > 0) {
            drawHistogram(json.histogram);
          }
        }
      } catch {
        // ignore fetch errors
      }
    };

    fetchHistogram();
    const interval = setInterval(fetchHistogram, 500);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

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
          if (hData.camera_format) setCameraFormat(hData.camera_format);
          if (hData.recording) setRecordingStatus(hData.recording);
          if (hData.accumulation_count !== undefined) {
            setAccumulationCount(hData.accumulation_count);
          }
          if (hData.processing) {
            setLiveProcessing(hData.processing);
          }
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

        const calRes = await fetch(`${API_BASE}/calibration/status`);
        if (calRes.ok) setCalibrationStatus(await calRes.json());

        fetchBaselineStatus(baselineKey);
      } catch (err) {
        console.error("Fetch error:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [isAdjustingMotor, baselineKey]);

  useEffect(() => {
    fetchBaselineStatus(baselineKey);
  }, [baselineKey]);

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

  const handleSaveCameraSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/camera/settings/save`, { method: 'POST' });
      if (res.ok) {
        showToast('Camera & processing settings saved to JSON as startup defaults');
      } else {
        showToast('Failed to save settings', 3000);
      }
    } catch (e: any) {
      showToast(`Error saving settings: ${e.message}`, 4000);
    }
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

  const PLANET_NAMES = ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'moon', 'sun'];

  const isPlanetQuery = (q: string) => {
    const lower = q.trim().toLowerCase();
    return PLANET_NAMES.includes(lower) || targets.some(t => t.type === 'planet' && (
      t.name.toLowerCase() === lower || (t.alt && t.alt.toLowerCase() === lower)
    ));
  };

  const searchSky = (text: string = searchText) => {
    const query = text.trim();
    if (!query || searching) return;
    setSearchIsOpen(false);

    // If it's a planet or Moon/Sun, compute live ephemeris via backend (don't use stale history/cache)
    if (!isPlanetQuery(query)) {
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
    }
    // 3. Fallback to backend live ephemeris / SIMBAD query
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
        setStreamEpoch(Date.now());
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

  const handleSwitchCameraFormat = (fmt: 'YUYV' | 'MJPG') => {
    setStatus(`Setting camera format to ${fmt}...`);
    fetch(`${API_BASE}/camera/format`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: fmt })
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setCameraFormat(data.camera_format);
          setStreamEpoch(Date.now());
          showToast(`Camera format set to ${data.camera_format}`);
        } else {
          showToast(`Format switch failed: ${data.error || 'Check camera support'}`, 4000);
        }
      })
      .catch(e => showToast(`Error: ${e.message}`, 4000));
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

  const handleRestart = async () => {
    if (!window.confirm("Restart AstroCam service? Live streaming and camera connection will briefly reset.")) {
      return;
    }
    setIsRestarting(true);
    setStatus("Restarting AstroCam service...");
    try {
      await fetch(`${API_BASE}/system/restart`, { method: 'POST' });
      showToast("Service restarting... Reconnecting soon.", 4000);
    } catch (e: any) {
      console.error("Error triggering restart:", e);
    }

    // Wait 3 seconds, then poll /status until the backend is back online
    setTimeout(() => {
      const startTime = Date.now();
      const pollInterval = setInterval(async () => {
        try {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 2000);
          const res = await fetch(`${API_BASE}/status`, { signal: controller.signal });
          clearTimeout(timeoutId);
          if (res.ok) {
            clearInterval(pollInterval);
            setIsRestarting(false);
            showToast("AstroCam restarted successfully!", 3000);
            setTimeout(() => {
              window.location.reload();
            }, 600);
          }
        } catch {
          if (Date.now() - startTime > 25000) {
            clearInterval(pollInterval);
            setIsRestarting(false);
            showToast("Restart timeout. Please check system status or refresh.", 5000);
          }
        }
      }, 1500);
    }, 2500);
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
            <button 
              className="restart-btn" 
              onClick={handleRestart} 
              disabled={isRestarting}
              title="Restart AstroCam service"
            >
              <RotateCcw size={12} className={isRestarting ? 'spin' : ''} />
              {isRestarting ? 'Restarting...' : 'Restart'}
            </button>
          </div>
        </header>

        <div 
          className="stream-container"
          onPointerDown={handlePointerDownStream}
          onPointerMove={handlePointerMoveStream}
          onPointerUp={handlePointerUpStream}
          onPointerCancel={handlePointerUpStream}
          title="Drag on stream to select Region of Interest (ROI)"
        >
          {/* Astrophotography Display Contrast / Stretch Filter (GPU accelerated) */}
          <svg style={{ position: 'absolute', width: 0, height: 0, pointerEvents: 'none' }} aria-hidden="true">
            <filter id="canvas-stretch" colorInterpolationFilters="sRGB">
              <feComponentTransfer>
                <feFuncR type="linear" slope={stretchSlope} intercept={stretchIntercept} />
                <feFuncG type="linear" slope={stretchSlope} intercept={stretchIntercept} />
                <feFuncB type="linear" slope={stretchSlope} intercept={stretchIntercept} />
              </feComponentTransfer>
            </filter>
          </svg>

          <img 
            key={streamEpoch}
            src={`${API_BASE}/stream?t=${streamEpoch}`} 
            alt="Live Stream" 
            className="video-preview" 
            style={{ filter: isDisplayStretched ? 'url(#canvas-stretch)' : undefined }}
            onError={() => {
              setTimeout(() => setStreamEpoch(Date.now()), 1500);
            }}
          />
          
          {/* ROI Bounding Box Overlay */}
          {(() => {
            if (!showRoiBox || !liveProcessing.histogram) return null;
            const fw = health.width || 1920;
            const fh = health.height || 1080;
            const activeRoi = liveProcessing.roi || [0, 0, fw, fh];
            const isEditing = (
              roiInputs.x !== activeRoi[0] ||
              roiInputs.y !== activeRoi[1] ||
              roiInputs.w !== activeRoi[2] ||
              roiInputs.h !== activeRoi[3]
            );
            const hasActiveCustomRoi = (
              activeRoi[0] > 0 || activeRoi[1] > 0 || activeRoi[2] < fw || activeRoi[3] < fh
            );
            const hasPreviewCustomRoi = (
              roiInputs.x > 0 || roiInputs.y > 0 || roiInputs.w < fw || roiInputs.h < fh
            );

            return (
              <>
                {/* Proposed / Preview ROI box while editing */}
                {isEditing && hasPreviewCustomRoi && (
                  <div
                    className="roi-bounding-box preview"
                    style={{
                      left: `${(roiInputs.x / fw) * 100}%`,
                      top: `${(roiInputs.y / fh) * 100}%`,
                      width: `${(roiInputs.w / fw) * 100}%`,
                      height: `${(roiInputs.h / fh) * 100}%`,
                    }}
                  >
                    <div className="roi-corner tl" />
                    <div className="roi-corner tr" />
                    <div className="roi-corner bl" />
                    <div className="roi-corner br" />
                    <span className="roi-bounding-box-label">
                      PREVIEW ROI: {roiInputs.w}x{roiInputs.h} ({roiInputs.x}, {roiInputs.y})
                    </span>
                  </div>
                )}

                {/* Active applied ROI box */}
                {hasActiveCustomRoi && (
                  <div
                    className="roi-bounding-box"
                    style={{
                      left: `${(activeRoi[0] / fw) * 100}%`,
                      top: `${(activeRoi[1] / fh) * 100}%`,
                      width: `${(activeRoi[2] / fw) * 100}%`,
                      height: `${(activeRoi[3] / fh) * 100}%`,
                    }}
                  >
                    <div className="roi-corner tl" />
                    <div className="roi-corner tr" />
                    <div className="roi-corner bl" />
                    <div className="roi-corner br" />
                    <span className="roi-bounding-box-label">
                      ROI: {activeRoi[2]}x{activeRoi[3]} ({activeRoi[0]}, {activeRoi[1]})
                    </span>
                  </div>
                )}
              </>
            );
          })()}

          <div className="stream-overlay">
            <span>{health.width || 0}x{health.height || 0} @ {(health.fps || 0).toFixed(1)} FPS • {health.camera_format || cameraFormat}</span>
            {accumulationCount > 1 ? (
              <span style={{ color: '#58a6ff', fontWeight: 600 }}>
                Stack: {health.acc_count ?? 1}/{accumulationCount} ({accumulationCount}x)
              </span>
            ) : (
              <span>Stack: 1x (Raw)</span>
            )}
            <span>Luminance: {(health.mean_brightness || 0).toFixed(1)}</span>
            {(health.corrupted_frames_dropped ?? 0) > 0 && (
              <span style={{ color: '#f85149', fontWeight: 600 }} title="Corrupted glitch frames dropped due to sudden star peak position jump">
                Corrupted: {health.corrupted_frames_dropped} dropped
              </span>
            )}
            {sky && (
              <span title="Azimuth from north / altitude above the horizon">
                Az {sky.azimuth.toFixed(1)}° · Alt {sky.altitude.toFixed(1)}°
              </span>
            )}
          </div>
          {liveProcessing.histogram && (
            <div className="stream-histogram">
              <span 
                className="stream-histogram-label" 
                title="Drag sky-blue / amber cursors to stretch display contrast. Double-click to reset."
              >
                HIST (LOG){isDisplayStretched ? ` • STRETCH [${blackPoint}–${whitePoint}]` : ''}{liveProcessing.roi && (liveProcessing.roi[2] !== 1920 || liveProcessing.roi[3] !== 1080) ? ` • ROI ${liveProcessing.roi[2]}x${liveProcessing.roi[3]}` : ''} • {liveProcessing.stride || 4}x
              </span>
              <canvas 
                ref={histCanvasRef} 
                width={512} 
                height={54} 
                className="stream-histogram-canvas" 
                title="Drag sky-blue (black point) or amber (white point) cursor to stretch display contrast. Double-click to reset."
                onPointerDown={handleHistPointerDown}
                onPointerMove={handleHistPointerMove}
                onPointerUp={handleHistPointerUp}
                onPointerCancel={handleHistPointerUp}
                onDoubleClick={handleHistDoubleClick}
              />
            </div>
          )}
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
                                            if (isPlanetQuery(h.name)) {
                                              searchSky(h.name);
                                            } else {
                                              setSearchIsOpen(false);
                                              setSearchText('');
                                              setSearchMsg(`Pointing at ${h.name}`);
                                              applySky({ ra: h.ra, dec: h.dec, target: h.name });
                                            }
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
                                          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                            {isPlanetQuery(h.name) ? '🪐 ' : ''}{h.name}
                                          </span>
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
                                        const isPlanet = t.type === 'planet' || isPlanetQuery(t.name);
                                        return (
                                          <div
                                            key={`t-${t.name}`}
                                            onClick={() => {
                                              if (isPlanet) {
                                                searchSky(t.name);
                                              } else {
                                                setSearchIsOpen(false);
                                                setSearchText('');
                                                setSearchMsg(`Pointing at ${label}`);
                                                applySky({ ra: t.ra, dec: t.dec, target: label });
                                              }
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
                                            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                              {isPlanet ? '🪐 ' : ''}{t.name}
                                            </span>
                                            <span style={{ fontSize: '10px', color: isPlanet ? 'var(--accent)' : 'var(--text-muted)' }}>
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

            <div className="control-group" style={{ marginTop: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <label style={{ margin: 0 }}>Black Calibration</label>
                {calibrationStatus.calibrated ? (
                  <span style={{ fontSize: '11px', color: '#10b981', fontWeight: 600 }}>
                    Calibrated ({calibrationStatus.meta?.defect_pixel_count ?? 0} hot px)
                  </span>
                ) : (
                  <span style={{ fontSize: '11px', color: '#9ca3af' }}>Uncalibrated</span>
                )}
              </div>
              <div className="preset-row">
                <button 
                  type="button"
                  className={calibrationStatus.calibrated ? 'active' : ''} 
                  onClick={handleStartCalibration}
                  disabled={calibrationStatus.is_calibrating || recordingStatus.is_recording}
                  title="Cover lens or point to pitch-black sky to calibrate sensor dark level and map defective hot pixels for mosaic weighting."
                >
                  {calibrationStatus.is_calibrating 
                    ? `Calibrating (${calibrationStatus.frames_collected}/${calibrationStatus.target_frames})...` 
                    : (calibrationStatus.calibrated ? 'Recalibrate' : 'Black Calibration')}
                </button>
                {calibrationStatus.calibrated && (
                  <button
                    type="button"
                    onClick={handleClearCalibration}
                    disabled={calibrationStatus.is_calibrating}
                    title="Clear black calibration and revert to uncalibrated"
                    style={{ maxWidth: '75px', background: 'rgba(239, 68, 68, 0.15)', borderColor: 'rgba(239, 68, 68, 0.4)', color: '#f87171' }}
                  >
                    Clear
                  </button>
                )}
              </div>
              <div style={{ fontSize: '11px', color: '#9ca3af', marginTop: '4px' }}>
                {calibrationStatus.calibrated
                  ? `Mapped ${calibrationStatus.meta?.defect_pixel_count} defect px (threshold >${calibrationStatus.meta?.threshold} ADU). Zero-weighted in mosaics.`
                  : 'Cover lens with cap before clicking to map hot pixels for mosaic down-weighting.'}
              </div>
            </div>
          </div>

          <div className="control-section">
            <div className="section-header"><Zap size={16} /> Live Processing Controls</div>

            <div className="control-group">
              <label>Pipeline Stages</label>
              <div className="preset-row wrap">
                <button
                  type="button"
                  className={liveProcessing.histogram ? 'active' : ''}
                  onClick={() => handleToggleProcessing('histogram')}
                  title="Toggle real-time histogram calculation"
                >
                  Histogram: {liveProcessing.histogram ? 'ON' : 'OFF'}
                </button>
                <button
                  type="button"
                  className={liveProcessing.corrupted_filter ? 'active' : ''}
                  onClick={() => handleToggleProcessing('corrupted_filter')}
                  title="Drop corrupted glitch frames due to USB line drop / star peak jump"
                >
                  Glitch Filter: {liveProcessing.corrupted_filter ? 'ON' : 'OFF'}
                </button>
                <button
                  type="button"
                  className={liveProcessing.luminance_check ? 'active' : ''}
                  onClick={() => handleToggleProcessing('luminance_check')}
                  title="Full-frame peak luminance and star center detection"
                >
                  Luminance: {liveProcessing.luminance_check ? 'ON' : 'OFF'}
                </button>
              </div>
            </div>

            <div className="control-group">
              <label>Histogram Region of Interest (ROI)</label>
              <div className="roi-grid">
                <div className="roi-field">
                  <span>X (Start):</span>
                  <input
                    type="number"
                    value={roiInputs.x}
                    min={0}
                    max={1870}
                    step={50}
                    onChange={e => setRoiInputs({ ...roiInputs, x: parseInt(e.target.value) || 0 })}
                  />
                </div>
                <div className="roi-field">
                  <span>Y (Start):</span>
                  <input
                    type="number"
                    value={roiInputs.y}
                    min={0}
                    max={1030}
                    step={50}
                    onChange={e => setRoiInputs({ ...roiInputs, y: parseInt(e.target.value) || 0 })}
                  />
                </div>
                <div className="roi-field">
                  <span>Width (W):</span>
                  <input
                    type="number"
                    value={roiInputs.w}
                    min={50}
                    max={1920}
                    step={50}
                    onChange={e => setRoiInputs({ ...roiInputs, w: parseInt(e.target.value) || 100 })}
                  />
                </div>
                <div className="roi-field">
                  <span>Height (H):</span>
                  <input
                    type="number"
                    value={roiInputs.h}
                    min={50}
                    max={1080}
                    step={50}
                    onChange={e => setRoiInputs({ ...roiInputs, h: parseInt(e.target.value) || 100 })}
                  />
                </div>
              </div>
              <div className="preset-row">
                <button type="button" onClick={handleApplyRoi} className="active" title="Apply Region of Interest">
                  Apply ROI
                </button>
                <button type="button" onClick={handleResetRoi} title="Reset to Full Frame 1920x1080">
                  Full Frame
                </button>
                <button
                  type="button"
                  className={showRoiBox ? 'active' : ''}
                  onClick={() => setShowRoiBox(prev => !prev)}
                  title="Toggle ROI Bounding Box overlay on stream"
                >
                  Box: {showRoiBox ? 'ON' : 'OFF'}
                </button>
              </div>
            </div>

            <div className="control-group">
              <label>Histogram Stride (Subsampling)</label>
              <div className="preset-row">
                {[1, 2, 4, 8, 16].map(s => (
                  <button
                    key={s}
                    type="button"
                    className={(liveProcessing.stride || 4) === s ? 'active' : ''}
                    onClick={() => handleSetStride(s)}
                    title={`Subsample every ${s} pixel${s > 1 ? 's' : ''}${s === 1 ? ' (Full Resolution)' : s === 4 ? ' (Default)' : ' (Higher FPS)'}`}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="control-section">
            <div className="section-header" style={{ justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <RefreshCw size={16} /> Camera Settings
              </div>
              <button
                type="button"
                className="save-defaults-btn"
                onClick={handleSaveCameraSettings}
                title="Save current camera and processing settings to config.json as startup defaults"
              >
                <Save size={12} /> Save Defaults
              </button>
            </div>

            <div className="control-group">
              <label>Pixel Format</label>
              <div className="preset-row">
                <button 
                  type="button"
                  className={cameraFormat === 'YUYV' ? 'active' : ''} 
                  onClick={() => handleSwitchCameraFormat('YUYV')}
                  title="Uncompressed YUYV (4:2:2). Recommended for astrophotography to prevent lossy JPEG macroblock compression and preserve true sensor noise."
                >
                  YUYV
                </button>
                <button 
                  type="button"
                  className={cameraFormat === 'MJPG' ? 'active' : ''} 
                  onClick={() => handleSwitchCameraFormat('MJPG')}
                  title="Hardware MJPEG compressed stream. Use as fallback if YUYV experiences driver or USB throughput issues."
                >
                  MJPG
                </button>
              </div>
            </div>

            <div className="control-group">
              <label>Auto Exposure</label>
              <div className="preset-row">
                <button 
                  type="button"
                  className={!controls.auto_exposure ? 'active' : ''} 
                  onClick={() => updateControl('auto_exposure', 0)}
                  title="Manual exposure mode"
                >
                  Manual (0)
                </button>
                <button 
                  type="button"
                  className={controls.auto_exposure ? 'active' : ''} 
                  onClick={() => updateControl('auto_exposure', 1)}
                  title="Automatic exposure mode (Aperture Priority AEC)"
                >
                  Auto (1)
                </button>
              </div>
            </div>

            {/* Setting-Indexed Baseline Dark Calibration Control */}
            <div className="control-group" style={{ marginBottom: '14px', paddingBottom: '10px', borderBottom: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <label style={{ margin: 0, fontWeight: 600, fontSize: '12px' }}>
                  Baseline ({baselineKey})
                </label>
                {baselineStatus.exists ? (
                  <span style={{ fontSize: '11px', color: '#10b981', fontWeight: 600 }}>
                    Saved
                  </span>
                ) : (
                  <span style={{ fontSize: '11px', color: '#f59e0b' }}>
                    No baseline
                  </span>
                )}
              </div>
              <div className="preset-row" style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                <button
                  type="button"
                  className={baselineStatus.exists ? '' : 'active'}
                  onClick={handleCaptureBaseline}
                  disabled={baselineStatus.is_capturing}
                  title="Cover lens with cap before clicking Capture to record the fixed dark offset baseline for this specific camera setting."
                  style={{ flex: 1.2 }}
                >
                  {baselineStatus.is_capturing ? 'Capturing...' : (baselineStatus.exists ? 'Recapture' : 'Capture')}
                </button>
                <button
                  type="button"
                  className={baselineStatus.subtract ? 'active' : ''}
                  onClick={handleToggleSubtract}
                  disabled={!baselineStatus.exists}
                  title={baselineStatus.exists ? (baselineStatus.subtract ? "Baseline subtraction active (click to disable)" : "Click to enable baseline subtraction") : "Capture a baseline for this setting first to enable subtraction"}
                  style={{ flex: 1.2 }}
                >
                  {baselineStatus.subtract ? 'Subtract: ON' : 'Subtract'}
                </button>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginLeft: '2px' }}>
                  <span style={{ fontSize: '11px', color: '#8b949e' }}>Pedestal:</span>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={1}
                    value={pedestalInput}
                    onChange={e => handleSetPedestal(parseInt(e.target.value) || 0)}
                    style={{
                      width: '44px',
                      padding: '4px 4px',
                      fontSize: '12px',
                      background: '#0d1117',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      borderRadius: '4px',
                      color: '#c9d1d9',
                      textAlign: 'center'
                    }}
                    title="Fixed positive pedestal added after baseline subtraction to prevent zero-clipping (default: 10)"
                  />
                </div>
              </div>
            </div>

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
                minVal = -10;
                maxVal = 266;
                const note = value === 0 ? ' (0 = Neutral Black Level Offset)' : (value || 0) < 0 ? ' (Low Offset - Warning: 0-Clipping)' : ' (High Pedestal)';
                labelText = `Brightness: ${value}${note}`;
              } else if (key === 'gain') {
                minVal = 0;
                maxVal = 128;
                const note = value === 0 ? ' (0 = Min Gain, Noise Clipped)' : value === 64 ? ' (64 = Recommended Astro Gain)' : ` (${value}/128 Analog Gain)`;
                labelText = `Gain: ${value}${note}`;
              } else if (key === 'contrast') {
                minVal = 0;
                maxVal = 100;
                const note = value === 50 ? ' (50 = Linear Default)' : '';
                labelText = `Contrast: ${value}${note}`;
              } else if (key === 'saturation') {
                minVal = 0;
                maxVal = 100;
                const note = value === 0 ? ' (0 = Monochrome, No Color Distortion)' : value === 64 ? ' (64 = Default Color)' : ` (${value}/100)`;
                labelText = `Saturation: ${value}${note}`;
              } else if (key === 'sharpness') {
                minVal = 0;
                maxVal = 100;
                const note = value === 50 ? ' (50 = Neutral Pass-through)' : (value || 0) < 50 ? ' (<50 = FW Low-pass Denoising/Blur)' : ' (>50 = Edge Sharpening)';
                labelText = `Sharpness: ${value}${note}`;
              } else if (key === 'gamma') {
                minVal = 100;
                maxVal = 500;
                stepVal = 1;
                const gammaRatio = ((value || 100) / 100).toFixed(2);
                const note = value === 100 ? ' (100 = 1.0γ Linear)' : value === 300 ? ' (300 = 3.0γ Default)' : ` (${gammaRatio}γ)`;
                labelText = `Gamma: ${value ?? 100}${note}`;
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
