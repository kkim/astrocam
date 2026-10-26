import { useState, useEffect, useRef, memo } from 'react';
import './App.css';
import { Camera, Sliders, Zap, Menu, X, Terminal, Grid, RefreshCw, RotateCcw, Compass, Circle, Square, Layers, Sparkles, Save, Clock, Star } from 'lucide-react';

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

interface MosaicMeta {
  timestamp?: number;
  elapsed_sec?: number;
  canvas_size?: [number, number];
  frame_size?: [number, number];
  total_files?: number;
  valid_files?: number;
  aligned_frames?: number;
  skipped_no_star?: number;
  skipped_corrupt?: number;
  skipped_step_gate?: number;
  frames_with_stripes?: number;
  drift_span_px?: [number, number];
  drift_angle_deg?: number | null;
  drift_speed_px_per_frame?: number | null;
  pair_angle_deg?: number | null;
  constellation_stars?: number;
  constellation_offsets?: [number, number][];
  pair_matches?: number;
  single_star_matches?: number;
  identity_swaps_prevented?: number;
  static_peaks_rejected?: number;
  static_peak_coords?: [number, number, number][];
  defect_mask_used?: boolean;
  raw_defects_count?: number;
  masked_px_count?: number;
  align_mode?: string;
  params?: {
    align_mode?: 'constellation' | 'single';
    min_snr?: number;
    max_step_px?: number | null;
    static_filter?: boolean;
    defect_radius?: number;
    defect_soft_px?: number;
    use_defect_mask?: boolean;
    feather_px?: number;
    stretch_method?: string;
    stretch_factor?: number;
    fov_mode?: string;
  };
}

interface SessionNavInfo {
  target?: string;
  stars?: string;
  ra?: number;
  dec?: number;
  backend?: string;
  max_mag?: number;
  camera_rotation_deg?: number;
  camera_flip_x?: boolean;
  hist_min?: number;
  hist_max?: number;
  anchor_x?: number;
  anchor_y?: number;
  show_crosshairs?: boolean;
  favorite?: boolean;
  altitude?: number;
  azimuth?: number;
  latitude?: number;
  longitude?: number;
  focal_length_mm?: number;
  pixel_size_um?: number;
  orientation?: string;
  note?: string;
  utc?: string;
  camera_settings?: Record<string, any>;
}

interface MosaicCatalogStar {
  rank: number;
  name: string;
  raw_name?: string;
  short_name: string;
  mag: number;
  ra: number;
  dec: number;
  dx_px: number;
  dy_px: number;
  dist_px: number;
  pa_deg: number;
}

interface GalleryItem {
  id: string;
  type?: 'sequence' | 'image';
  name?: string;
  favorite?: boolean;
  target?: string;
  stars?: string;
  ra?: number;
  dec?: number;
  orientation?: string;
  note?: string;
  nav?: SessionNavInfo;
  time?: string;
  date?: string;
  datetime?: string;
  frames?: number;
  first_frame?: string;
  thumbnail: string;
  url: string;
  first_frame_url?: string;
  format?: string;
  camera_settings?: Record<string, any>;
  has_mosaic?: boolean;
  mosaic_status?: 'ready' | 'processing' | 'queued' | 'none';
  mosaic_url?: string;
  mosaic_mtime?: number;
  mosaic_progress?: {
    percent: number;
    message?: string;
    stage?: string;
    step?: number;
    total_steps?: number;
    current?: number;
    total?: number;
  };
  mosaic_queue_pos?: number;
  mosaic_meta?: MosaicMeta;
}

const formatGalleryItem = (item: any): GalleryItem => {
  if (typeof item === 'string') {
    return {
      id: item,
      type: 'image',
      favorite: false,
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
    favorite: Boolean(item.favorite || item.nav?.favorite),
    thumbnail: thumbUrl,
    url: fullUrl,
    mosaic_url: mosaicUrl,
    mosaic_status: item.mosaic_status || 'none',
    mosaic_progress: item.mosaic_progress || undefined,
    mosaic_queue_pos: item.mosaic_queue_pos || 0,
    has_mosaic: !!item.has_mosaic
  };
};

const GalleryCard = memo(({
  item,
  onSelect,
  onToggleFavorite,
}: {
  item: GalleryItem;
  onSelect: (item: GalleryItem) => void;
  onToggleFavorite?: (item: GalleryItem) => void;
}) => {
  const isSeq = item.type === 'sequence';
  const statusDesc = item.mosaic_status === 'ready' 
    ? 'Ready (Click to View)' 
    : item.mosaic_status === 'processing' 
      ? (item.mosaic_progress?.message || 'Processing...') 
      : item.mosaic_status === 'queued'
        ? `Queued (#${item.mosaic_queue_pos || 1} in line)`
        : 'Not Created (Click to Stitch Mosaic)';
  const titleTip = isSeq
    ? `${item.datetime || item.id} • ${item.frames || 1} frames • Mosaic: ${statusDesc}`
    : `${item.datetime || item.id} • Single frame`;

  return (
    <div 
      className="capture-item" 
      title={titleTip} 
      onClick={() => onSelect(item)}
    >
      <img src={item.thumbnail} alt={item.datetime || item.id} loading="lazy" decoding="async" />

      {/* Favorite Button on Card */}
      <button
        type="button"
        className={`capture-fav-btn ${item.favorite ? 'active' : ''}`}
        title={item.favorite ? 'Favorited (Click to remove from favorites)' : 'Favorite this capture'}
        onClick={(e) => {
          e.stopPropagation();
          onToggleFavorite?.(item);
        }}
      >
        <Star
          size={13}
          fill={item.favorite ? '#fbbf24' : 'none'}
          color={item.favorite ? '#fbbf24' : 'rgba(255, 255, 255, 0.75)'}
        />
      </button>

      {isSeq && (
        <>
          {item.mosaic_status === 'ready' && (
            <div className="mosaic-badge ready" title="16-bit Mosaic Ready (Click to View)">
              <Sparkles size={11} />
              <span>Mosaic Ready</span>
            </div>
          )}
          {item.mosaic_status === 'processing' && (
            <div 
              className="mosaic-badge processing" 
              title={item.mosaic_progress?.message || "Mosaic is being processed..."}
            >
              <RefreshCw size={11} className="spin" />
              <span>
                {item.mosaic_progress && item.mosaic_progress.percent !== undefined && item.mosaic_progress.percent > 0
                  ? `${Math.round(item.mosaic_progress.percent)}%`
                  : 'Processing...'}
              </span>
            </div>
          )}
          {item.mosaic_status === 'queued' && (
            <div 
              className="mosaic-badge queued" 
              title={`Queued (#${item.mosaic_queue_pos || 1} in line)`}
            >
              <Clock size={11} />
              <span>Queued #{item.mosaic_queue_pos || 1}</span>
            </div>
          )}
          {item.mosaic_status === 'none' && (
            <div className="mosaic-badge none" title="Click thumbnail to stitch mosaic">
              <Layers size={11} />
              <span>Stitch Mosaic</span>
            </div>
          )}
          {item.mosaic_status === 'processing' && item.mosaic_progress && (
            <div className="mosaic-progress-bar" title={item.mosaic_progress.message}>
              <div 
                className="mosaic-progress-fill" 
                style={{ width: `${Math.min(100, Math.max(4, item.mosaic_progress.percent || 0))}%` }} 
              />
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
        {(item.stars || item.target) && (
          <div style={{ fontWeight: 600, color: 'var(--accent)', fontSize: '11px', marginBottom: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            ✨ {item.stars || item.target}
          </div>
        )}
        {(item.ra !== undefined || item.orientation) && (
          <div style={{ fontSize: '10px', color: '#a5d6ff', marginBottom: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {item.ra !== undefined && item.dec !== undefined ? `RA ${Number(item.ra).toFixed(2)}° Dec ${Number(item.dec).toFixed(2)}°` : ''}
            {item.orientation ? `${item.ra !== undefined ? ' · ' : ''}🧭 ${item.orientation}` : ''}
          </div>
        )}
        {item.note && (
          <div style={{ fontSize: '10px', color: '#d2a8ff', marginBottom: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={item.note}>
            📝 {item.note}
          </div>
        )}
        {item.camera_settings && Object.keys(item.camera_settings).length > 0 && (
          <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {item.camera_settings.exposure !== undefined ? `${Math.round(item.camera_settings.exposure * 0.1)}ms` : ''}
            {item.camera_settings.gain !== undefined ? ` · G${item.camera_settings.gain}` : ''}
            {item.camera_settings.accumulation_count > 1 ? ` · ${item.camera_settings.accumulation_count}x` : ''}
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
}, (prev, next) => (
  prev.item.id === next.item.id &&
  prev.item.favorite === next.item.favorite &&
  prev.item.thumbnail === next.item.thumbnail &&
  prev.item.frames === next.item.frames &&
  prev.item.mosaic_status === next.item.mosaic_status &&
  prev.item.mosaic_queue_pos === next.item.mosaic_queue_pos &&
  prev.item.mosaic_progress?.percent === next.item.mosaic_progress?.percent &&
  prev.item.mosaic_progress?.message === next.item.mosaic_progress?.message &&
  prev.item.target === next.item.target &&
  prev.item.stars === next.item.stars &&
  prev.item.orientation === next.item.orientation &&
  prev.item.note === next.item.note &&
  prev.item.ra === next.item.ra &&
  prev.item.dec === next.item.dec
));

interface Linear16MosaicData {
  width: number;
  height: number;
  channels: number;
  autoMin: number;
  autoMax: number;
  minNz: number;
  maxVal: number;
  pixels: Uint16Array;
  histogram: number[];
}

const MosaicViewerModal = ({
  item,
  onClose,
  isStandaloneTab = false,
}: {
  item: GalleryItem;
  onClose: () => void;
  isStandaloneTab?: boolean;
}) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [mosaicData, setMosaicData] = useState<Linear16MosaicData | null>(null);
  const [minPoint, setMinPoint] = useState<number>(0);
  const [maxPoint, setMaxPoint] = useState<number>(65535);
  const [activeHandle, setActiveHandle] = useState<'min' | 'max' | null>(null);
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoverInfo, setHoverInfo] = useState<{ x: number; y: number; val: number } | null>(null);

  // Stack Note, Orientation, Diagnostics & Stitch Controls state
  const [showInfoPanel, setShowInfoPanel] = useState<boolean>(true);
  const [showStitchControls, setShowStitchControls] = useState<boolean>(false);
  const [mosaicMeta, setMosaicMeta] = useState<MosaicMeta | null>(item.mosaic_meta || null);
  const [navInfo, setNavInfo] = useState<SessionNavInfo>(item.nav || {});
  const navInfoRef = useRef<SessionNavInfo>(item.nav || {});
  const [isFavorite, setIsFavorite] = useState<boolean>(Boolean(item.favorite || item.nav?.favorite));
  const [viewSavedMsg, setViewSavedMsg] = useState<string>('');
  const [mosaicStatus, setMosaicStatus] = useState<string>(item.mosaic_status || 'ready');
  const [mosaicProgress, setMosaicProgress] = useState<{ percent: number; message?: string; stage?: string } | null>(
    item.mosaic_progress || null
  );
  const [reloadCounter, setReloadCounter] = useState<number>(0);

  // Stack Note form fields
  const [noteStars, setNoteStars] = useState<string>(item.stars || item.target || '');
  const [noteRa, setNoteRa] = useState<string>(item.ra !== undefined ? String(item.ra) : '');
  const [noteDec, setNoteDec] = useState<string>(item.dec !== undefined ? String(item.dec) : '');
  const [noteOrientation, setNoteOrientation] = useState<string>(item.orientation || '');
  const [noteText, setNoteText] = useState<string>(item.note || '');
  const [noteSaving, setNoteSaving] = useState<boolean>(false);
  const [noteSaveMsg, setNoteSaveMsg] = useState<string>('');
  const noteInitializedRef = useRef<boolean>(false);

  // Sky Catalog, Target Autocomplete, Camera Rotation & Star Crosshairs state
  const [targets, setTargets] = useState<TargetItem[]>([]);
  const [searchText, setSearchText] = useState<string>(item.target || '');
  const [searchIsOpen, setSearchIsOpen] = useState<boolean>(false);
  const [skyBackend, setSkyBackend] = useState<string>(item.nav?.backend || 'astropy');
  const [skyBackends, setSkyBackends] = useState<string[]>(['skyfield', 'astropy']);
  const [catalogStars, setCatalogStars] = useState<MosaicCatalogStar[]>([]);
  const [totalCatalogStars, setTotalCatalogStars] = useState<number>(0);
  const [skyLoading, setSkyLoading] = useState<boolean>(false);
  const [skyMsg, setSkyMsg] = useState<string>('');
  const [cameraRotationDeg, setCameraRotationDeg] = useState<number>(item.nav?.camera_rotation_deg ?? 0);
  const [cameraFlipX, setCameraFlipX] = useState<boolean>(Boolean(item.nav?.camera_flip_x));
  const [showStarCrosshairs, setShowStarCrosshairs] = useState<boolean>(true);
  const [maxMag, setMaxMag] = useState<number>(item.nav?.max_mag ?? 8.5);
  const [detectedPeaks, setDetectedPeaks] = useState<{ x: number; y: number; raw: number }[]>([]);
  const [anchorPos, setAnchorPos] = useState<{ x: number; y: number } | null>(null);
  const [anchorPeakIdx, setAnchorPeakIdx] = useState<number>(0);
  const skyInitializedRef = useRef<boolean>(false);

  const visibleCatalogStars = catalogStars.filter((s, idx) => idx === 0 || s.mag <= maxMag);

  // Stitch parameters
  const [alignMode, setAlignMode] = useState<'constellation' | 'single'>('constellation');
  const [minSnr, setMinSnr] = useState<number>(3.0);
  const [maxStepPx, setMaxStepPx] = useState<number>(0); // 0 = Auto
  const [staticFilter, setStaticFilter] = useState<boolean>(true);
  const [defectRadius, setDefectRadius] = useState<number>(1);
  const [useDefectMask, setUseDefectMask] = useState<boolean>(true);
  const [featherPx, setFeatherPx] = useState<number>(12);
  const paramsInitializedRef = useRef<boolean>(false);

  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const histCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const minPointRef = useRef<number>(0);
  const maxPointRef = useRef<number>(65535);
  const dragPointerRef = useRef<{ lastX: number } | null>(null);
  const panDragRef = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(null);

  const queryMosaicCatalog = async (opts?: {
    query?: string;
    backend?: string;
    ra?: number;
    dec?: number;
  }) => {
    setSkyLoading(true);
    setSkyMsg('');
    try {
      const payload: Record<string, any> = {
        backend: opts?.backend || skyBackend,
        top_n: 50,
      };
      if (opts?.query !== undefined && opts.query.trim() !== '') {
        payload.query = opts.query.trim();
      } else {
        const rVal = opts?.ra !== undefined ? opts.ra : parseFloat(noteRa);
        const dVal = opts?.dec !== undefined ? opts.dec : parseFloat(noteDec);
        if (!isNaN(rVal)) payload.ra = rVal;
        if (!isNaN(dVal)) payload.dec = dVal;
      }
      const res = await fetch(`${API_BASE}/mosaic/sky/${encodeURIComponent(item.id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.success) {
        if (data.backends && Array.isArray(data.backends)) setSkyBackends(data.backends);
        if (data.backend) setSkyBackend(data.backend);
        if (data.ra !== undefined) setNoteRa(String(data.ra));
        if (data.dec !== undefined) setNoteDec(String(data.dec));
        if (data.target && opts?.query) {
          setSearchText(data.target);
          setNoteStars(data.target);
        }
        setNavInfo((prev) => ({
          ...prev,
          target: data.target || prev.target,
          ra: data.ra ?? prev.ra,
          dec: data.dec ?? prev.dec,
          altitude: data.altitude ?? prev.altitude,
          azimuth: data.azimuth ?? prev.azimuth,
          backend: data.backend ?? prev.backend,
        }));
        setCatalogStars(data.top_stars || []);
        setTotalCatalogStars(data.total_catalog_stars || 0);
        if (opts?.query) {
          setSkyMsg(`Resolved ${data.target} (${(data.top_stars || []).length} top / ${data.total_catalog_stars || 0} stars)`);
        }
      } else {
        setSkyMsg(data.error || 'Catalog lookup failed');
      }
    } catch (e: any) {
      setSkyMsg(`Error: ${e.message}`);
    } finally {
      setSkyLoading(false);
    }
  };

  // Load autocomplete targets catalog once when modal opens
  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/mock/sky/targets`)
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => {
        if (active && Array.isArray(list)) setTargets(list);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const titleLabel = noteStars || item.target || '';
    if (isStandaloneTab) {
      document.title = `Mosaic • ${titleLabel ? titleLabel + ' • ' : ''}${item.datetime || item.id}`;
      return;
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose, isStandaloneTab, noteStars, item.target, item.datetime, item.id]);

  // Fetch mosaic status, diagnostics, and stack note metadata; poll while processing/queued
  useEffect(() => {
    let active = true;
    let timer: any = null;

    const pollStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/mosaic/status/${encodeURIComponent(item.id)}`);
        if (!res.ok || !active) return;
        const data = await res.json();
        if (!active) return;

        if (data.mosaic_meta) {
          setMosaicMeta(data.mosaic_meta);
          if (!paramsInitializedRef.current && data.mosaic_meta.params) {
            paramsInitializedRef.current = true;
            const p = data.mosaic_meta.params;
            if (p.align_mode) setAlignMode(p.align_mode);
            if (p.min_snr !== undefined) setMinSnr(p.min_snr);
            if (p.max_step_px !== undefined && p.max_step_px !== null) setMaxStepPx(p.max_step_px);
            if (p.static_filter !== undefined) setStaticFilter(p.static_filter);
            if (p.defect_radius !== undefined) setDefectRadius(p.defect_radius);
            if (p.use_defect_mask !== undefined) setUseDefectMask(p.use_defect_mask);
            if (p.feather_px !== undefined) setFeatherPx(p.feather_px);
          }
        }

        if (data.nav) {
          setNavInfo(data.nav);
          navInfoRef.current = data.nav;
          if (!noteInitializedRef.current) {
            noteInitializedRef.current = true;
            setNoteStars(data.nav.stars || data.nav.target || item.target || '');
            setSearchText(data.nav.target || item.target || '');
            setNoteRa(data.nav.ra !== undefined && data.nav.ra !== null ? String(data.nav.ra) : '');
            setNoteDec(data.nav.dec !== undefined && data.nav.dec !== null ? String(data.nav.dec) : '');
            if (data.nav.backend) setSkyBackend(data.nav.backend);
            if (data.nav.max_mag !== undefined && data.nav.max_mag !== null) {
              setMaxMag(Number(data.nav.max_mag));
            }
            if (data.nav.camera_rotation_deg !== undefined && data.nav.camera_rotation_deg !== null) {
              setCameraRotationDeg(Number(data.nav.camera_rotation_deg));
            }
            if (data.nav.camera_flip_x !== undefined) {
              setCameraFlipX(Boolean(data.nav.camera_flip_x));
            }
            if (data.nav.show_crosshairs !== undefined) {
              setShowStarCrosshairs(Boolean(data.nav.show_crosshairs));
            }
            if (data.nav.favorite !== undefined) {
              setIsFavorite(Boolean(data.nav.favorite));
            }
            if (data.nav.hist_min !== undefined && data.nav.hist_max !== undefined) {
              const hmin = Number(data.nav.hist_min);
              const hmax = Number(data.nav.hist_max);
              setMinPoint(hmin);
              setMaxPoint(hmax);
              minPointRef.current = hmin;
              maxPointRef.current = hmax;
            }
            if (data.nav.anchor_x !== undefined && data.nav.anchor_y !== undefined) {
              setAnchorPos({ x: Number(data.nav.anchor_x), y: Number(data.nav.anchor_y) });
            }
            if (data.nav.orientation) {
              setNoteOrientation(data.nav.orientation);
            } else if (data.mosaic_meta?.drift_angle_deg !== undefined && data.mosaic_meta?.drift_angle_deg !== null) {
              setNoteOrientation(`Drift ${data.mosaic_meta.drift_angle_deg}° on sensor`);
            }
            setNoteText(data.nav.note || '');
          }
          if (!skyInitializedRef.current && data.nav.ra !== undefined && data.nav.dec !== undefined) {
            skyInitializedRef.current = true;
            queryMosaicCatalog({
              backend: data.nav.backend || 'astropy',
              ra: Number(data.nav.ra),
              dec: Number(data.nav.dec),
            });
          }
        } else if (!skyInitializedRef.current && item.target) {
          skyInitializedRef.current = true;
          queryMosaicCatalog({ query: item.target, backend: 'astropy' });
        }

        const nextStatus = data.mosaic_status || 'none';
        setMosaicProgress(data.mosaic_progress || null);

        setMosaicStatus((prevStatus) => {
          if ((prevStatus === 'processing' || prevStatus === 'queued') && nextStatus === 'ready') {
            setReloadCounter((c) => c + 1);
          }
          return nextStatus;
        });

        if (nextStatus === 'processing' || nextStatus === 'queued') {
          timer = setTimeout(pollStatus, 1000);
        }
      } catch {
        // ignore transient network errors
      }
    };

    pollStatus();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [item.id, item.target, mosaicStatus]);

  const handleToggleModalFavorite = async () => {
    const next = !isFavorite;
    setIsFavorite(next);
    try {
      const res = await fetch(`${API_BASE}/mosaic/note/${encodeURIComponent(item.id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ favorite: next }),
      });
      const data = await res.json();
      if (data.success && data.nav) {
        setNavInfo(data.nav);
        navInfoRef.current = data.nav;
      }
    } catch {
      // transient network error
    }
  };

  const handleSaveViewState = async () => {
    try {
      const payload: Record<string, any> = {
        hist_min: minPoint,
        hist_max: maxPoint,
        camera_rotation_deg: cameraRotationDeg,
        camera_flip_x: cameraFlipX,
        max_mag: maxMag,
        show_crosshairs: showStarCrosshairs,
        anchor_x: anchorPos ? anchorPos.x : undefined,
        anchor_y: anchorPos ? anchorPos.y : undefined,
        favorite: isFavorite,
      };
      const res = await fetch(`${API_BASE}/mosaic/note/${encodeURIComponent(item.id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.success && data.nav) {
        setNavInfo(data.nav);
        navInfoRef.current = data.nav;
        setViewSavedMsg('Saved!');
        setTimeout(() => setViewSavedMsg(''), 2000);
      }
    } catch {
      // transient network error
    }
  };

  // Auto-persist view settings (histogram window, camera rotation, mirror, max_mag, anchor)
  const isViewInitialLoadRef = useRef<boolean>(true);
  useEffect(() => {
    if (isViewInitialLoadRef.current) {
      if (mosaicData !== null && noteInitializedRef.current) {
        isViewInitialLoadRef.current = false;
      }
      return;
    }
    const timer = setTimeout(() => {
      fetch(`${API_BASE}/mosaic/note/${encodeURIComponent(item.id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hist_min: minPoint,
          hist_max: maxPoint,
          camera_rotation_deg: cameraRotationDeg,
          camera_flip_x: cameraFlipX,
          max_mag: maxMag,
          show_crosshairs: showStarCrosshairs,
          anchor_x: anchorPos ? anchorPos.x : undefined,
          anchor_y: anchorPos ? anchorPos.y : undefined,
          favorite: isFavorite,
        }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data?.success && data?.nav) {
            navInfoRef.current = data.nav;
          }
        })
        .catch(() => {});
    }, 700);
    return () => clearTimeout(timer);
  }, [minPoint, maxPoint, cameraRotationDeg, cameraFlipX, maxMag, showStarCrosshairs, anchorPos, isFavorite, mosaicData]);

  const handleSaveStackNote = async () => {
    setNoteSaving(true);
    setNoteSaveMsg('');
    try {
      const parsedRa = noteRa.trim() !== '' ? parseFloat(noteRa) : undefined;
      const parsedDec = noteDec.trim() !== '' ? parseFloat(noteDec) : undefined;
      const payload: Record<string, any> = {
        target: (searchText.trim() || noteStars.trim()),
        stars: noteStars.trim(),
        backend: skyBackend,
        max_mag: maxMag,
        camera_rotation_deg: cameraRotationDeg,
        camera_flip_x: cameraFlipX,
        hist_min: minPoint,
        hist_max: maxPoint,
        anchor_x: anchorPos ? anchorPos.x : undefined,
        anchor_y: anchorPos ? anchorPos.y : undefined,
        show_crosshairs: showStarCrosshairs,
        favorite: isFavorite,
        orientation: noteOrientation.trim(),
        note: noteText.trim(),
      };
      if (parsedRa !== undefined && !isNaN(parsedRa)) payload.ra = parsedRa;
      if (parsedDec !== undefined && !isNaN(parsedDec)) payload.dec = parsedDec;

      const res = await fetch(`${API_BASE}/mosaic/note/${encodeURIComponent(item.id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.success) {
        if (data.nav) setNavInfo(data.nav);
        setNoteSaveMsg('Saved!');
        setTimeout(() => setNoteSaveMsg(''), 2500);
      } else {
        setNoteSaveMsg(data.error || 'Save failed');
      }
    } catch (e: any) {
      setNoteSaveMsg(`Error: ${e.message}`);
    } finally {
      setNoteSaving(false);
    }
  };

  const handleTriggerRemosaic = async () => {
    setError(null);
    setMosaicStatus('processing');
    setMosaicProgress({ percent: 0, message: 'Starting remosaic...', stage: 'starting' });
    try {
      const res = await fetch(`${API_BASE}/mosaic/generate/${encodeURIComponent(item.id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          force: true,
          align_mode: alignMode,
          min_snr: minSnr,
          max_step_px: maxStepPx > 0 ? maxStepPx : null,
          static_filter: staticFilter,
          defect_radius: defectRadius,
          use_defect_mask: useDefectMask,
          feather_px: featherPx,
        }),
      });
      const data = await res.json();
      if (!data.success) {
        setError(data.error || 'Failed to start remosaic');
        setMosaicStatus('ready');
      }
    } catch (e: any) {
      setError(`Remosaic error: ${e.message}`);
      setMosaicStatus('ready');
    }
  };

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(`${API_BASE}/mosaic/linear16/${encodeURIComponent(item.id)}?t=${reloadCounter}`)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Failed to load 16-bit linear mosaic (HTTP ${res.status})`);
        }
        return res.arrayBuffer();
      })
      .then((buffer) => {
        if (cancelled) return;
        if (buffer.byteLength < 32) {
          throw new Error('Invalid 16-bit mosaic buffer');
        }
        const header = new Uint32Array(buffer, 0, 8);
        const magic = header[0];
        if (magic !== 0x36314d41) {
          throw new Error('Unrecognized 16-bit mosaic format header');
        }
        const width = header[1];
        const height = header[2];
        const channels = header[3];
        const autoMin = header[4];
        const autoMax = header[5];
        const minNz = header[6];
        const maxVal = header[7];

        const pixels = new Uint16Array(buffer, 32, width * height * channels);

        // Compute 256-bin histogram over valid (>0) 16-bit pixels (0..65535 -> bin = v >>> 8)
        const hist = new Array<number>(256).fill(0);
        if (channels === 1) {
          for (let i = 0; i < pixels.length; i++) {
            const v = pixels[i];
            if (v > 0) hist[v >>> 8]++;
          }
        } else {
          for (let i = 0; i < pixels.length; i += 3) {
            const r = pixels[i];
            const g = pixels[i + 1];
            const b = pixels[i + 2];
            const lum = (r + g + b) / 3;
            if (lum > 0) hist[lum >>> 8]++;
          }
        }

        // Detect top brightest star centroids on the 16-bit mosaic to anchor Star #1
        const getLum = (px: number, py: number): number => {
          const idx = (py * width + px) * channels;
          return channels === 1 ? pixels[idx] : (pixels[idx] + pixels[idx + 1] + pixels[idx + 2]) / 3;
        };
        const thresh = Math.max(autoMin + (autoMax - autoMin) * 0.3, autoMin + 512);
        const rawCandidates: { x: number; y: number; score: number; raw: number }[] = [];
        const border = 14;
        for (let y = border; y < height - border; y += 2) {
          for (let x = border; x < width - border; x += 2) {
            const v = getLum(x, y);
            if (v <= thresh) continue;
            // Require at least 4 neighbors in 5x5 above threshold so hot pixels are ignored
            let sum = 0;
            let cnt = 0;
            for (let dy = -2; dy <= 2; dy++) {
              for (let dx = -2; dx <= 2; dx++) {
                const nv = getLum(x + dx, y + dy);
                if (nv > autoMin) {
                  sum += nv - autoMin;
                  if (nv > thresh) cnt++;
                }
              }
            }
            if (cnt >= 4) {
              rawCandidates.push({ x, y, score: sum, raw: Math.round(v) });
            }
          }
        }
        rawCandidates.sort((a, b) => b.score - a.score);
        const peaks: { x: number; y: number; raw: number }[] = [];
        for (const cand of rawCandidates) {
          if (peaks.length >= 6) break;
          if (peaks.some((p) => Math.hypot(cand.x - p.x, cand.y - p.y) < 30)) continue;
          // Refine center-of-mass centroid in ±5px box
          let wx = 0;
          let wy = 0;
          let wsum = 0;
          let maxRaw = cand.raw;
          for (let dy = -5; dy <= 5; dy++) {
            for (let dx = -5; dx <= 5; dx++) {
              const px = cand.x + dx;
              const py = cand.y + dy;
              if (px < 0 || px >= width || py < 0 || py >= height) continue;
              const lv = getLum(px, py);
              if (lv > maxRaw) maxRaw = Math.round(lv);
              const w = Math.max(0, lv - autoMin);
              wx += px * w;
              wy += py * w;
              wsum += w;
            }
          }
          const cx = wsum > 0 ? +(wx / wsum).toFixed(2) : cand.x;
          const cy = wsum > 0 ? +(wy / wsum).toFixed(2) : cand.y;
          peaks.push({ x: cx, y: cy, raw: maxRaw });
        }
        setDetectedPeaks(peaks);
        setAnchorPeakIdx(0);

        // Restore saved anchor position if present, otherwise anchor on peak #0
        const savedAnchorX = navInfoRef.current.anchor_x ?? item.nav?.anchor_x;
        const savedAnchorY = navInfoRef.current.anchor_y ?? item.nav?.anchor_y;
        if (savedAnchorX !== undefined && savedAnchorY !== undefined && savedAnchorX >= 0 && savedAnchorY >= 0) {
          setAnchorPos({ x: savedAnchorX, y: savedAnchorY });
        } else if (peaks.length > 0) {
          setAnchorPos({ x: peaks[0].x, y: peaks[0].y });
        } else {
          setAnchorPos({ x: width / 2, y: height / 2 });
        }

        // Restore saved histogram window if present, otherwise autoMin/autoMax
        const savedMin = navInfoRef.current.hist_min ?? item.nav?.hist_min;
        const savedMax = navInfoRef.current.hist_max ?? item.nav?.hist_max;
        const startMin = (savedMin !== undefined && savedMin >= 0 && savedMin < 65535) ? savedMin : autoMin;
        const startMax = (savedMax !== undefined && savedMax > startMin && savedMax <= 65535) ? savedMax : autoMax;
        minPointRef.current = startMin;
        maxPointRef.current = startMax;
        setMinPoint(startMin);
        setMaxPoint(startMax);
        setMosaicData({
          width,
          height,
          channels,
          autoMin,
          autoMax,
          minNz,
          maxVal,
          pixels,
          histogram: hist,
        });
        setLoading(false);
      })
      .catch((err: any) => {
        if (!cancelled) {
          setError(err.message || 'Error loading 16-bit mosaic');
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [item.id, reloadCounter]);

  // Draw thin crosshairs (+) for the Top N catalog stars rotated around the #1 anchor star
  useEffect(() => {
    if (!mosaicData || !overlayCanvasRef.current) return;
    const cvs = overlayCanvasRef.current;
    const { width, height } = mosaicData;
    if (cvs.width !== width) cvs.width = width;
    if (cvs.height !== height) cvs.height = height;
    const ctx = cvs.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    if (!showStarCrosshairs || !anchorPos) return;

    const theta = (cameraRotationDeg * Math.PI) / 180.0;
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);

    const starsToDraw: MosaicCatalogStar[] =
      visibleCatalogStars.length > 0
        ? visibleCatalogStars
        : [
            {
              rank: 1,
              name: searchText || noteStars || 'Target #1',
              short_name: searchText || noteStars || 'Target #1',
              mag: 0,
              ra: 0,
              dec: 0,
              dx_px: 0,
              dy_px: 0,
              dist_px: 0,
              pa_deg: 0,
            },
          ];

    ctx.save();
    for (let i = 0; i < starsToDraw.length; i++) {
      const s = starsToDraw[i];
      const dx0 = cameraFlipX ? -s.dx_px : s.dx_px;
      const dy0 = s.dy_px;
      const sx = anchorPos.x + dx0 * cosT - dy0 * sinT;
      const sy = anchorPos.y + dx0 * sinT + dy0 * cosT;

      const isPrimary = i === 0;
      const gap = isPrimary ? 6 : 4.5;
      const arm = isPrimary ? 18 : 14;
      const color = isPrimary ? '#fbbf24' : '#38bdf8';

      // Subtle dashed line from #1 anchor to #2 star to help visualize rotation alignment
      if (i === 1) {
        ctx.save();
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.32)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(anchorPos.x, anchorPos.y);
        ctx.lineTo(sx, sy);
        ctx.stroke();
        ctx.restore();
      }

      // Dark halo behind thin cross for contrast on bright stars
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.72)';
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(sx - arm, sy);
      ctx.lineTo(sx - gap, sy);
      ctx.moveTo(sx + gap, sy);
      ctx.lineTo(sx + arm, sy);
      ctx.moveTo(sx, sy - arm);
      ctx.lineTo(sx, sy - gap);
      ctx.moveTo(sx, sy + gap);
      ctx.lineTo(sx, sy + arm);
      ctx.stroke();

      // Crisp 1.2px thin cross (+)
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.moveTo(sx - arm, sy);
      ctx.lineTo(sx - gap, sy);
      ctx.moveTo(sx + gap, sy);
      ctx.lineTo(sx + arm, sy);
      ctx.moveTo(sx, sy - arm);
      ctx.lineTo(sx, sy - gap);
      ctx.moveTo(sx, sy + gap);
      ctx.lineTo(sx, sy + arm);
      ctx.stroke();

      // Label next to crosshair
      const magStr = s.mag ? ` (${s.mag.toFixed(1)}m)` : '';
      const label = `#${s.rank} ${s.short_name}${magStr}`;
      ctx.font = '600 12px monospace';
      ctx.fillStyle = 'rgba(0, 0, 0, 0.78)';
      ctx.fillText(label, sx + 8, sy - 6);
      ctx.fillStyle = color;
      ctx.fillText(label, sx + 7, sy - 7);
    }
    ctx.restore();
  }, [
    mosaicData,
    anchorPos,
    visibleCatalogStars,
    cameraRotationDeg,
    cameraFlipX,
    showStarCrosshairs,
    maxMag,
    searchText,
    noteStars,
  ]);

  // Render 16-bit pixels through [minPoint, maxPoint] LUT onto the display canvas
  useEffect(() => {
    if (!mosaicData || !imageCanvasRef.current) return;
    const cvs = imageCanvasRef.current;
    const { width, height, channels, pixels } = mosaicData;
    if (cvs.width !== width) cvs.width = width;
    if (cvs.height !== height) cvs.height = height;
    const ctx = cvs.getContext('2d');
    if (!ctx) return;

    const lut = new Uint8Array(65536);
    const minP = Math.max(0, Math.min(65534, minPoint));
    const maxP = Math.max(minP + 1, Math.min(65535, maxPoint));
    const invSpan = 255.0 / (maxP - minP);

    for (let v = 1; v < 65536; v++) {
      if (v <= minP) lut[v] = 0;
      else if (v >= maxP) lut[v] = 255;
      else lut[v] = ((v - minP) * invSpan + 0.5) | 0;
    }
    lut[0] = 0;

    const imgData = ctx.createImageData(width, height);
    if (channels === 1) {
      const out32 = new Uint32Array(imgData.data.buffer);
      const len = width * height;
      for (let i = 0; i < len; i++) {
        const g = lut[pixels[i]];
        out32[i] = 0xff000000 | (g << 16) | (g << 8) | g;
      }
    } else {
      const out8 = imgData.data;
      const len = width * height;
      for (let i = 0, p = 0, q = 0; i < len; i++, p += 3, q += 4) {
        out8[q] = lut[pixels[p]];
        out8[q + 1] = lut[pixels[p + 1]];
        out8[q + 2] = lut[pixels[p + 2]];
        out8[q + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [mosaicData, minPoint, maxPoint]);

  // Draw bottom-right 256-bin histogram & min/max cursors (matching live stream histogram style)
  useEffect(() => {
    if (!mosaicData || !histCanvasRef.current) return;
    const cvs = histCanvasRef.current;
    const w = 512;
    const h = 54;
    if (cvs.width !== w) cvs.width = w;
    if (cvs.height !== h) cvs.height = h;
    const ctx = cvs.getContext('2d');
    if (!ctx) return;

    const yAxis = 44;
    ctx.clearRect(0, 0, w, h);

    const data = mosaicData.histogram;
    const n = data.length;
    if (n === 0) return;

    const logCounts = data.map((c) => (c > 0 ? Math.log10(c + 1) : 0));
    const maxLog = Math.max(...logCounts, 1.0);
    const barW = w / n;

    ctx.fillStyle = '#58a6ff';
    for (let i = 0; i < n; i++) {
      if (logCounts[i] <= 0) continue;
      const barH = Math.max(1, Math.round((logCounts[i] / maxLog) * (yAxis - 2)));
      const x = i * barW;
      ctx.fillRect(x, yAxis - barH, barW, barH);
    }

    // Horizontal baseline x-axis
    ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.fillRect(0, yAxis, w, 1);

    // Ticks below x-axis
    for (let i = 0; i < n; i++) {
      const x = Math.round(i * barW);
      if (i % 64 === 0) {
        ctx.fillStyle = '#e6edf3';
        ctx.fillRect(x - 1, yAxis + 1, 2, 7);
      } else if (i % 16 === 0) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.fillRect(x, yAxis + 1, 1, 4);
      }
    }

    // Draw Min Point handle (Sky Blue cursor & pointer)
    const bx = Math.max(1, Math.min(w - 2, Math.round((minPoint / 65535) * w)));
    ctx.fillStyle = '#38bdf8';
    ctx.fillRect(bx - 1, 0, 2, yAxis);
    ctx.beginPath();
    ctx.moveTo(bx - 4, yAxis + 9);
    ctx.lineTo(bx + 4, yAxis + 9);
    ctx.lineTo(bx, yAxis + 1);
    ctx.fill();

    // Draw Max Point handle (Amber cursor & pointer)
    const wx = Math.max(1, Math.min(w - 2, Math.round((maxPoint / 65535) * w)));
    ctx.fillStyle = '#fbbf24';
    ctx.fillRect(wx - 1, 0, 2, yAxis);
    ctx.beginPath();
    ctx.moveTo(wx - 4, yAxis + 9);
    ctx.lineTo(wx + 4, yAxis + 9);
    ctx.lineTo(wx, yAxis + 1);
    ctx.fill();
  }, [mosaicData, minPoint, maxPoint]);

  const handleHistPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const val = Math.max(0, Math.min(65535, Math.round((x / rect.width) * 65535)));

    const distMin = Math.abs(val - minPointRef.current);
    const distMax = Math.abs(val - maxPointRef.current);

    dragPointerRef.current = { lastX: e.clientX };

    if (distMin <= distMax) {
      const nextMin = Math.min(val, maxPointRef.current - 1);
      minPointRef.current = nextMin;
      setMinPoint(nextMin);
      setActiveHandle('min');
    } else {
      const nextMax = Math.max(val, minPointRef.current + 1);
      maxPointRef.current = nextMax;
      setMaxPoint(nextMax);
      setActiveHandle('max');
    }
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handleHistPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.stopPropagation();
    if (!activeHandle) return;
    const rect = e.currentTarget.getBoundingClientRect();

    if (e.shiftKey && dragPointerRef.current) {
      // Fine-grain adjustment when holding Shift
      const dx = e.clientX - dragPointerRef.current.lastX;
      dragPointerRef.current.lastX = e.clientX;
      const step = Math.round((dx / rect.width) * 4096);
      if (activeHandle === 'min') {
        const nextMin = Math.max(0, Math.min(maxPointRef.current - 1, minPointRef.current + step));
        minPointRef.current = nextMin;
        setMinPoint(nextMin);
      } else {
        const nextMax = Math.min(65535, Math.max(minPointRef.current + 1, maxPointRef.current + step));
        maxPointRef.current = nextMax;
        setMaxPoint(nextMax);
      }
      return;
    }

    dragPointerRef.current = { lastX: e.clientX };
    const x = e.clientX - rect.left;
    const val = Math.max(0, Math.min(65535, Math.round((x / rect.width) * 65535)));

    if (activeHandle === 'min') {
      const nextMin = Math.min(val, maxPointRef.current - 1);
      minPointRef.current = nextMin;
      setMinPoint(nextMin);
    } else if (activeHandle === 'max') {
      const nextMax = Math.max(val, minPointRef.current + 1);
      maxPointRef.current = nextMax;
      setMaxPoint(nextMax);
    }
  };

  const handleHistPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.stopPropagation();
    setActiveHandle(null);
    dragPointerRef.current = null;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  };

  const handleApplyAutoStretch = () => {
    if (!mosaicData) return;
    minPointRef.current = mosaicData.autoMin;
    maxPointRef.current = mosaicData.autoMax;
    setMinPoint(mosaicData.autoMin);
    setMaxPoint(mosaicData.autoMax);
  };

  const handleResetFullRange = () => {
    minPointRef.current = 0;
    maxPointRef.current = 65535;
    setMinPoint(0);
    setMaxPoint(65535);
  };

  const handleWheelCanvas = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
    setZoom((prev) => {
      const next = Math.max(1, Math.min(12, +(prev * factor).toFixed(2)));
      if (next === 1) setPan({ x: 0, y: 0 });
      return next;
    });
  };

  const handleCanvasPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest('.stream-histogram, .mosaic-viewer-header, .mosaic-side-panel')) return;
    if (e.altKey && mosaicData && imageCanvasRef.current) {
      const rect = imageCanvasRef.current.getBoundingClientRect();
      const px = Math.round(((e.clientX - rect.left) / rect.width) * mosaicData.width);
      const py = Math.round(((e.clientY - rect.top) / rect.height) * mosaicData.height);
      if (px >= 0 && px < mosaicData.width && py >= 0 && py < mosaicData.height) {
        setAnchorPos({ x: px, y: py });
        return;
      }
    }
    panDragRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y };
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {}
  };

  const handleCycleAnchorPeak = () => {
    if (detectedPeaks.length === 0) return;
    const nextIdx = (anchorPeakIdx + 1) % detectedPeaks.length;
    setAnchorPeakIdx(nextIdx);
    setAnchorPos({ x: detectedPeaks[nextIdx].x, y: detectedPeaks[nextIdx].y });
  };

  const handleCopyTopStarsToNote = () => {
    if (visibleCatalogStars.length === 0) return;
    const summary = visibleCatalogStars
      .map((s) => `${s.short_name} (${s.mag.toFixed(1)}m)`)
      .join(' + ');
    setNoteStars(summary);
  };

  const handleCanvasPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (mosaicData && imageCanvasRef.current) {
      const rect = imageCanvasRef.current.getBoundingClientRect();
      const px = Math.floor(((e.clientX - rect.left) / rect.width) * mosaicData.width);
      const py = Math.floor(((e.clientY - rect.top) / rect.height) * mosaicData.height);
      if (px >= 0 && px < mosaicData.width && py >= 0 && py < mosaicData.height) {
        const idx = (py * mosaicData.width + px) * mosaicData.channels;
        const val =
          mosaicData.channels === 1
            ? mosaicData.pixels[idx]
            : Math.round(
                (mosaicData.pixels[idx] + mosaicData.pixels[idx + 1] + mosaicData.pixels[idx + 2]) / 3
              );
        setHoverInfo({ x: px, y: py, val });
      } else {
        setHoverInfo(null);
      }
    }

    if (!panDragRef.current) return;
    const dx = e.clientX - panDragRef.current.startX;
    const dy = e.clientY - panDragRef.current.startY;
    setPan({ x: panDragRef.current.panX + dx, y: panDragRef.current.panY + dy });
  };

  const handleCanvasPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    panDragRef.current = null;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {}
  };

  const tifDownloadUrl = `${API_BASE}/captures/${item.id}/mosaic_v1/mosaic_linear_16bit.tif`;
  const isStitching = mosaicStatus === 'processing' || mosaicStatus === 'queued';
  const displayTarget = noteStars || navInfo.target || item.target || '';

  return (
    <div className="mosaic-viewer-backdrop" onClick={onClose}>
      <div
        className="mosaic-viewer-Stage"
        onClick={(e) => e.stopPropagation()}
        onWheel={handleWheelCanvas}
        onPointerDown={handleCanvasPointerDown}
        onPointerMove={handleCanvasPointerMove}
        onPointerUp={handleCanvasPointerUp}
        onPointerCancel={handleCanvasPointerUp}
      >
        <div className="mosaic-viewer-header">
          <div className="mosaic-viewer-title">
            <Sparkles size={16} color="#58a6ff" />
            <span>{displayTarget ? `${displayTarget} • ` : ''}{item.datetime || item.id}</span>
            {mosaicData && (
              <span className="mosaic-viewer-meta">
                {mosaicData.width}×{mosaicData.height} • 16-bit Linear ({mosaicMeta?.aligned_frames ?? item.frames ?? 1}
                {mosaicMeta?.valid_files ? `/${mosaicMeta.valid_files}` : ''} frames)
              </span>
            )}
            {mosaicMeta && (
              <span className="mosaic-diag-pill" title="Alignment diagnostic summary">
                {mosaicMeta.constellation_stars && mosaicMeta.constellation_stars > 1
                  ? `${mosaicMeta.constellation_stars}-Star Constellation`
                  : '1-Star Track'}
                {mosaicMeta.identity_swaps_prevented ? ` • ${mosaicMeta.identity_swaps_prevented} swaps resolved` : ''}
                {mosaicMeta.static_peaks_rejected ? ` • ${mosaicMeta.static_peaks_rejected} warm px rejected` : ''}
              </span>
            )}
          </div>
          <div className="mosaic-viewer-actions">
            <button
              type="button"
              className={`mosaic-viewer-btn fav ${isFavorite ? 'active' : ''}`}
              onClick={handleToggleModalFavorite}
              title={isFavorite ? 'Favorited stack (click to remove from favorites)' : 'Favorite this stack'}
            >
              <Star
                size={14}
                fill={isFavorite ? '#fbbf24' : 'none'}
                color={isFavorite ? '#fbbf24' : 'currentColor'}
              />
              <span>{isFavorite ? 'Favorite' : 'Add Favorite'}</span>
            </button>
            <button
              type="button"
              className={`mosaic-viewer-btn ${showInfoPanel ? 'active' : ''}`}
              onClick={() => setShowInfoPanel((v) => !v)}
              title="Toggle Stack Note, Coordinates, Orientation & Diagnostics"
            >
              <Compass size={14} /> Stack Note & Info
            </button>
            <button
              type="button"
              className={`mosaic-viewer-btn ${showStitchControls ? 'active' : ''}`}
              onClick={() => setShowStitchControls((v) => !v)}
              title="Configure alignment & mosaic parameters"
            >
              <Sliders size={14} /> Stitch Controls
            </button>
            <button
              type="button"
              className="mosaic-viewer-btn remosaic"
              onClick={handleTriggerRemosaic}
              disabled={isStitching}
              title="Re-run star alignment and re-stitch this mosaic with current settings"
            >
              <RefreshCw size={14} className={isStitching ? 'spin' : ''} />
              {isStitching
                ? mosaicProgress?.percent
                  ? `Remosaicing ${Math.round(mosaicProgress.percent)}%`
                  : 'Remosaicing...'
                : 'Remosaic'}
            </button>
            {zoom > 1 && (
              <button
                type="button"
                className="mosaic-viewer-btn"
                onClick={() => {
                  setZoom(1);
                  setPan({ x: 0, y: 0 });
                }}
              >
                Reset Zoom ({zoom.toFixed(1)}×)
              </button>
            )}
            <a
              href={tifDownloadUrl}
              download={`${item.id}_mosaic_linear_16bit.tif`}
              className="mosaic-viewer-btn"
              title="Download raw 16-bit linear TIFF"
            >
              Download 16-bit TIFF
            </a>
            {!isStandaloneTab && (
              <button
                type="button"
                className="mosaic-viewer-btn close"
                onClick={onClose}
                title="Close viewer (Esc)"
              >
                <X size={16} /> Close
              </button>
            )}
          </div>
        </div>

        {isStitching && (
          <div className="mosaic-live-progress-banner">
            <div className="mosaic-live-progress-text">
              <RefreshCw size={13} className="spin" />
              <span>{mosaicProgress?.message || 'Processing mosaic alignment...'}</span>
              <strong>{Math.round(mosaicProgress?.percent || 0)}%</strong>
            </div>
            <div className="mosaic-live-progress-track">
              <div
                className="mosaic-live-progress-fill"
                style={{ width: `${Math.min(100, Math.max(3, mosaicProgress?.percent || 3))}%` }}
              />
            </div>
          </div>
        )}

        {(showInfoPanel || showStitchControls) && (
          <div
            className="mosaic-side-panel"
            onPointerDown={(e) => e.stopPropagation()}
            onPointerMove={(e) => e.stopPropagation()}
            onPointerUp={(e) => e.stopPropagation()}
            onWheel={(e) => e.stopPropagation()}
          >
            {showInfoPanel && (
              <div className="mosaic-panel-card">
                <div className="mosaic-panel-title">
                  <span>✨ Stack Note & Navigation</span>
                  {navInfo.altitude !== undefined && navInfo.azimuth !== undefined && (
                    <span className="mosaic-panel-sub">
                      Alt {Number(navInfo.altitude).toFixed(1)}° · Az {Number(navInfo.azimuth).toFixed(1)}°
                    </span>
                  )}
                </div>

                {/* Target Search with Autocomplete + DB Source (like Live View Navigation) */}
                <div className="mosaic-nav-search-box" style={{ position: 'relative' }}>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <input
                      type="text"
                      placeholder="Search star/target (e.g. Menkalinan, Mizar, Vega)..."
                      value={searchText}
                      onFocus={() => setSearchIsOpen(true)}
                      onChange={(e) => {
                        setSearchText(e.target.value);
                        setSearchIsOpen(true);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && searchText.trim()) {
                          setSearchIsOpen(false);
                          queryMosaicCatalog({ query: searchText.trim() });
                        } else if (e.key === 'Escape') {
                          setSearchIsOpen(false);
                        }
                      }}
                      style={{ flex: 1, fontSize: '11px' }}
                    />
                    <button
                      type="button"
                      className="mosaic-panel-btn primary"
                      disabled={skyLoading || !searchText.trim()}
                      onClick={() => {
                        setSearchIsOpen(false);
                        queryMosaicCatalog({ query: searchText.trim() });
                      }}
                      style={{ padding: '4px 8px' }}
                    >
                      {skyLoading ? '…' : 'Go'}
                    </button>
                  </div>

                  {searchIsOpen && (
                    <>
                      <div
                        style={{ position: 'fixed', inset: 0, zIndex: 998 }}
                        onClick={() => setSearchIsOpen(false)}
                      />
                      <div className="mosaic-autocomplete-menu">
                        {(() => {
                          const q = searchText.trim().toLowerCase();
                          const matchedTargets = targets
                            .filter(
                              (t) =>
                                !q ||
                                t.name.toLowerCase().includes(q) ||
                                (t.alt && t.alt.toLowerCase().includes(q))
                            )
                            .slice(0, 30);

                          if (matchedTargets.length === 0) {
                            return (
                              <div style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: '11px' }}>
                                No local match. Click &apos;Go&apos; to resolve via SIMBAD.
                              </div>
                            );
                          }

                          return matchedTargets.map((t) => {
                            const label = t.alt ? `${t.name} (${t.alt})` : t.name;
                            const isPlanet = t.type === 'planet';
                            return (
                              <div
                                key={`mt-${t.name}`}
                                className="mosaic-autocomplete-item"
                                onClick={() => {
                                  setSearchIsOpen(false);
                                  setSearchText(label);
                                  setNoteStars(label);
                                  setNoteRa(String(t.ra));
                                  setNoteDec(String(t.dec));
                                  queryMosaicCatalog({
                                    query: t.name,
                                    ra: t.ra,
                                    dec: t.dec,
                                  });
                                }}
                              >
                                <span style={{ fontWeight: 600, color: '#e6edf3' }}>
                                  {isPlanet ? '🪐 ' : ''}
                                  {t.name}
                                </span>
                                <span style={{ fontSize: '10px', color: '#8b949e' }}>
                                  {t.alt ? t.alt : t.type}
                                </span>
                              </div>
                            );
                          });
                        })()}
                      </div>
                    </>
                  )}
                </div>

                {skyMsg && (
                  <div
                    style={{
                      fontSize: '10px',
                      color:
                        skyMsg.startsWith('Error') || skyMsg.startsWith('Could') || skyMsg.startsWith('Catalog')
                          ? '#f85149'
                          : '#58a6ff',
                    }}
                  >
                    {skyMsg}
                  </div>
                )}

                <div className="mosaic-form-grid">
                  <label className="mosaic-field">
                    <span>Star Catalog (DB Source)</span>
                    <select
                      value={skyBackend}
                      onChange={(e) => {
                        const nextBackend = e.target.value;
                        setSkyBackend(nextBackend);
                        queryMosaicCatalog({ backend: nextBackend });
                      }}
                    >
                      {skyBackends.map((b) => (
                        <option key={b} value={b}>
                          {b === 'astropy' ? 'astropy (Gaia DR3 + HIP)' : 'skyfield (Hipparcos)'}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="mosaic-field">
                    <span style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span>Max Mag (≤ {maxMag.toFixed(1)}m)</span>
                      <button
                        type="button"
                        className="mosaic-link-btn"
                        onClick={() => setShowStarCrosshairs((v) => !v)}
                        title="Toggle star crosshair overlay on/off"
                      >
                        {showStarCrosshairs ? 'Overlay: ON' : 'Overlay: OFF'}
                      </button>
                    </span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <input
                        type="range"
                        min={3.0}
                        max={12.0}
                        step={0.1}
                        value={maxMag}
                        onChange={(e) => {
                          setShowStarCrosshairs(true);
                          setMaxMag(parseFloat(e.target.value));
                        }}
                        style={{ flex: 1, accentColor: '#58a6ff' }}
                      />
                      <input
                        type="number"
                        min={1.0}
                        max={13.0}
                        step={0.1}
                        value={maxMag}
                        onChange={(e) => {
                          const v = parseFloat(e.target.value);
                          if (!isNaN(v)) {
                            setShowStarCrosshairs(true);
                            setMaxMag(Math.max(1.0, Math.min(13.0, v)));
                          }
                        }}
                        style={{ width: '48px', textAlign: 'right' }}
                      />
                    </div>
                  </label>
                </div>

                {/* Brightest Stars in Scene <= Max Mag */}
                <div className="mosaic-top-stars-box">
                  <div className="mosaic-top-stars-header">
                    <span>
                      Stars ≤ {maxMag.toFixed(1)}m in Scene ({visibleCatalogStars.length}/{totalCatalogStars})
                    </span>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      {detectedPeaks.length > 1 && (
                        <button
                          type="button"
                          className="mosaic-link-btn"
                          onClick={handleCycleAnchorPeak}
                          title="Switch which bright star on the mosaic canvas is anchored as Star #1 (or Alt+Click on the canvas)"
                        >
                          Anchor Peak #{anchorPeakIdx + 1}/{detectedPeaks.length}
                        </button>
                      )}
                      {visibleCatalogStars.length > 0 && (
                        <button
                          type="button"
                          className="mosaic-link-btn"
                          onClick={handleCopyTopStarsToNote}
                          title="Copy visible catalog star names into Stars / Target Name field"
                        >
                          Use in Note
                        </button>
                      )}
                    </div>
                  </div>
                  {visibleCatalogStars.length === 0 ? (
                    <div className="mosaic-top-stars-empty">
                      {skyLoading ? 'Querying star catalog...' : 'No catalog stars loaded yet.'}
                    </div>
                  ) : (
                    <div className="mosaic-top-stars-list" style={{ maxHeight: '110px', overflowY: 'auto' }}>
                      {visibleCatalogStars.map((s) => (
                        <div key={`cs-${s.rank}`} className="mosaic-top-star-row">
                          <span
                            className="mosaic-star-rank"
                            style={{ color: s.rank === 1 ? '#fbbf24' : '#38bdf8' }}
                          >
                            +{s.rank}
                          </span>
                          <span className="mosaic-star-name" title={s.name}>
                            {s.name}
                          </span>
                          <span className="mosaic-star-mag">{s.mag.toFixed(2)}m</span>
                          <span className="mosaic-star-dist">
                            {s.rank === 1 ? 'anchor' : `${Math.round(s.dist_px)}px`}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Camera Module Rotation Control */}
                <div className="mosaic-rotation-box">
                  <div className="mosaic-rotation-header">
                    <span>Camera Module Rotation (°)</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <input
                        type="number"
                        min={-180}
                        max={180}
                        step={0.5}
                        value={cameraRotationDeg}
                        onChange={(e) => {
                          const v = parseFloat(e.target.value);
                          if (!isNaN(v)) setCameraRotationDeg(Math.max(-180, Math.min(180, v)));
                        }}
                        className="mosaic-rot-num"
                      />
                      <button
                        type="button"
                        className={`mosaic-mini-chip ${cameraFlipX ? 'active' : ''}`}
                        onClick={() => setCameraFlipX((f) => !f)}
                        title="Mirror East/West parity if optical train flips X"
                      >
                        Mirror X
                      </button>
                    </div>
                  </div>
                  <input
                    type="range"
                    min={-180}
                    max={180}
                    step={0.5}
                    value={cameraRotationDeg}
                    onChange={(e) => setCameraRotationDeg(parseFloat(e.target.value))}
                    className="mosaic-rot-slider"
                  />
                  <div className="mosaic-rot-presets">
                    {([-180, -90, 0, 90, 180] as const).map((ang) => (
                      <button
                        key={ang}
                        type="button"
                        className={`mosaic-mini-chip ${cameraRotationDeg === ang ? 'active' : ''}`}
                        onClick={() => setCameraRotationDeg(ang)}
                      >
                        {ang > 0 ? `+${ang}°` : `${ang}°`}
                      </button>
                    ))}
                    {mosaicMeta?.drift_angle_deg !== undefined && mosaicMeta?.drift_angle_deg !== null && (
                      <button
                        type="button"
                        className="mosaic-mini-chip"
                        onClick={() => {
                          const dAng = Number(mosaicMeta.drift_angle_deg);
                          setCameraRotationDeg(dAng);
                          setNoteOrientation(
                            `Camera rot ${dAng}° (Drift ${mosaicMeta.drift_angle_deg}°${
                              mosaicMeta.pair_angle_deg !== null && mosaicMeta.pair_angle_deg !== undefined
                                ? `, Pair ${mosaicMeta.pair_angle_deg}°`
                                : ''
                            })`
                          );
                        }}
                        title="Set rotation from measured star drift angle"
                      >
                        Drift ({mosaicMeta.drift_angle_deg}°)
                      </button>
                    )}
                  </div>
                </div>

                <div className="mosaic-form-grid">
                  <label className="mosaic-field full">
                    <span>Stars / Target Note</span>
                    <input
                      type="text"
                      placeholder="e.g. Menkalinan (β Aur), Companion stars..."
                      value={noteStars}
                      onChange={(e) => setNoteStars(e.target.value)}
                    />
                  </label>
                  <label className="mosaic-field">
                    <span>RA (°)</span>
                    <input
                      type="number"
                      step="any"
                      placeholder="e.g. 89.882"
                      value={noteRa}
                      onChange={(e) => setNoteRa(e.target.value)}
                      onBlur={() => {
                        const r = parseFloat(noteRa);
                        const d = parseFloat(noteDec);
                        if (!isNaN(r) && !isNaN(d)) {
                          queryMosaicCatalog({ ra: r, dec: d });
                        }
                      }}
                    />
                  </label>
                  <label className="mosaic-field">
                    <span>Dec (°)</span>
                    <input
                      type="number"
                      step="any"
                      placeholder="e.g. 44.947"
                      value={noteDec}
                      onChange={(e) => setNoteDec(e.target.value)}
                      onBlur={() => {
                        const r = parseFloat(noteRa);
                        const d = parseFloat(noteDec);
                        if (!isNaN(r) && !isNaN(d)) {
                          queryMosaicCatalog({ ra: r, dec: d });
                        }
                      }}
                    />
                  </label>
                  <label className="mosaic-field full">
                    <span style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Picture Orientation Note</span>
                      <span
                        style={{ color: '#58a6ff', cursor: 'pointer' }}
                        onClick={() =>
                          setNoteOrientation(
                            `Cam rot ${cameraRotationDeg.toFixed(1)}°${cameraFlipX ? ' (MirrorX)' : ''}${
                              mosaicMeta?.drift_angle_deg !== undefined && mosaicMeta?.drift_angle_deg !== null
                                ? ` • Drift ${mosaicMeta.drift_angle_deg}°`
                                : ''
                            }`
                          )
                        }
                        title="Click to insert current camera rotation & drift angle"
                      >
                        Insert Rot ({cameraRotationDeg.toFixed(1)}°)
                      </span>
                    </span>
                    <input
                      type="text"
                      placeholder="e.g. Camera rotated ~31° CCW; West drift is 121°"
                      value={noteOrientation}
                      onChange={(e) => setNoteOrientation(e.target.value)}
                    />
                  </label>
                  <label className="mosaic-field full">
                    <span>Stack Note</span>
                    <textarea
                      rows={2}
                      placeholder="Quick notes for this stack (seeing, clouds, star pair, camera angle...)"
                      value={noteText}
                      onChange={(e) => setNoteText(e.target.value)}
                    />
                  </label>
                </div>

                <div className="mosaic-panel-footer">
                  <span className="mosaic-save-msg">{noteSaveMsg}</span>
                  <button
                    type="button"
                    className="mosaic-panel-btn primary"
                    onClick={handleSaveStackNote}
                    disabled={noteSaving}
                  >
                    <Save size={12} /> {noteSaving ? 'Saving...' : 'Save Note & View'}
                  </button>
                </div>

                {mosaicMeta && (
                  <div className="mosaic-diag-box">
                    <div className="mosaic-diag-header">Alignment Diagnostics</div>
                    <div className="mosaic-diag-row">
                      <span>Aligned Frames:</span>
                      <strong>
                        {mosaicMeta.aligned_frames} / {mosaicMeta.valid_files}{' '}
                        ({mosaicMeta.valid_files ? ((mosaicMeta.aligned_frames! / mosaicMeta.valid_files) * 100).toFixed(1) : 100}%)
                      </strong>
                    </div>
                    <div className="mosaic-diag-row">
                      <span>Constellation Stars:</span>
                      <strong>
                        {mosaicMeta.constellation_stars || 1} ({mosaicMeta.pair_matches || 0} multi-star,{' '}
                        {mosaicMeta.single_star_matches || 0} single)
                      </strong>
                    </div>
                    <div className="mosaic-diag-row">
                      <span>Rank Swaps Prevented:</span>
                      <strong style={{ color: (mosaicMeta.identity_swaps_prevented || 0) > 0 ? '#3fb950' : undefined }}>
                        {mosaicMeta.identity_swaps_prevented ?? 0}
                      </strong>
                    </div>
                    <div className="mosaic-diag-row">
                      <span>Warm Peaks / Gate:</span>
                      <strong>
                        {mosaicMeta.static_peaks_rejected ?? 0} static purged • {mosaicMeta.skipped_step_gate ?? 0} gated
                      </strong>
                    </div>
                    <div className="mosaic-diag-row">
                      <span>Sensor Drift & Angle:</span>
                      <strong>
                        {mosaicMeta.drift_span_px ? `${mosaicMeta.drift_span_px[0]}×${mosaicMeta.drift_span_px[1]} px` : '—'}
                        {mosaicMeta.drift_angle_deg !== null && mosaicMeta.drift_angle_deg !== undefined
                          ? ` • ${mosaicMeta.drift_angle_deg}°`
                          : ''}
                        {mosaicMeta.pair_angle_deg !== null && mosaicMeta.pair_angle_deg !== undefined
                          ? ` (Pair ${mosaicMeta.pair_angle_deg}°)`
                          : ''}
                      </strong>
                    </div>
                  </div>
                )}
              </div>
            )}

            {showStitchControls && (
              <div className="mosaic-panel-card">
                <div className="mosaic-panel-title">
                  <span>⚙️ Mosaic Stitch Controls</span>
                  <span className="mosaic-panel-sub">
                    {mosaicMeta?.elapsed_sec ? `${mosaicMeta.elapsed_sec.toFixed(1)}s` : ''}
                  </span>
                </div>

                <div className="mosaic-form-grid">
                  <label className="mosaic-field full">
                    <span>Tracking Algorithm</span>
                    <select
                      value={alignMode}
                      onChange={(e) => setAlignMode(e.target.value as 'constellation' | 'single')}
                    >
                      <option value="constellation">Constellation + Trajectory (Multi-Star)</option>
                      <option value="single">Single Brightest Star (Legacy)</option>
                    </select>
                  </label>
                  <label className="mosaic-field">
                    <span>Min Star SNR ({minSnr.toFixed(1)})</span>
                    <input
                      type="number"
                      min={1.5}
                      max={10.0}
                      step={0.5}
                      value={minSnr}
                      onChange={(e) => setMinSnr(Math.max(1.0, parseFloat(e.target.value) || 3.0))}
                    />
                  </label>
                  <label className="mosaic-field">
                    <span>Max Step Gate (0=Auto)</span>
                    <input
                      type="number"
                      min={0}
                      max={500}
                      step={5}
                      value={maxStepPx}
                      onChange={(e) => setMaxStepPx(Math.max(0, parseFloat(e.target.value) || 0))}
                    />
                  </label>
                  <label className="mosaic-field">
                    <span>Defect Radius (px)</span>
                    <input
                      type="number"
                      min={0}
                      max={4}
                      step={1}
                      value={defectRadius}
                      onChange={(e) => setDefectRadius(Math.max(0, Math.min(4, parseInt(e.target.value) || 0)))}
                    />
                  </label>
                  <label className="mosaic-field">
                    <span>Seam Feather (px)</span>
                    <input
                      type="number"
                      min={0}
                      max={60}
                      step={2}
                      value={featherPx}
                      onChange={(e) => setFeatherPx(Math.max(0, parseInt(e.target.value) || 0))}
                    />
                  </label>
                </div>

                <div className="mosaic-toggle-row">
                  <button
                    type="button"
                    className={`mosaic-chip-btn ${staticFilter ? 'active' : ''}`}
                    onClick={() => setStaticFilter((v) => !v)}
                    title="Automatically reject stationary warm pixels that stay at fixed sensor coordinates across the sequence"
                  >
                    Static Warm-Px Filter: {staticFilter ? 'ON' : 'OFF'}
                  </button>
                  <button
                    type="button"
                    className={`mosaic-chip-btn ${useDefectMask ? 'active' : ''}`}
                    onClick={() => setUseDefectMask((v) => !v)}
                    title="Apply dark baseline defect mask and neighbor dilation"
                  >
                    Dark Defect Mask: {useDefectMask ? 'ON' : 'OFF'}
                  </button>
                </div>

                <div className="mosaic-panel-footer">
                  <button
                    type="button"
                    className="mosaic-panel-btn primary full"
                    onClick={handleTriggerRemosaic}
                    disabled={isStitching}
                  >
                    <RefreshCw size={12} className={isStitching ? 'spin' : ''} />
                    {isStitching ? 'Remosaicing...' : 'Remosaic with Settings'}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="mosaic-viewer-canvas-wrap" title="Drag to pan • Scroll to zoom • Alt+Click to move Star #1 crosshair anchor">
          {loading && <div className="mosaic-viewer-status">Loading 16-bit Linear Mosaic...</div>}
          {error && <div className="mosaic-viewer-status error">{error}</div>}
          <canvas
            ref={imageCanvasRef}
            className="mosaic-viewer-canvas"
            style={{
              display: mosaicData && !loading ? 'block' : 'none',
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            }}
          />
          <canvas
            ref={overlayCanvasRef}
            className="mosaic-viewer-canvas mosaic-overlay-canvas"
            style={{
              display: mosaicData && !loading && showStarCrosshairs ? 'block' : 'none',
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            }}
          />
        </div>

        {mosaicData && (
          <div className="stream-overlay" style={{ top: '58px', right: '16px' }}>
            <span>
              Range: [{minPoint} – {maxPoint}] / 65535
            </span>
            <span>
              Valid Min/Max: {mosaicData.minNz} – {mosaicData.maxVal}
            </span>
            {hoverInfo && (
              <span style={{ color: '#58a6ff', fontWeight: 600 }}>
                Cursor ({hoverInfo.x}, {hoverInfo.y}): {hoverInfo.val} ADU
              </span>
            )}
          </div>
        )}

        {mosaicData && (
          <div
            className="stream-histogram mosaic-viewer-histogram"
            onPointerDown={(e) => e.stopPropagation()}
            onPointerMove={(e) => e.stopPropagation()}
            onPointerUp={(e) => e.stopPropagation()}
          >
            <div className="mosaic-hist-controls">
              <span
                className="stream-histogram-label"
                title="Drag sky-blue (min) or amber (max) cursor on histogram, or hold Shift while dragging for fine 16-bit adjustment."
              >
                HIST (16-BIT LINEAR) • [{minPoint}–{maxPoint}]
              </span>
              <div className="mosaic-minmax-inputs">
                <label className="minmax-field min">
                  <span>Min</span>
                  <input
                    type="number"
                    min={0}
                    max={maxPoint - 1}
                    step={10}
                    value={minPoint}
                    onChange={(e) => {
                      const v = Math.max(0, Math.min(maxPoint - 1, parseInt(e.target.value) || 0));
                      minPointRef.current = v;
                      setMinPoint(v);
                    }}
                  />
                </label>
                <label className="minmax-field max">
                  <span>Max</span>
                  <input
                    type="number"
                    min={minPoint + 1}
                    max={65535}
                    step={50}
                    value={maxPoint}
                    onChange={(e) => {
                      const v = Math.min(65535, Math.max(minPoint + 1, parseInt(e.target.value) || 1));
                      maxPointRef.current = v;
                      setMaxPoint(v);
                    }}
                  />
                </label>
                <button
                  type="button"
                  className="hist-mini-btn"
                  onClick={handleApplyAutoStretch}
                  title={`Auto stretch (${mosaicData.autoMin} – ${mosaicData.autoMax})`}
                >
                  Auto
                </button>
                <button
                  type="button"
                  className="hist-mini-btn"
                  onClick={handleResetFullRange}
                  title="Full 16-bit range (0 – 65535)"
                >
                  0–65535
                </button>
                <button
                  type="button"
                  className={`mosaic-hist-save-btn ${viewSavedMsg ? 'saved' : ''}`}
                  onClick={handleSaveViewState}
                  title="Save current histogram window & camera rotation as defaults for this stack"
                >
                  <Save size={11} /> {viewSavedMsg || 'Save View'}
                </button>
              </div>
            </div>
            <canvas
              ref={histCanvasRef}
              width={512}
              height={54}
              className="stream-histogram-canvas"
              title="Drag sky-blue (min) or amber (max) cursor to stretch 16-bit linear mosaic. Hold Shift while dragging for 10x finer control. Double-click to toggle Auto / Full."
              onPointerDown={handleHistPointerDown}
              onPointerMove={handleHistPointerMove}
              onPointerUp={handleHistPointerUp}
              onPointerCancel={handleHistPointerUp}
              onDoubleClick={() => {
                if (minPoint === 0 && maxPoint === 65535) {
                  handleApplyAutoStretch();
                } else {
                  handleResetFullRange();
                }
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

function App() {
  const urlParams = new URLSearchParams(window.location.search);
  const standaloneMosaicId = urlParams.get('mosaic');
  if (standaloneMosaicId) {
    const standaloneItem: GalleryItem = {
      id: standaloneMosaicId,
      type: 'sequence',
      target: urlParams.get('target') || undefined,
      datetime: urlParams.get('datetime') || standaloneMosaicId,
      frames: urlParams.get('frames') ? parseInt(urlParams.get('frames')!, 10) : undefined,
      thumbnail: '',
      url: '',
      has_mosaic: true,
      mosaic_status: 'ready',
    };
    return (
      <MosaicViewerModal
        item={standaloneItem}
        onClose={() => window.close()}
        isStandaloneTab={true}
      />
    );
  }

  return <DashboardApp />;
}

function DashboardApp() {
  const [controls, setControls] = useState<Controls>({
    brightness: 0, contrast: 50, saturation: 0, gain: 64,
    exposure: 1000, sharpness: 50, gamma: 100, average: 1, auto_exposure: 0
  });
  const [status, setStatus] = useState<string>('Ready');
  const [logs, setLogs] = useState<string[]>([]);
  const [captures, setCaptures] = useState<GalleryItem[]>([]);
  const [galleryFilter, setGalleryFilter] = useState<'all' | 'favorites'>('all');

  const handleToggleFavorite = async (item: GalleryItem) => {
    const nextFav = !item.favorite;
    setCaptures((prev) =>
      prev.map((c) =>
        c.id === item.id
          ? { ...c, favorite: nextFav, nav: { ...(c.nav || {}), favorite: nextFav } }
          : c
      )
    );
    try {
      await fetch(`${API_BASE}/mosaic/note/${encodeURIComponent(item.id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ favorite: nextFav }),
      });
      showToast(nextFav ? '★ Added to Favorites' : 'Removed from Favorites');
    } catch (e: any) {
      showToast(`Error updating favorite: ${e.message}`, 3000);
    }
  };
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
  const [skyInputs, setSkyInputs] = useState<Record<string, string>>({});
  const [editingKey, setEditingKey] = useState<string | null>(null);
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
    if ((e.target as HTMLElement).closest('button, input, select, textarea, .stream-histogram')) return;

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
            const formatted = data.map(formatGalleryItem);
            formatted.sort((a, b) => (b.datetime || b.id).localeCompare(a.datetime || a.id));
            setCaptures(formatted);
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
    if (sky?.status?.state === 'loading') {
      const id = setInterval(loadSky, 800);
      return () => clearInterval(id);
    }
  }, [sky?.status?.state]);

  // Az/Alt change continuously as the sky rotates, so refresh them periodically
  useEffect(() => {
    const id = setInterval(loadSky, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!sky) return;
    setSkyInputs(prev => {
      const next = { ...prev };
      const keys = ['latitude', 'longitude', 'ra', 'dec', 'focal_length_mm', 'pixel_size_um'] as const;
      for (const k of keys) {
        if (editingKey !== k || next[k] === undefined) {
          next[k] = sky[k] !== undefined && sky[k] !== null ? String(sky[k]) : '';
        }
      }
      return next;
    });
  }, [sky, editingKey]);

  const getChangedInputs = (): Partial<SkyConfig> => {
    if (!sky) return {};
    const changes: Partial<SkyConfig> = {};
    const keys = ['latitude', 'longitude', 'ra', 'dec', 'focal_length_mm', 'pixel_size_um'] as const;
    for (const k of keys) {
      if (skyInputs[k] !== undefined && skyInputs[k] !== '') {
        const val = parseFloat(skyInputs[k]);
        if (!isNaN(val) && val !== sky[k]) {
          (changes as any)[k] = val;
        }
      }
    }
    return changes;
  };
  const unsavedChanges = getChangedInputs();
  const hasUnsavedInputs = Object.keys(unsavedChanges).length > 0;

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
    setSearchText(query);

    // If it's a planet or Moon/Sun, compute live ephemeris via backend (don't use stale history/cache)
    if (!isPlanetQuery(query)) {
      // 1. Check recent history (instant, offline)
      const knownHistory = sky?.history.find(h => h.name.toLowerCase() === query.toLowerCase());
      if (knownHistory) {
        setSearchText(knownHistory.name);
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
        const displayName = knownTarget.alt ? `${knownTarget.name} (${knownTarget.alt})` : knownTarget.name;
        setSearchText(displayName);
        setSearchMsg(`Pointing at ${displayName}`);
        applySky({ ra: knownTarget.ra, dec: knownTarget.dec, target: displayName });
        return;
      }
    }
    // 3. Fallback to backend live ephemeris / SIMBAD query
    setSearching(true);
    setSearchMsg(`Pointing at ${query}...`);
    setSky(prev => prev ? { ...prev, status: { ...prev.status, state: 'loading', message: `Resolving ${query}...` } } : prev);
    fetch(`${API_BASE}/mock/sky/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    }).then(r => r.json()).then(d => {
      if (d.sky) setSky(d.sky);
      if (d.success) {
        setSearchText(d.name || query);
        setSearchMsg(`Pointing at ${d.name || query}`);
      } else {
        setSearchMsg(d.error || 'Search failed');
      }
    }).catch(() => setSearchMsg('Search failed (backend unreachable)'))
      .finally(() => setSearching(false));
  };

  const applySky = (changes: Partial<SkyConfig>) => {
    // Only the edited fields are sent; values are persisted server-side in config.json
    setSky(prev => prev ? { ...prev, ...changes, status: { ...prev.status, state: 'loading', message: 'Loading star catalog...' } } : prev);
    fetch(`${API_BASE}/mock/sky`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(changes)
    }).then(r => r.json()).then(d => {
      if (d && d.success) setSky(d);
    }).catch(e => console.error("Error setting sky:", e));
  };

  const handleGo = () => {
    setSearchIsOpen(false);
    const query = searchText.trim();
    if (query && (!sky || query.toLowerCase() !== (sky.target || '').toLowerCase())) {
      searchSky(query);
    } else if (hasUnsavedInputs) {
      applySky(unsavedChanges);
    } else if (query) {
      searchSky(query);
    }
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
                  const formatted = gData.map(formatGalleryItem);
                  formatted.sort((a, b) => (b.datetime || b.id).localeCompare(a.datetime || a.id));
                  setCaptures(formatted);
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
      const params = new URLSearchParams({ mosaic: item.id });
      if (item.target) params.set('target', item.target);
      if (item.datetime) params.set('datetime', item.datetime);
      if (item.frames !== undefined) params.set('frames', String(item.frames));
      window.open(`${window.location.origin}${window.location.pathname}?${params.toString()}`, '_blank');
      return;
    }

    if (item.mosaic_status === 'processing') {
      const msg = item.mosaic_progress?.message ? `Mosaic: ${item.mosaic_progress.message}` : 'Mosaic is currently being processed.';
      showToast(`${msg} Opening preview...`, 2500);
      const targetUrl = item.mosaic_url || item.url;
      window.open(`${targetUrl}?t=${Date.now()}`, '_blank');
      return;
    }

    if (item.mosaic_status === 'queued') {
      showToast(`Mosaic is queued (#${item.mosaic_queue_pos || 1} in line). It will start automatically once active jobs finish.`, 3000);
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
        if (data.status === 'queued') {
          showToast(`Mosaic queued (#${data.queue_pos || 1} in line)`, 3000);
        }
        setTimeout(() => {
          fetch(`${API_BASE}/gallery`)
            .then(r => r.json())
            .then(gData => {
              if (Array.isArray(gData)) {
                const formatted = gData.map(formatGalleryItem);
                formatted.sort((a, b) => (b.datetime || b.id).localeCompare(a.datetime || a.id));
                setCaptures(formatted);
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
            <div 
              className="stream-histogram"
              onPointerDown={(e) => e.stopPropagation()}
              onPointerMove={(e) => e.stopPropagation()}
              onPointerUp={(e) => e.stopPropagation()}
            >
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
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Grid size={14} /> Gallery
                <div className="gallery-filter-tabs">
                  <button
                    type="button"
                    className={`gallery-filter-tab ${galleryFilter === 'all' ? 'active' : ''}`}
                    onClick={() => setGalleryFilter('all')}
                  >
                    All ({captures.length})
                  </button>
                  <button
                    type="button"
                    className={`gallery-filter-tab ${galleryFilter === 'favorites' ? 'active' : ''}`}
                    onClick={() => setGalleryFilter('favorites')}
                  >
                    ★ Favorites ({captures.filter((c) => c.favorite).length})
                  </button>
                </div>
              </div>
              <span className="count-badge">
                {galleryFilter === 'favorites' ? captures.filter((c) => c.favorite).length : captures.length}
              </span>
            </div>
            <div className="captures-grid">
              {(() => {
                const displayed = galleryFilter === 'favorites' ? captures.filter((c) => c.favorite) : captures;
                if (!Array.isArray(displayed) || displayed.length === 0) {
                  return (
                    <div className="empty-msg">
                      {galleryFilter === 'favorites'
                        ? 'No favorited sequences yet. Click the ★ icon on any card to add it to your favorites!'
                        : 'No sequences recorded yet'}
                    </div>
                  );
                }
                return displayed.map((item) => (
                  <GalleryCard
                    key={item.id}
                    item={item}
                    onSelect={handleThumbnailClick}
                    onToggleFavorite={handleToggleFavorite}
                  />
                ));
              })()}
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
                          onClick={handleGo}
                          disabled={searching || (!searchText.trim() && !hasUnsavedInputs)}
                          style={{ fontSize: '11px', minWidth: '40px' }}
                        >
                          {searching ? '...' : (hasUnsavedInputs && !searchText.trim() ? 'Apply' : 'Go')}
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
                                            setSearchText(h.name);
                                            if (isPlanetQuery(h.name)) {
                                              searchSky(h.name);
                                            } else {
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
                                              setSearchIsOpen(false);
                                              setSearchText(label);
                                              if (isPlanet) {
                                                searchSky(t.name);
                                              } else {
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
                      ] as [keyof SkyConfig, string][]).map(([key, label]) => {
                        const isDirty = skyInputs[key] !== undefined && skyInputs[key] !== '' && parseFloat(skyInputs[key]) !== (sky ? (sky as any)[key] : undefined);
                        return (
                          <label key={key} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                            <span style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px' }}>
                              <span>{label}</span>
                              {isDirty && <span style={{ color: 'var(--accent, #58a6ff)', fontSize: '9px', fontWeight: 600 }}>unsaved</span>}
                            </span>
                            <input
                              type="number"
                              step="any"
                              value={skyInputs[key] ?? ''}
                              onFocus={() => setEditingKey(key)}
                              onChange={e => setSkyInputs(prev => ({ ...prev, [key]: e.target.value }))}
                              onBlur={() => {
                                setEditingKey(null);
                                const v = parseFloat(skyInputs[key] || '');
                                if (!isNaN(v) && v !== (sky ? (sky as any)[key] : undefined)) {
                                  applySky({ [key]: v } as Partial<SkyConfig>);
                                }
                              }}
                              onKeyDown={e => {
                                if (e.key === 'Enter') {
                                  (e.target as HTMLInputElement).blur();
                                }
                              }}
                              style={{
                                width: '100%',
                                fontFamily: 'monospace',
                                fontSize: '11px',
                                borderColor: isDirty ? 'var(--accent, #58a6ff)' : undefined
                              }}
                            />
                          </label>
                        );
                      })}
                    </div>
                    {hasUnsavedInputs && (
                      <div style={{ marginTop: '6px', display: 'flex', justifyContent: 'flex-end', gap: '6px' }}>
                        <button
                          onClick={() => {
                            if (sky) {
                              const reset: Record<string, string> = {};
                              (['latitude', 'longitude', 'ra', 'dec', 'focal_length_mm', 'pixel_size_um'] as const).forEach(k => {
                                reset[k] = sky[k] !== undefined && sky[k] !== null ? String(sky[k]) : '';
                              });
                              setSkyInputs(reset);
                              setEditingKey(null);
                            }
                          }}
                          style={{ fontSize: '10px', padding: '2px 8px' }}
                        >
                          Discard
                        </button>
                        <button
                          onClick={() => {
                            setEditingKey(null);
                            applySky(unsavedChanges);
                          }}
                          className="btn-primary"
                          style={{ fontSize: '10px', padding: '2px 8px' }}
                        >
                          Apply Changes
                        </button>
                      </div>
                    )}
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
