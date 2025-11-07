import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { API_BASE_URL, INFERENCE_API_URL } from './config.js';
import './hka-ui.css';
import {
  Folder,
  Play,
  Trash2,
  RefreshCcw,
  Search,
  UploadCloud,
  Settings
} from 'lucide-react';

// Status class helpers (Tailwind palette approximation)
function statusClass(key) {
  switch ((key || '').toString().toLowerCase()) {
    case 'approved':
      return 'bg-green-900/40 text-green-400';
    case 'processing':
      return 'bg-cyan-900/40 text-cyan-400';
    case 'pending':
    case 'available':
    case 'retrieved':
    case 'received':
      return 'bg-amber-900/30 text-amber-400';
    case 'accepted':
      return 'bg-emerald-900/40 text-emerald-400';
    default:
      return 'bg-gray-900/60 text-gray-400';
  }
}
function dotClass(key) {
  switch ((key || '').toString().toLowerCase()) {
    case 'approved':
      return 'bg-green-500';
    case 'processing':
      return 'bg-cyan-500';
    case 'pending':
    case 'available':
    case 'retrieved':
    case 'received':
      return 'bg-amber-500';
    case 'accepted':
      return 'bg-emerald-500';
    default:
      return 'bg-gray-500';
  }
}


export default function Worklist() {
  const navigate = useNavigate();
  const username = localStorage.getItem('username');

  // Data
  const [folders, setFolders] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Selection and pagination
  const [selected, setSelected] = useState(new Set());
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  // Filters and sorting
  const [searchText, setSearchText] = useState('');
  const [selectedStatus, setSelectedStatus] = useState('');
  const [sortField, setSortField] = useState('LastUpdated');
  const [sortDirection, setSortDirection] = useState('desc');

  // Date filters
  const [quickDatePreset, setQuickDatePreset] = useState('');
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [showDateModal, setShowDateModal] = useState(false);

  // UI state
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [toDelete, setToDelete] = useState(null);
  // Removed annotate confirmation popup; click acts immediately
  const [showRetrieveModal, setShowRetrieveModal] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);

  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [annotatingId, setAnnotatingId] = useState(null);
  // (Annotate modal had transient fields; reverting to simple confirm)

  // Upload
  const fileInputRef = useRef(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  // CRA env var for upload endpoint (defaults to 'upload')
  const uploadEndpoint = (process.env.REACT_APP_UPLOAD_ENDPOINT || 'upload').replace(/^\/+|\/+$/g, '');

  useEffect(() => {
    fetchFolders();
  }, []);

  const fetchFolders = async () => {
    setIsLoading(true);
    setError('');
    try {
      // Prefer DB-backed worklist, fall back to MinIO listing
      let response;
      try {
        response = await axios.get(`${API_BASE_URL}/studies`);
      } catch (e) {
        response = await axios.get(`${API_BASE_URL}/folders/`);
      }
      const list = response?.data?.folders || [];
      setFolders(list);
    } catch (e) {
      console.error('Failed to fetch folders', e);
      setError('Failed to fetch worklist');
    } finally {
      setIsLoading(false);
    }
  };

  // Transform folders to study-like entries used by the new UI
  const entries = useMemo(() => {
    return (folders || []).map((f) => ({
      id: f.name,
      PatientName: f.patient_name || f.name || 'Unknown',
      PatientID: f.patient_id || f.name || '',
      AccessionNumber: f.accession || f.name || '',
      StudyDescription: f.description || '-',
      StudyInstanceUID: f.study_instance_uid || '',
      Modality: f.modality || 'Unknown',
      Status: f.status || 'Unknown',
      CreatedByUsername: f.processing_by || f.completed_by || '',
      LastUpdated: f.last_updated || f.updated_at || f.created_at || null,
      _raw: f,
    }));
  }, [folders]);

  const availableStatuses = useMemo(() => {
    const set = new Set(entries.map((e) => (e.Status || 'Unknown').toString().toLowerCase()));
    return Array.from(set).sort();
  }, [entries]);

  const withinRange = (d) => {
    if (!d || (!startDate && !endDate)) return true;
    const ts = new Date(d).getTime();
    if (Number.isNaN(ts)) return true;
    if (startDate && ts < startDate.getTime()) return false;
    if (endDate && ts > endDate.getTime()) return false;
    return true;
  };

  const filtered = useMemo(() => {
    const q = searchText.trim().toLowerCase();
    return entries.filter((e) => {
      const matchText = !q ||
        (e.PatientName && e.PatientName.toLowerCase().includes(q)) ||
        (e.PatientID && e.PatientID.toLowerCase().includes(q)) ||
        (e.AccessionNumber && e.AccessionNumber.toLowerCase().includes(q));
      const matchStatus = !selectedStatus || (e.Status || '').toString().toLowerCase() === selectedStatus;
      const matchDate = withinRange(e.LastUpdated);
      return matchText && matchStatus && matchDate;
    });
  }, [entries, searchText, selectedStatus, startDate, endDate]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const dir = sortDirection === 'asc' ? 1 : -1;
      const av = a[sortField];
      const bv = b[sortField];
      if (sortField === 'LastUpdated') {
        const at = av ? new Date(av).getTime() : 0;
        const bt = bv ? new Date(bv).getTime() : 0;
        return (at - bt) * dir;
      }
      const as = (av || '').toString().toLowerCase();
      const bs = (bv || '').toString().toLowerCase();
      if (as < bs) return -1 * dir;
      if (as > bs) return 1 * dir;
      return 0;
    });
    return arr;
  }, [filtered, sortField, sortDirection]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const pageData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return sorted.slice(start, start + pageSize);
  }, [sorted, currentPage]);

  useEffect(() => {
    // reset to first page on filter/sort changes
    setCurrentPage(1);
  }, [searchText, selectedStatus, sortField, sortDirection]);

  // Selection helpers
  const allVisibleSelected = pageData.length > 0 && pageData.every((e) => selected.has(e.id));
  const toggleSelectAllVisible = () => {
    const next = new Set(selected);
    if (allVisibleSelected) {
      pageData.forEach((e) => next.delete(e.id));
    } else {
      pageData.forEach((e) => next.add(e.id));
    }
    setSelected(next);
  };
  const toggleRow = (id, checked) => {
    const next = new Set(selected);
    if (checked) next.add(id); else next.delete(id);
    setSelected(next);
  };
  const allSelectedApproved = () => {
    if (selected.size === 0) return false;
    const map = new Map(entries.map((e) => [e.id, e]));
    for (const id of selected) {
      const e = map.get(id);
      const key = getStatusKey(e?.Status);
      if (key !== 'approved') return false;
    }
    return true;
  };

  // Status helpers
  const getStatusKey = (s) => (s || 'unknown').toString().toLowerCase();
  const getStatusLabel = (s) => s || 'Unknown';

  // Actions
  const handleAnnotateClick = async (entry) => {
  const status = entry._raw?.status;
  const processingBy = entry._raw?.processing_by;
  
  // If already completed or being processed by someone else, just view
  if (status === 'Completed' || (processingBy && processingBy !== username)) {
    navigate(`/dashboard`);
    return;
  }
  
  try {
    setError('');
    setAnnotatingId(entry.id);
    
    // Fetch presigned NIfTI URL for this study from backend
    console.log(`Fetching NIfTI URL for folder: ${entry.id}`);
    const presign = await axios.get(`${API_BASE_URL}/folders/${encodeURIComponent(entry.id)}/nifti-url`);
    console.log('Presign response:', presign.data);
    
    const niftiUrl = presign?.data?.nifti_url || presign?.data?.url || presign?.data?.presigned_url;
    if (!niftiUrl) {
      throw new Error('No NIfTI URL returned for this study');
    }
    
    // Extract metadata from the presign response
    const meta = presign?.data?.meta || {};
    
    // Get age - try multiple sources
    let age = meta.age || entry?._raw?.age || entry?.Age;
    if (age && typeof age === 'string') {
      age = parseFloat(age);
    }
    
    // Get gender - try multiple sources and normalize
    let gender = meta.gender || entry?._raw?.gender || entry?.Gender;
    if (gender) {
      const g = gender.toString().trim().toLowerCase();
      if (g === 'male' || g === 'm') {
        gender = 'M';
      } else if (g === 'female' || g === 'f') {
        gender = 'F';
      }
    }
    
    // Debug: Log what we found
    console.log('Debug metadata extraction:', {
      'meta.age': meta.age,
      'entry._raw.age': entry?._raw?.age,
      'entry.Age': entry?.Age,
      'final age': age,
      'meta.gender': meta.gender,
      'entry._raw.gender': entry?._raw?.gender,
      'entry.Gender': entry?.Gender,
      'final gender': gender,
      'entry._raw': entry?._raw
    });
    
    // Validate we have required metadata
    if (!age || !gender) {
      throw new Error(`Missing required metadata: age=${age}, gender=${gender}. Please ensure age and gender are provided when uploading the study.`);
    }
    
    // Build payload for inference
    const payload = {
      nifti_url: niftiUrl,
      age: age,
      gender: gender
    };
    
    if (username) {
      payload.username = username;
    }
    
    console.log('Sending inference payload:', payload);
    
    // Run both inference endpoints in parallel
    const [brainAgeResponse, normativeResponse] = await Promise.all([
      axios.post(`${INFERENCE_API_URL}/brain-age`, payload, {
        headers: { 'Content-Type': 'application/json' }
      }),
      axios.post(`${INFERENCE_API_URL}/normative`, payload, {
        headers: { 'Content-Type': 'application/json' }
      }),
    ]);
    
    console.log('Brain Age Response:', brainAgeResponse.data);
    console.log('Normative Response:', normativeResponse.data);
    
    // Combine results and navigate to dashboard with data
    const analysisResults = {
      brainAge: brainAgeResponse.data,
      normative: normativeResponse.data
    };
    
    // Navigate to dashboard with results as state
    navigate('/dashboard', { 
      state: { 
        analysisResults,
        patientAge: age,
        patientGender: gender
      }
    });
    
  } catch (err) {
    console.error('Inference failed:', err);
    console.error('Error response:', err?.response?.data);
    
    const msg = err?.response?.data?.detail || err?.response?.data?.message || err?.message || 'Failed to run analysis';
    setError(msg);
  } finally {
    setAnnotatingId(null);
  }
};

  // Print actions removed
  const deleteStudy = (entry) => {
    setToDelete(entry);
    setShowDeleteConfirm(true);
  };
  const confirmDelete = async () => {
    // Placeholder: implement API delete if available
    console.log('Delete', toDelete);
    setShowDeleteConfirm(false);
    setToDelete(null);
  };
  const cancelDelete = () => {
    setShowDeleteConfirm(false);
    setToDelete(null);
  };

  // Sorting toggle
  const toggleSort = (field) => {
    if (sortField === field) {
      setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Date preset handler (simplified)
  const onQuickPresetChange = (val) => {
    setQuickDatePreset(val);
    const now = new Date();
    const start = new Date();
    if (val === 'today') {
      start.setHours(0, 0, 0, 0);
      setStartDate(start); setEndDate(now);
    } else if (val === 'yesterday') {
      const y1 = new Date(now); y1.setDate(now.getDate() - 1); y1.setHours(0,0,0,0);
      const y2 = new Date(now); y2.setDate(now.getDate() - 1); y2.setHours(23,59,59,999);
      setStartDate(y1); setEndDate(y2);
    } else if (val === '2days') {
      const d1 = new Date(now); d1.setDate(now.getDate() - 2); d1.setHours(0,0,0,0);
      const d2 = new Date(now); d2.setDate(now.getDate() - 2); d2.setHours(23,59,59,999);
      setStartDate(d1); setEndDate(d2);
    } else if (val === 'last7days') {
      const d1 = new Date(now); d1.setDate(now.getDate() - 7); d1.setHours(0,0,0,0);
      setStartDate(d1); setEndDate(now);
    } else if (val === 'custom') {
      setShowDateModal(true);
    } else {
      setStartDate(null); setEndDate(null);
    }
  };

  const clearDateRange = (e) => {
    e?.stopPropagation?.();
    setQuickDatePreset('');
    setStartDate(null);
    setEndDate(null);
  };

  const formattedDate = (d) => {
    if (!d) return '';
    const dt = new Date(d);
    if (Number.isNaN(dt.getTime())) return '';
    return dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  };
  const formattedTime = (d) => {
    if (!d) return '';
    const dt = new Date(d);
    if (Number.isNaN(dt.getTime())) return '';
    return dt.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
  };

  // Upload handlers (placeholders)
  const onFileSelected = (e) => {
    setError('');
    setSelectedFiles(Array.from(e.target.files || []));
  };
  const processSelectedFile = async () => {
    if (!selectedFiles.length) return;
    setIsLoading(true);
    setError('');
    try {
      const form = new FormData();
      selectedFiles.forEach((f) => form.append('files', f));
      if (age) form.append('age', age);
      if (gender) form.append('gender', gender);
      if (username) form.append('username', username);

      await axios.post(`${API_BASE_URL}/${uploadEndpoint}`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        maxBodyLength: Infinity,
      });

      setSelectedFiles([]);
      // Refresh worklist after successful upload
      await fetchFolders();
    } catch (e) {
      console.error('Upload failed', e);
      const msg = e?.response?.data?.message || e?.message || 'Upload failed';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="home-container min-h-screen bg-gray-900 text-gray-100">
      {/* Action Bar */}
      <div className="action-bar bg-gray-800 border-b border-gray-700 px-4 py-3 flex items-center gap-3">
        <div className="upload-section">
          <label htmlFor="dicomUpload" className="upload-btn">
            <UploadCloud className="w-4 h-4" />
            Upload Study Files
          </label>
          <input id="dicomUpload" ref={fileInputRef} type="file" multiple accept=".dcm,.dicom,.nii,.nii.gz,.gz" onChange={onFileSelected} style={{ display: 'none' }} />
          <div className="inline-field">
            <label htmlFor="age">Age</label>
            <input
              id="age"
              type="number"
              min="0"
              max="120"
              placeholder="Years"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className="input-sm"
              style={{ width: '80px' }}
            />
          </div>
          <div className="inline-field">
            <label htmlFor="gender">Gender</label>
            <select
              id="gender"
              value={gender}
              onChange={(e) => setGender(e.target.value)}
              className="input-sm"
              style={{ width: '120px' }}
            >
              <option value="">Select</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>
          {selectedFiles.length > 0 && (
            <div className="file-selected">
              <span>{selectedFiles.length === 1 ? selectedFiles[0].name : `${selectedFiles.length} files selected`}</span>
              <button className="btn btn-primary btn-xs" onClick={processSelectedFile} disabled={isLoading}>
                {isLoading ? 'Uploading…' : 'Upload to Worklist'}
              </button>
            </div>
          )}
          {error && (
            <div className="text-sm" style={{ color: '#f87171' }}>{error}</div>
          )}
        </div>
        {/* </div> */}
        

      {/* Worklist Main */}
      <div className="worklist-main p-6 max-w-[1600px] mx-auto w-full">
        <div className="worklist-header flex items-center justify-between mb-6">
          <div className="header-left flex items-baseline gap-3">
            <h1 className="worklist-title text-2xl font-semibold text-white m-0">Study Worklist</h1>
            <span className="study-count text-sm text-gray-400 bg-gray-800 px-2 py-0.5 rounded-full">{entries.length} studies</span>
          </div>
          <div className="header-actions flex gap-2">
            <button className="btn btn-ghost btn-sm inline-flex items-center gap-2 px-3 py-2 bg-gray-800 rounded border border-gray-700" onClick={fetchFolders} disabled={isLoading}>
              <RefreshCcw className="w-4 h-4" /> Refresh
            </button>
            {/* Print actions removed as requested */}
          </div>
        </div>

        <div className="worklist-content bg-gray-800 border border-pink-600 rounded-xl overflow-hidden shadow">
          {/* Controls */}
          <div className="worklist-controls flex flex-nowrap whitespace-nowrap gap-3 p-4 items-center justify-start overflow-x-auto bg-gray-900/40 border-b border-gray-700">
            <input
              type="text"
              className="form-control flex-1 min-w-[220px] px-3 py-2 bg-gray-900 border border-gray-700 rounded-md"
              placeholder="Search by Patient Name or ID"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
            />
            <select
              className="form-control w-[200px] flex-none px-3 py-2 bg-gray-900 border border-gray-700 rounded-md"
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
            >
              <option value="">All Statuses</option>
              {availableStatuses.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <div className="date-filter flex items-center gap-2 flex-none">
              <select
                className="form-control px-3 py-2 bg-gray-900 border border-gray-700 rounded-md"
                value={quickDatePreset}
                onChange={(e) => onQuickPresetChange(e.target.value)}
              >
                <option value="">All time</option>
                <option value="today">Today</option>
                <option value="yesterday">Yesterday</option>
                <option value="2days">Day Before Yesterday</option>
                <option value="last7days">Last 7 days</option>
                <option value="custom">Custom</option>
              </select>
              {quickDatePreset === 'custom' && (
                <button className="px-3 py-2 bg-gray-900 border border-gray-700 rounded-md" onClick={() => setShowDateModal(true)}>Pick Range</button>
              )}
            </div>
          </div>

          {/* Table */}
          <div className="table-wrapper w-full">
            <table className="worklist-table w-full border-collapse text-sm">
              <thead className="bg-pink-500">
                <tr className="bg-pink-500 sticky top-0 z-10">
                  <th className="col-select p-3 border-b border-gray-700 w-[40px] text-center">
                    <input type="checkbox" checked={allVisibleSelected} onChange={toggleSelectAllVisible} aria-label="Select all visible" />
                  </th>
                  <th className="col-patient p-3 border-b border-gray-700 cursor-pointer" onClick={() => toggleSort('PatientName')}>
                    Patient {sortField === 'PatientName' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="col-study p-3 border-b border-gray-700 cursor-pointer" onClick={() => toggleSort('AccessionNumber')}>
                    Study Details {sortField === 'AccessionNumber' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="col-modality p-3 border-b border-gray-700">Modality</th>
                  <th className="col-status p-3 border-b border-gray-700 cursor-pointer" onClick={() => toggleSort('Status')}>
                    Status {sortField === 'Status' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="col-createdby p-3 border-b border-gray-700 cursor-pointer" onClick={() => toggleSort('CreatedByUsername')}>
                    Created By {sortField === 'CreatedByUsername' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="col-date p-3 border-b border-gray-700 cursor-pointer" onClick={() => toggleSort('LastUpdated')}>
                    Date {sortField === 'LastUpdated' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  <th className="col-actions p-3 border-b border-gray-700">Actions</th>
                </tr>
              </thead>
              <tbody>
                {isLoading && (
                  <tr>
                    <td colSpan={8} className="p-8 text-center text-gray-400">Loading studies...</td>
                  </tr>
                )}
                {!isLoading && sorted.length === 0 && (
                  <tr>
                    <td colSpan={8} className="p-8 text-center text-gray-400">No Studies Available</td>
                  </tr>
                )}
                {pageData.map((entry) => (
                  <tr key={entry.id} className="study-row hover:bg-gray-900/40">
                    <td className="col-select p-3 text-center">
                      <input type="checkbox" checked={selected.has(entry.id)} onChange={(e) => toggleRow(entry.id, e.target.checked)} aria-label="Select row" />
                    </td>
                    <td className="col-patient p-3">
                      <div className="patient-cell flex items-center gap-3">
                        <div className="patient-avatar w-9 h-9 rounded-full bg-gradient-to-tr from-cyan-900 to-blue-600 flex items-center justify-center text-white">
                          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21V19A4 4 0 0 0 16 15H8A4 4 0 0 0 4 19V21"></path><circle cx="12" cy="7" r="4"></circle></svg>
                        </div>
                        <div className="patient-info min-w-0 flex-1">
                          <div className="patient-name font-semibold text-white truncate">{entry.PatientName || 'Unknown Patient'}</div>
                          <div className="patient-id text-xs text-gray-400">{entry.PatientID || 'No ID'}</div>
                        </div>
                      </div>
                    </td>
                    <td className="col-study p-3">
                      <div className="study-cell flex flex-col gap-0.5">
                        <div className="accession-number font-semibold text-white text-sm">{entry.AccessionNumber || 'No Accession'}</div>
                        <div className="study-description text-gray-400 text-sm truncate">{entry.StudyDescription || 'No Description'}</div>
                        {entry.StudyInstanceUID && (
                          <div className="study-uid text-xs text-gray-500">UID: {String(entry.StudyInstanceUID).slice(0, 20)}...</div>
                        )}
                      </div>
                    </td>
                    <td className="col-modality p-3">
                      <div className="modality-badge inline-flex items-center px-2 py-0.5 bg-gray-900 border border-gray-700 rounded-md">
                        <span className="modality-text text-xs font-semibold text-white uppercase tracking-wider">{entry.Modality || 'Unknown'}</span>
                      </div>
                    </td>
                    <td className="col-status p-3">
                      <span className={`status-badge inline-flex items-center gap-2 px-2 py-0.5 rounded-full text-xs font-medium ${statusClass(getStatusKey(entry.Status))}`}>
                        <span className={`status-dot w-1.5 h-1.5 rounded-full ${dotClass(getStatusKey(entry.Status))}`}></span>
                        {getStatusLabel(entry.Status)}
                      </span>
                    </td>
                    <td className="col-createdby p-3">
                      {entry.CreatedByUsername || '—'}
                    </td>
                    <td className="col-date p-3">
                      <div className="date-cell flex flex-col">
                        <div className="date-primary text-white text-sm">{formattedDate(entry.LastUpdated) || '—'}</div>
                        <div className="date-time text-xs text-gray-400">{formattedTime(entry.LastUpdated)}</div>
                      </div>
                    </td>
                    <td className="col-actions p-3">
                      <div className="action-buttons flex items-center gap-2">
                        {/* Print buttons removed */}
                        <button className="btn btn-ghost btn-sm inline-flex items-center gap-1 px-2 py-1 bg-gray-900 border border-gray-700 rounded text-red-400" onClick={() => deleteStudy(entry)} title="Delete from worklist">
                          <Trash2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleAnnotateClick(entry)}
                          disabled={annotatingId === entry.id}
                          className={`px-3 py-1 rounded-md text-sm inline-flex items-center gap-1.5 ${
                            (entry._raw?.status === 'Completed' || (entry._raw?.processing_by && entry._raw?.processing_by !== username))
                              ? 'bg-blue-600 hover:bg-blue-500'
                              : (annotatingId === entry.id ? 'bg-gray-600 cursor-not-allowed' : 'bg-cyan-600 hover:bg-cyan-500')
                          } text-white transition`}
                        >
                          <Play className="w-4 h-4" />
                          {annotatingId === entry.id
                            ? 'Analyzing…'
                            : ((entry._raw?.status === 'Completed' || (entry._raw?.processing_by && entry._raw?.processing_by !== username)) ? 'View' : 'Annotate')}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {/* Pagination */}
            <div className="pagination-controls flex items-center justify-center gap-4 my-3 text-gray-200">
              <button className="px-2 py-1 bg-gray-900 border border-gray-700 rounded" disabled={currentPage === 1} onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}>&larr;</button>
              <span>Page {currentPage} of {totalPages}</span>
              <button className="px-2 py-1 bg-gray-900 border border-gray-700 rounded" disabled={currentPage === totalPages} onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}>&rarr;</button>
            </div>
          </div>
        </div>
      </div>

      {/* Annotate confirmation popup removed: click acts immediately */}

      {/* Delete confirmation modal */}
      {showDeleteConfirm && (
        <div className="modal-overlay fixed inset-0 bg-black/70 flex items-center justify-center">
          <div className="modal-content bg-gray-800 rounded-xl border border-gray-700 max-w-lg w-full">
            <div className="modal-header flex items-center justify-between p-4 border-b border-gray-700">
              <h2 className="modal-title text-lg font-semibold text-white">Confirm Deletion</h2>
              <button className="modal-close text-gray-400" onClick={cancelDelete}>✕</button>
            </div>
            <div className="modal-body p-4">
              <p>Are you sure you want to delete the study for <strong>{toDelete?.PatientName}</strong>?</p>
              <p className="text-sm text-gray-400">Accession Number: {toDelete?.AccessionNumber || '—'}</p>
              <div className="modal-actions flex justify-end gap-2 mt-4">
                <button className="btn btn-secondary px-3 py-2 bg-gray-700 rounded" onClick={cancelDelete}>Cancel</button>
                <button className="btn btn-danger px-3 py-2 bg-red-600 rounded" onClick={confirmDelete}>Delete Study</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Retrieve & Config modals (placeholders) */}
      {showRetrieveModal && (
        <div className="modal-overlay fixed inset-0 bg-black/70 flex items-center justify-center" onClick={() => setShowRetrieveModal(false)}>
          <div className="modal-content bg-gray-800 rounded-xl border border-gray-700 max-w-xl w-full" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header p-4 border-b border-gray-700 flex justify-between items-center">
              <h2 className="text-lg font-semibold">Retrieve DICOM Study</h2>
              <button onClick={() => setShowRetrieveModal(false)}>✕</button>
            </div>
            <div className="modal-body p-4 text-gray-300">Coming soon…</div>
          </div>
        </div>
      )}
      {showConfigModal && (
        <div className="modal-overlay fixed inset-0 bg-black/70 flex items-center justify-center" onClick={() => setShowConfigModal(false)}>
          <div className="modal-content bg-gray-800 rounded-xl border border-gray-700 max-w-xl w-full" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header p-4 border-b border-gray-700 flex justify-between items-center">
              <h2 className="text-lg font-semibold">Measurement Configuration</h2>
              <button onClick={() => setShowConfigModal(false)}>✕</button>
            </div>
            <div className="modal-body p-4 text-gray-300">Coming soon…</div>
          </div>
        </div>
      )}

      {/* Custom date range modal (simplified) */}
      {showDateModal && (
        <div className="date-modal-overlay fixed inset-0 bg-black/60 flex items-center justify-center" onClick={() => setShowDateModal(false)}>
          <div className="date-modal bg-gray-800 border border-gray-700 rounded-xl w-[420px] max-w-[90vw]" onClick={(e) => e.stopPropagation()}>
            <div className="date-modal-header flex items-center justify-end p-2 border-b border-gray-700">
              <button className="modal-close" onClick={() => setShowDateModal(false)}>✕</button>
            </div>
            <div className="date-modal-body p-4 flex gap-3">
              <div className="date-summary grid gap-2">
                <div>
                  <span className="block text-xs text-gray-400">Start date</span>
                  <span className="text-sm">{startDate ? formattedDate(startDate) : '—'}</span>
                </div>
                <div>
                  <span className="block text-xs text-gray-400">End date</span>
                  <span className="text-sm">{endDate ? formattedDate(endDate) : '—'}</span>
                </div>
              </div>
              <div className="calendar-panel flex-1">
                <div className="flex flex-col gap-2">
                  <input type="date" className="form-control px-2 py-1 bg-gray-900 border border-gray-700 rounded" onChange={(e) => setStartDate(e.target.value ? new Date(e.target.value) : null)} />
                  <input type="date" className="form-control px-2 py-1 bg-gray-900 border border-gray-700 rounded" onChange={(e) => setEndDate(e.target.value ? new Date(e.target.value) : null)} />
                  <div className="flex justify-end gap-2">
                    <button className="px-3 py-1 bg-gray-700 rounded" onClick={(e) => { clearDateRange(e); setShowDateModal(false); }}>Clear</button>
                    <button className="px-3 py-1 bg-cyan-600 rounded" onClick={() => setShowDateModal(false)}>Apply</button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
    </div>
  );
}
 
