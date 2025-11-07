import React, { useEffect, useMemo, useState } from 'react';
import { useKeycloak } from '@react-keycloak/web';
import axios from 'axios';
import { API_BASE_URL } from './config.js';
import './hka-ui.css';

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch (_) {
    return iso;
  }
}

export default function SftpFiles() {
  const { keycloak } = useKeycloak();
  const username = keycloak?.tokenParsed?.preferred_username || 'user';
  const uploadEndpoint = (process.env.REACT_APP_UPLOAD_ENDPOINT || 'upload').replace(/^\/+|\/+$/g, '');

  const storageKey = `niiUploads:${username}`;
  const [uploads, setUploads] = useState([]);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [uploadedFileNames, setUploadedFileNames] = useState([]);
  const [query, setQuery] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Load persisted uploads for this user
  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw) setUploads(JSON.parse(raw));
    } catch (_) {}
  }, [storageKey]);

  // Persist on change
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(uploads));
    } catch (_) {}
  }, [uploads, storageKey]);

  const onFilesSelected = (e) => {
    setError('');
    const list = Array.from(e.target.files || []);
    if (list.length === 0) return;
    const invalid = list.filter((f) => !/\.nii(\.gz)?$/i.test(f.name));
    if (invalid.length) {
      setError(`Only .nii or .nii.gz files are allowed. Invalid: ${invalid.map((f) => f.name).join(', ')}`);
      return;
    }
    setPendingFiles(list);
    setUploadedFileNames(list.map((f) => f.name));
    e.target.value = '';
  };

  const processSelectedFile = async () => {
    if (!pendingFiles.length) return;
    setIsLoading(true);
    setError('');
    try {
      const form = new FormData();
      pendingFiles.forEach((f) => form.append('files', f));
      if (age) form.append('age', age);
      if (gender) form.append('gender', gender);
      if (username) form.append('username', username);

      await axios.post(`${API_BASE_URL}/${uploadEndpoint}`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        maxBodyLength: Infinity,
      });

      // Optimistically record the upload locally with timestamp
      const now = new Date().toISOString();
      setUploads((prev) => {
        const next = [...prev];
        for (const f of pendingFiles) {
          const idx = next.findIndex((x) => x.name === f.name);
          const item = {
            id: `${now}-${f.name}`,
            name: f.name,
            createdAt: now,
            createdBy: username,
            status: 'Uploaded',
          };
          if (idx >= 0) next[idx] = item; else next.push(item);
        }
        return next.sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
      });

      setPendingFiles([]);
      setUploadedFileNames([]);
    } catch (e) {
      console.error('Upload failed', e);
      const msg = e?.response?.data?.message || e?.message || 'Upload failed';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  const removeItem = (id) => {
    setUploads((prev) => prev.filter((x) => x.id !== id));
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return uploads;
    return uploads.filter((u) => u.name.toLowerCase().includes(q));
  }, [uploads, query]);

  return (
    <div className="home-container">
      <div className="action-bar">
        <div className="upload-section">
          <label htmlFor="niiUpload" className="upload-btn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15V19A2 2 0 0 1 19 21H5A2 2 0 0 1 3 19V15"></path>
              <polyline points="17,8 12,3 7,8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            Upload .nii/.nii.gz
          </label>
          <input id="niiUpload" type="file" accept=".nii,.nii.gz,.gz" multiple onChange={onFilesSelected} style={{ display: 'none' }} />

          {uploadedFileNames.length > 0 && (
            <div className="file-selected">
              {uploadedFileNames.length === 1 ? (
                <div className="file-info">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6A2 2 0 0 0 4 4V20A2 2 0 0 0 6 22H18A2 2 0 0 0 20 20V8Z"></path>
                    <polyline points="14,2 14,8 20,8"></polyline>
                  </svg>
                  <span className="file-name">{uploadedFileNames[0]}</span>
                </div>
              ) : (
                <div className="file-info">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6A2 2 0 0 0 4 4V20A2 2 0 0 0 6 22H18A2 2 0 0 0 20 20V8Z"></path>
                    <polyline points="14,2 14,8 20,8"></polyline>
                  </svg>
                  <span className="file-name">{uploadedFileNames.length} files selected</span>
                </div>
              )}
              <button className="btn btn-primary btn-xs" onClick={processSelectedFile} disabled={isLoading}>
                {isLoading ? 'Uploading…' : 'Upload to Worklist'}
              </button>
            </div>
          )}

          {uploadedFileNames.length > 1 && (
            <div className="selected-files-list">
              <h4>Selected Files:</h4>
              {uploadedFileNames.map((n, i) => (
                <div key={n} className="file-item">
                  <span className="file-name-small">{i + 1}. {n}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Age & Gender inputs */}
        <div style={{ display: 'flex', alignItems: 'end', gap: '12px' }}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <label htmlFor="age" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>Age</label>
            <input
              id="age"
              type="number"
              min="0"
              max="120"
              placeholder="Years"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              style={{ padding: 'var(--space-2)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border-primary)', borderRadius: 'var(--radius-md)', width: '6rem' }}
            />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <label htmlFor="gender" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>Gender</label>
            <select
              id="gender"
              value={gender}
              onChange={(e) => setGender(e.target.value)}
              style={{ padding: 'calc(var(--space-2) - 2px)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border-primary)', borderRadius: 'var(--radius-md)', width: '8rem' }}
            >
              <option value="">Select</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
              <option value="Unknown">Unknown</option>
            </select>
          </div>
        </div>
        <div className="spacer" />
      </div>

      <div className="worklist-main">
        <div className="worklist-header">
          <div className="header-left">
            <h1 className="worklist-title">Study Worklist</h1>
            {uploads.length > 0 && (
              <span className="study-count">{uploads.length} studies</span>
            )}
          </div>
          <div className="header-actions">
            <button className="btn btn-ghost btn-sm" onClick={() => setUploads((prev) => [...prev])} disabled={isLoading}>
              Refresh
            </button>
          </div>
        </div>

        <div className="worklist-content">
          <div className="worklist-table-container">
            <div className="table-wrapper">
              <table className="worklist-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Created Date</th>
                    <th>Created By</th>
                    <th>Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((u) => (
                    <tr className="study-row" key={u.id}>
                      <td style={{ fontFamily: 'monospace' }}>{u.name}</td>
                      <td>
                        <div className="date-cell">
                          <div className="date-primary">{new Date(u.createdAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</div>
                          <div className="date-time">{new Date(u.createdAt).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })}</div>
                        </div>
                      </td>
                      <td>{u.createdBy}</td>
                      <td>
                        <span className="status-badge status-available">
                          <span className="status-dot" />
                          {u.status}
                        </span>
                      </td>
                      <td>
                        <div className="action-buttons">
                          <button className="btn btn-ghost btn-sm" onClick={() => removeItem(u.id)}>Remove</button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {filtered.length === 0 && (
                    <tr>
                      <td colSpan={5} style={{ padding: 12, textAlign: 'center', color: 'var(--text-secondary)' }}>No uploads yet</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
