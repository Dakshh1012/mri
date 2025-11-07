// Centralized API base URL configuration
// CRA uses `process.env.REACT_APP_*` for env vars.

const API_PROTOCOL = process.env.REACT_APP_API_PROTOCOL || 'http';
const API_HOST = process.env.REACT_APP_API_HOST || '127.0.0.1';
const API_PORT = process.env.REACT_APP_API_PORT || '7000';
const Inference_API = process.env.REACT_APP_API_PORT || '8000';


export const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || `${API_PROTOCOL}://${API_HOST}:${API_PORT}`;

// Inference API base URL (for brain-age and normative endpoints)
export const INFERENCE_API_URL =
  process.env.REACT_APP_INFERENCE_API_URL || process.env.INFERENCE_API_URL || 'http://localhost:8000';


