import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './config.js';
import {
  Eye,
  Download,
  FileText,
  Calendar,
  User,
  Activity,
  ChevronRight,
  ChevronDown,
  Image as ImageIcon,
  AlertCircle,
  RefreshCw
} from 'lucide-react';

const DicomViewer = ({ folder, onClose }) => {
  const [dicomData, setDicomData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedSeries, setExpandedSeries] = useState(new Set());
  const [previewImages, setPreviewImages] = useState({});
  const [loadingPreviews, setLoadingPreviews] = useState(new Set());

  useEffect(() => {
    if (folder) {
      fetchDicomStudies();
    }
  }, [folder]);

  const fetchDicomStudies = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE_URL}/folders/${encodeURIComponent(folder)}/dicom-studies`);
      setDicomData(response.data);
    } catch (err) {
      console.error('Error fetching DICOM studies:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to fetch DICOM studies');
    } finally {
      setLoading(false);
    }
  };

  const downloadDicomFile = async (instanceId, filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/orthanc/instances/${instanceId}/file`, {
        responseType: 'blob'
      });
      
      // Create a blob URL and trigger download
      const blob = new Blob([response.data], { type: 'application/dicom' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename || `dicom-${instanceId}.dcm`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error downloading DICOM file:', err);
      alert('Failed to download DICOM file: ' + (err.response?.data?.detail || err.message));
    }
  };

  const loadPreviewImage = async (instanceId) => {
    if (previewImages[instanceId] || loadingPreviews.has(instanceId)) {
      return;
    }

    setLoadingPreviews(prev => new Set([...prev, instanceId]));
    
    try {
      const response = await axios.get(`${API_BASE_URL}/orthanc/instances/${instanceId}/preview`, {
        responseType: 'blob'
      });
      
      const imageUrl = window.URL.createObjectURL(response.data);
      setPreviewImages(prev => ({ ...prev, [instanceId]: imageUrl }));
    } catch (err) {
      console.error('Error loading preview:', err);
      setPreviewImages(prev => ({ ...prev, [instanceId]: null }));
    } finally {
      setLoadingPreviews(prev => {
        const next = new Set(prev);
        next.delete(instanceId);
        return next;
      });
    }
  };

  const toggleSeries = (seriesId) => {
    const newExpanded = new Set(expandedSeries);
    if (newExpanded.has(seriesId)) {
      newExpanded.delete(seriesId);
    } else {
      newExpanded.add(seriesId);
    }
    setExpandedSeries(newExpanded);
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    // DICOM dates are in format YYYYMMDD
    if (dateStr.length === 8) {
      const year = dateStr.substring(0, 4);
      const month = dateStr.substring(4, 6);
      const day = dateStr.substring(6, 8);
      return `${year}-${month}-${day}`;
    }
    return dateStr;
  };

  const formatTime = (timeStr) => {
    if (!timeStr) return 'N/A';
    // DICOM times are in format HHMMSS.ffffff
    const timePart = timeStr.split('.')[0];
    if (timePart.length >= 6) {
      const hour = timePart.substring(0, 2);
      const minute = timePart.substring(2, 4);
      const second = timePart.substring(4, 6);
      return `${hour}:${minute}:${second}`;
    }
    return timeStr;
  };

  if (loading) {
    return (
      <div className="dicom-viewer-overlay fixed inset-0 bg-black/80 flex items-center justify-center z-50">
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 max-w-md">
          <div className="flex items-center gap-3 text-gray-300">
            <RefreshCw className="w-6 h-6 animate-spin" />
            <span>Loading DICOM studies...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dicom-viewer-overlay fixed inset-0 bg-black/80 flex items-center justify-center z-50">
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 max-w-md">
          <div className="flex items-center gap-3 text-red-400 mb-4">
            <AlertCircle className="w-6 h-6" />
            <span className="font-semibold">Error Loading DICOM Data</span>
          </div>
          <p className="text-gray-300 mb-4">{error}</p>
          <div className="flex gap-2">
            <button
              onClick={fetchDicomStudies}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Retry
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  const { dicom_instances, orthanc_available } = dicomData || {};

  if (!dicom_instances || dicom_instances.length === 0) {
    return (
      <div className="dicom-viewer-overlay fixed inset-0 bg-black/80 flex items-center justify-center z-50">
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 max-w-md">
          <div className="text-center">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No DICOM Files</h3>
            <p className="text-gray-300 mb-4">This folder doesn't contain any DICOM files.</p>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dicom-viewer-overlay fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg border border-gray-700 max-w-6xl max-h-[90vh] w-full mx-4 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div>
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <FileText className="w-6 h-6" />
              DICOM Files - {folder}
            </h2>
            <p className="text-gray-400 text-sm mt-1">
              {dicom_instances.length} DICOM file{dicom_instances.length !== 1 ? 's' : ''} found
              {!orthanc_available && ' (Orthanc unavailable - limited functionality)'}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          <div className="space-y-4">
            {dicom_instances.map((instance, index) => (
              <div key={instance.instance_id || index} className="bg-gray-750 rounded-lg border border-gray-600 p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-3">
                      <ImageIcon className="w-5 h-5 text-blue-400" />
                      <h3 className="font-semibold text-white">{instance.filename}</h3>
                      <span className={`px-2 py-1 rounded text-xs ${
                        instance.current_status === 'available' ? 'bg-green-600 text-green-100' :
                        instance.current_status === 'not_found' ? 'bg-red-600 text-red-100' :
                        instance.current_status === 'error' ? 'bg-yellow-600 text-yellow-100' :
                        'bg-gray-600 text-gray-100'
                      }`}>
                        {instance.current_status || 'unknown'}
                      </span>
                    </div>

                    {/* DICOM Info Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      {instance.dicom_info && (
                        <>
                          <div className="flex items-center gap-2">
                            <User className="w-4 h-4 text-gray-400" />
                            <span className="text-gray-400">Patient:</span>
                            <span className="text-gray-200">{instance.dicom_info.patient_name || 'N/A'}</span>
                          </div>
                          
                          <div className="flex items-center gap-2">
                            <Calendar className="w-4 h-4 text-gray-400" />
                            <span className="text-gray-400">Study Date:</span>
                            <span className="text-gray-200">{formatDate(instance.dicom_info.study_date)}</span>
                          </div>

                          <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-gray-400" />
                            <span className="text-gray-400">Modality:</span>
                            <span className="text-gray-200">{instance.dicom_info.modality || 'N/A'}</span>
                          </div>

                          <div className="flex items-center gap-2">
                            <FileText className="w-4 h-4 text-gray-400" />
                            <span className="text-gray-400">Series:</span>
                            <span className="text-gray-200">{instance.dicom_info.series_description || 'N/A'}</span>
                          </div>
                        </>
                      )}
                    </div>

                    {/* Additional Tags if available */}
                    {instance.tags && Object.keys(instance.tags).length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-600">
                        <h4 className="text-sm font-semibold text-gray-300 mb-2">DICOM Tags:</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                          {Object.entries(instance.tags).map(([key, value]) => (
                            <div key={key} className="flex">
                              <span className="text-gray-400 w-24 truncate">{key}:</span>
                              <span className="text-gray-200 flex-1">{value || 'N/A'}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Actions and Preview */}
                  <div className="ml-4 flex flex-col items-end gap-2">
                    {instance.instance_id && orthanc_available && (
                      <>
                        <button
                          onClick={() => loadPreviewImage(instance.instance_id)}
                          disabled={loadingPreviews.has(instance.instance_id)}
                          className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
                        >
                          <Eye className="w-4 h-4" />
                          {loadingPreviews.has(instance.instance_id) ? 'Loading...' : 'Preview'}
                        </button>
                        
                        <button
                          onClick={() => downloadDicomFile(instance.instance_id, instance.filename)}
                          className="flex items-center gap-2 px-3 py-1.5 bg-green-600 text-white rounded text-sm hover:bg-green-700"
                        >
                          <Download className="w-4 h-4" />
                          Download
                        </button>
                      </>
                    )}

                    {/* Preview Image */}
                    {previewImages[instance.instance_id] && (
                      <div className="mt-2 border border-gray-600 rounded">
                        <img
                          src={previewImages[instance.instance_id]}
                          alt="DICOM Preview"
                          className="max-w-32 max-h-32 rounded"
                        />
                      </div>
                    )}
                    
                    {previewImages[instance.instance_id] === null && (
                      <div className="mt-2 px-2 py-1 bg-gray-700 rounded text-xs text-gray-400">
                        Preview not available
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 bg-gray-750 rounded-b-lg">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <span>Folder: {folder}</span>
            <span>
              {dicom_instances.length} DICOM file{dicom_instances.length !== 1 ? 's' : ''}
              {orthanc_available ? ' • Orthanc connected' : ' • Orthanc unavailable'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DicomViewer;