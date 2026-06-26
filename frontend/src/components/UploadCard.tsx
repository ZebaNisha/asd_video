import React, { useState, useContext } from 'react';
import { useDropzone } from 'react-dropzone';
import { uploadFile } from '../api';
import { AppContext } from '../context/AppContext';
import { useToast } from './Toast';
import { FiUploadCloud, FiFile, FiAlertCircle } from 'react-icons/fi';

export const UploadCard: React.FC = () => {
  const { dispatch } = useContext(AppContext);
  const [uploading, setUploading] = useState(false);
  const [dragError, setDragError] = useState<string | null>(null);
  const toast = useToast();

  const onDrop = async (acceptedFiles: File[], fileRejections: any[]) => {
    setDragError(null);

    if (fileRejections.length > 0) {
      const errorMsg = fileRejections[0].errors[0]?.message || 'Invalid file format or size.';
      setDragError(errorMsg);
      toast.addToast(errorMsg, 'error');
      return;
    }

    const file = acceptedFiles[0];
    if (!file) return;

    // Double check size limit (500MB)
    if (file.size > 500 * 1024 * 1024) {
      const sizeErr = 'File exceeds maximum limit of 500MB.';
      setDragError(sizeErr);
      toast.addToast(sizeErr, 'error');
      return;
    }

    setUploading(true);
    toast.addToast('Uploading video and initializing neural pipeline...', 'info');
    
    try {
      const result = await uploadFile(file);
      
      toast.addToast('Upload complete! Pipeline analysis started.', 'success');
      
      // Update global context with the active job info
      dispatch({
        type: 'SET_ACTIVE_JOB',
        payload: { id: result.id, status: result.status }
      });
      dispatch({ type: 'SET_PIPELINE_STEP', payload: 1 });
      
    } catch (err: any) {
      console.error(err);
      toast.addToast(err.message || 'Analysis initialization failed.', 'error');
    } finally {
      setUploading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      'video/mp4': ['.mp4'],
      'video/quicktime': ['.mov'],
      'video/x-msvideo': ['.avi'],
    },
    maxSize: 500 * 1024 * 1024,
  });

  return (
    <div 
      className="premium-card" 
      style={{
        maxWidth: '700px',
        margin: '0 auto',
        textAlign: 'center',
        padding: 'calc(var(--space-unit) * 6)',
      }}
    >
      <div
        {...getRootProps()}
        style={{
          border: '2px dashed var(--color-border)',
          borderRadius: 'var(--radius-lg)',
          padding: 'calc(var(--space-unit) * 8) calc(var(--space-unit) * 4)',
          cursor: uploading ? 'not-allowed' : 'pointer',
          backgroundColor: isDragActive ? 'var(--color-surface-hover)' : 'var(--color-background)',
          borderColor: isDragActive ? 'var(--color-primary)' : 'var(--color-border)',
          transition: 'all var(--transition-normal)',
          boxShadow: isDragActive ? '0 0 20px var(--color-primary-glow)' : 'none',
        }}
        className={isDragActive ? 'animate-glow' : ''}
      >
        <input {...getInputProps()} disabled={uploading} />
        
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
          <div 
            style={{
              height: '64px',
              width: '64px',
              borderRadius: '50%',
              backgroundColor: isDragActive ? 'var(--color-primary-glow)' : 'var(--color-surface-hover)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: isDragActive ? 'var(--color-primary)' : 'var(--color-text-secondary)',
              transition: 'all var(--transition-normal)',
            }}
          >
            <FiUploadCloud size={32} className={uploading ? 'animate-bounce' : ''} />
          </div>

          <div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '4px' }}>
              {uploading ? 'Processing Video File...' : 'Drag & Drop Video'}
            </h3>
            <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.875rem' }}>
              {uploading ? 'Deploying network model' : 'or click to browse your local files'}
            </p>
          </div>

          <div 
            style={{
              padding: '4px 12px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'var(--color-surface-hover)',
              border: '1px solid var(--color-border)',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.75rem',
              color: 'var(--color-text-muted)',
            }}
          >
            <FiFile size={12} />
            <span>MP4 • AVI • MOV • Max 500MB</span>
          </div>
        </div>
      </div>

      {dragError && (
        <div 
          style={{
            marginTop: '16px',
            color: 'var(--color-error)',
            fontSize: '0.875rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '6px',
          }}
        >
          <FiAlertCircle size={14} />
          <span>{dragError}</span>
        </div>
      )}

      {uploading && (
        <div style={{ marginTop: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
          <div className="animate-spin rounded-full border-2 border-border border-t-indigo-500 h-6 w-6" />
          <span style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
            Transmitting stream... Please do not close this browser window.
          </span>
        </div>
      )}
    </div>
  );
};

export default UploadCard;
