import React, { useState } from 'react';
import { FiDownload, FiRefreshCw, FiChevronDown, FiChevronUp, FiAlertTriangle, FiActivity, FiLayers } from 'react-icons/fi';

interface ResultCardProps {
  jobId: string;
  diagnosis: string;
  confidence: number; // e.g. 91.3
  model: string;
  processingTime: string; // e.g. '13.2 sec'
  asdProbability?: number;
  tdProbability?: number;
  rawClassification?: string;
  onRunAgain: () => void;
}

export const ResultCard: React.FC<ResultCardProps> = ({
  jobId,
  diagnosis,
  confidence,
  model,
  processingTime,
  asdProbability,
  tdProbability,
  rawClassification,
  onRunAgain,
}) => {
  const [detailsExpanded, setDetailsExpanded] = useState(false);

  // Normalize confidence (convert 0.913 to 91.3 if needed)
  const displayConfidence = confidence <= 1 ? confidence * 100 : confidence;
  const isASD = diagnosis.toUpperCase().includes('ASD') || diagnosis.toUpperCase() === 'AUTISM SPECTRUM DISORDER';

  // Calculate SVG stroke offset for gauge
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (Math.min(100, Math.max(0, displayConfidence)) / 100) * circumference;

  // Gauge colors
  const gaugeColor = isASD ? 'var(--color-error)' : 'var(--color-success)';

  return (
    <div className="premium-card flex flex-col gap-6" style={{ maxWidth: '700px', margin: '0 auto', textAlign: 'left' }}>
      {/* Header banner */}
      <div 
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid var(--color-border)',
          paddingBottom: '16px',
        }}
      >
        <div>
          <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FiAlertTriangle style={{ color: gaugeColor }} />
            Diagnostic Analysis Report
          </h2>
          <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.875rem' }}>
            Verification complete. Diagnostic report ready for review.
          </p>
        </div>
        <span 
          style={{
            fontSize: '0.75rem',
            padding: '4px 10px',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'var(--color-surface-hover)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-text-muted)',
            fontFamily: 'monospace'
          }}
        >
          REF: #{jobId.slice(0, 8)}
        </span>
      </div>

      {/* Main Results Dashboard Grid */}
      <div 
        style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 1.5fr',
          gap: '24px',
          alignItems: 'center'
        }}
        className="grid-cols-2"
      >
        {/* Confidence Circle Gauge */}
        <div className="radial-gauge-container" style={{ margin: '0 auto' }}>
          <svg className="radial-gauge-svg" viewBox="0 0 140 140">
            <circle className="radial-gauge-bg" cx="70" cy="70" r={radius} />
            <circle 
              className="radial-gauge-fill" 
              cx="70" 
              cy="70" 
              r={radius} 
              style={{
                stroke: gaugeColor,
                strokeDasharray: circumference,
                strokeDashoffset: strokeDashoffset,
                filter: `drop-shadow(0 0 4px ${gaugeColor})`
              }}
            />
          </svg>
          <div className="radial-gauge-value">
            <span style={{ fontSize: '1.85rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
              {displayConfidence.toFixed(1)}%
            </span>
            <p style={{ fontSize: '0.675rem', color: 'var(--color-text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Confidence
            </p>
          </div>
        </div>

        {/* Diagnostic Metadata */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
              Prediction Outcome
            </span>
            <h1 
              style={{ 
                fontSize: '1.65rem', 
                fontWeight: 700, 
                color: isASD ? 'var(--color-error)' : 'var(--color-success)',
                lineHeight: 1.2,
                marginTop: '4px'
              }}
            >
              {isASD ? 'Autism Spectrum Disorder (ASD)' : 'Typical Development (TD)'}
            </h1>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>Model Architecture</span>
              <p style={{ fontSize: '0.925rem', fontWeight: 500, marginTop: '2px' }}>
                {model || 'Bi-LSTM v2.1'}
              </p>
            </div>
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>Duration</span>
              <p style={{ fontSize: '0.925rem', fontWeight: 500, marginTop: '2px' }}>
                {processingTime || '13.2 sec'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Accordion panel for Technical details */}
      <div 
        style={{
          border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          backgroundColor: 'var(--color-background)'
        }}
      >
        <button
          onClick={() => setDetailsExpanded(!detailsExpanded)}
          style={{
            width: '100%',
            backgroundColor: 'var(--color-surface-hover)',
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            color: 'var(--color-text-primary)',
            fontSize: '0.875rem',
            fontWeight: 500,
            borderBottom: detailsExpanded ? '1px solid var(--color-border)' : 'none',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, textAlign: 'left' }}>
            <FiActivity size={16} className="text-primary" />
            <span>Technical Verification details</span>
          </div>
          {detailsExpanded ? <FiChevronUp size={16} /> : <FiChevronDown size={16} />}
        </button>

        {detailsExpanded && (
          <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '0.825rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }} className="grid-cols-2">
              <div style={{ padding: '8px 12px', borderRadius: 'var(--radius-sm)', backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)' }}>
                <span style={{ color: 'var(--color-text-muted)' }}>ASD Probability</span>
                <p style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--color-error)', marginTop: '2px' }}>
                  {asdProbability !== undefined ? `${(asdProbability * 100).toFixed(2)}%` : isASD ? `${displayConfidence.toFixed(2)}%` : `${(100 - displayConfidence).toFixed(2)}%`}
                </p>
              </div>
              <div style={{ padding: '8px 12px', borderRadius: 'var(--radius-sm)', backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)' }}>
                <span style={{ color: 'var(--color-text-muted)' }}>TD Probability</span>
                <p style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--color-success)', marginTop: '2px' }}>
                  {tdProbability !== undefined ? `${(tdProbability * 100).toFixed(2)}%` : !isASD ? `${displayConfidence.toFixed(2)}%` : `${(100 - displayConfidence).toFixed(2)}%`}
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', color: 'var(--color-text-secondary)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed var(--color-border)', paddingBottom: '4px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><FiLayers size={12} /> Convolutional Encoder</span>
                <span style={{ marginLeft: 'auto', fontFamily: 'monospace' }}>VGG16 (ImageNet-frozen, Layer fc2)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed var(--color-border)', paddingBottom: '4px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><FiActivity size={12} /> Recurrent Network</span>
                <span style={{ marginLeft: 'auto', fontFamily: 'monospace' }}>Bidirectional LSTM (128-dim recurrent layer)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed var(--color-border)', paddingBottom: '4px' }}>
                <span>Classification Label</span>
                <span style={{ marginLeft: 'auto', fontFamily: 'monospace' }}>{rawClassification || (isASD ? 'ASD_Positive' : 'TD_Negative')}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Feature Sequence Shape</span>
                <span style={{ marginLeft: 'auto', fontFamily: 'monospace' }}>120 frames × 4096 features (scale-invariant)</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Button controls */}
      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        <a 
          href={`/reports/${jobId}/download/json`}
          className="btn-indigo flex items-center gap-2"
          download
        >
          <FiDownload size={16} />
          <span>Export JSON Report</span>
        </a>
        <a 
          href={`/reports/${jobId}/download/csv`}
          className="btn-outline flex items-center gap-2"
          download
          style={{ textDecoration: 'none' }}
        >
          <FiDownload size={16} />
          <span>Export CSV Report</span>
        </a>
        <button 
          onClick={onRunAgain}
          className="btn-outline flex items-center gap-2"
          style={{ marginLeft: 'auto' }}
        >
          <FiRefreshCw size={14} />
          <span>Run Another Analysis</span>
        </button>
      </div>
    </div>
  );
};

export default ResultCard;
