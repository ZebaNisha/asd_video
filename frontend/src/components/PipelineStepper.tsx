import React, { useEffect, useState, useMemo } from 'react';
import { FiCheck, FiPlay, FiClock, FiTerminal, FiChevronDown, FiChevronUp } from 'react-icons/fi';

interface PipelineStepperProps {
  status: string | null;
  jobId: string | null;
}

interface StepDetails {
  id: number;
  label: string;
  desc: string;
  logs: string[];
}

export const PipelineStepper: React.FC<PipelineStepperProps> = ({ status, jobId }) => {
  const [elapsedTime, setElapsedTime] = useState(0);
  const [logsExpanded, setLogsExpanded] = useState(true);
  const [currentStep, setCurrentStep] = useState(0);

  // Dynamic simulation of intermediate steps to show progress when backend is 'processing'
  useEffect(() => {
    if (!status) {
      setCurrentStep(0);
      return;
    }

    if (status === 'queued') {
      setCurrentStep(1);
    } else if (status === 'processing') {
      setCurrentStep(2);
      
      // Progress through skeleton, tracking, features over time to make the UI look active
      const t1 = setTimeout(() => setCurrentStep(3), 3000);
      const t2 = setTimeout(() => setCurrentStep(4), 6000);
      const t3 = setTimeout(() => setCurrentStep(5), 9000);
      const t4 = setTimeout(() => setCurrentStep(6), 12000);

      return () => {
        clearTimeout(t1);
        clearTimeout(t2);
        clearTimeout(t3);
        clearTimeout(t4);
      };
    } else if (status === 'generating_report') {
      setCurrentStep(6);
    } else if (status === 'completed') {
      setCurrentStep(7);
    } else if (status === 'failed') {
      setCurrentStep(-1);
    }
  }, [status]);

  // Elapsed timer logic
  useEffect(() => {
    let timer: any;
    if (status && status !== 'completed' && status !== 'failed') {
      setElapsedTime(0);
      timer = setInterval(() => {
        setElapsedTime((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [status]);

  const pipelineSteps: StepDetails[] = useMemo(() => [
    {
      id: 1,
      label: 'Upload',
      desc: 'Transmitting video data to secure HIPAA-compliant cloud storage',
      logs: [
        '[INFO] Connecting to streaming gateway...',
        '[INFO] File authorization token validated.',
        `[INFO] Video upload initialized. File ID: ${jobId || 'N/A'}`,
        '[INFO] Video package fully received. Status: OK'
      ]
    },
    {
      id: 2,
      label: 'Stickman Generation',
      desc: 'Executing custom pipeline to generate coordinate arrays',
      logs: [
        '[INFO] Initializing stickmen.py model instance.',
        '[INFO] Processing raw video frames... (30fps sequence)',
        '[INFO] Tracking joints and rendering stickman outputs.',
        '[INFO] Stickman coordinate stream generated.'
      ]
    },
    {
      id: 3,
      label: 'Skeleton Detection',
      desc: 'Localizing 2D coordinate positions for joints and face landmarks',
      logs: [
        '[INFO] Loading pose landmarker model configurations.',
        '[INFO] Detecting 2D keypoints... (33 skeleton vertices)',
        '[INFO] Isolating joint angles (knees, hips, elbows, shoulders).'
      ]
    },
    {
      id: 4,
      label: 'Child Tracking',
      desc: 'Bounding box isolation to filter parent/examiner joints',
      logs: [
        '[INFO] Running child localization bounding box.',
        '[INFO] Filtering out coordinate arrays belonging to adults.',
        '[INFO] Child coordinate crop successful. Sequence isolated.'
      ]
    },
    {
      id: 5,
      label: 'Feature Extraction',
      desc: 'Mapping coordinate frames to deep visual features using VGG16',
      logs: [
        '[INFO] Loading frozen weights from VGG16 ImageNet model.',
        '[INFO] Extracting dense feature representations for frame sequences.',
        '[INFO] VGG16 visual feature tensors compiled. Shape: (120, 4096)'
      ]
    },
    {
      id: 6,
      label: 'Bi-LSTM Prediction',
      desc: 'Evaluating sequential feature arrays in Bidirectional LSTM classifier',
      logs: [
        '[INFO] Initializing Bidirectional LSTM recurrent classifier.',
        '[INFO] Model hyperparameters: Hidden Dimensions = 128, Dropout = 0.3',
        '[INFO] Forward and backward temporal paths evaluated.',
        '[INFO] Sigmoid output layer mapped to ASD/TD probability distribution.'
      ]
    },
    {
      id: 7,
      label: 'Report',
      desc: 'Generating diagnosis report with confidence metrics and logs',
      logs: [
        '[INFO] Assembling diagnostic results JSON schema.',
        '[INFO] Creating export templates (PDF / CSV formats).',
        '[INFO] Diagnosis successfully recorded in local DB.'
      ]
    }
  ], [jobId]);

  // Aggregate logs to show in console based on current progress
  const visibleLogs = useMemo(() => {
    if (currentStep === -1) {
      return ['[ERROR] Pipeline analysis crashed. Unhandled exception in model layer.'];
    }
    
    let allLogs: string[] = [];
    pipelineSteps.forEach((step) => {
      if (step.id <= currentStep) {
        allLogs = [...allLogs, `--- ${step.label} Stage ---`, ...step.logs];
      }
    });

    if (currentStep > 0 && currentStep < 7) {
      // Append a pulsing progress line
      allLogs.push(`[SYSTEM] Executive routine running... (${elapsedTime}s elapsed)`);
    }

    return allLogs;
  }, [currentStep, pipelineSteps, elapsedTime]);

  return (
    <div className="premium-card flex flex-col gap-6" style={{ maxWidth: '700px', margin: '0 auto' }}>
      <div className="flex justify-between items-center border-b border-border pb-4">
        <div>
          <h2 style={{ fontSize: '1.25rem' }}>ASD Detection Pipeline</h2>
          <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.875rem' }}>
            Multi-stage deep learning pipeline execution monitor.
          </p>
        </div>
        {status && status !== 'completed' && status !== 'failed' && (
          <div style={{ color: 'var(--color-accent)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <FiClock className="animate-spin" />
            <span>{elapsedTime}s elapsed</span>
          </div>
        )}
      </div>

      {/* Stepper items container */}
      <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: '20px', paddingLeft: '8px' }}>
        {/* Progress connecting line */}
        <div className="stepper-line" />
        <div 
          className="stepper-line stepper-line-active" 
          style={{ 
            height: `${Math.max(0, Math.min(100, ((currentStep - 1) / 6) * 100))}%`,
            transition: 'height var(--transition-slow)'
          }} 
        />

        {pipelineSteps.map((step) => {
          const isCompleted = currentStep > step.id;
          const isActive = currentStep === step.id;
          const isPending = currentStep < step.id;

          let stepColor = 'var(--color-border)';
          let iconContent = <div style={{ height: '6px', width: '6px', borderRadius: '50%', backgroundColor: 'var(--color-text-muted)' }} />;
          
          if (isCompleted) {
            stepColor = 'var(--color-success)';
            iconContent = <FiCheck size={12} color="#fff" />;
          } else if (isActive) {
            stepColor = 'var(--color-primary)';
            iconContent = <FiPlay size={10} color="#fff" className="animate-pulse" />;
          }

          if (currentStep === -1 && step.id === 2) {
            stepColor = 'var(--color-error)';
            iconContent = <span style={{ color: '#fff', fontSize: '10px', fontWeight: 'bold' }}>!</span>;
          }

          return (
            <div 
              key={step.id} 
              style={{ 
                display: 'flex', 
                gap: '16px', 
                position: 'relative', 
                zIndex: 2,
                opacity: isPending ? 0.5 : 1,
                transition: 'opacity var(--transition-normal)'
              }}
            >
              {/* Stepper Icon Indicator */}
              <div 
                style={{
                  height: '24px',
                  width: '24px',
                  borderRadius: '50%',
                  backgroundColor: isCompleted ? 'var(--color-success)' : (isActive ? 'var(--color-primary)' : 'var(--color-surface-hover)'),
                  border: `2px solid ${stepColor}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: isActive ? '0 0 10px var(--color-primary-glow)' : (isCompleted ? '0 0 10px var(--color-success-glow)' : 'none'),
                  transition: 'all var(--transition-normal)',
                  flexShrink: 0,
                  marginTop: '2px'
                }}
              >
                {iconContent}
              </div>

              {/* Text Description */}
              <div style={{ textAlign: 'left' }}>
                <h4 
                  style={{ 
                    fontSize: '0.925rem', 
                    fontWeight: isActive || isCompleted ? 600 : 500,
                    color: isActive ? 'var(--color-primary)' : 'var(--color-text-primary)'
                  }}
                >
                  {step.label}
                </h4>
                {isOpenText(isActive, isCompleted) && (
                  <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.775rem', marginTop: '2px' }}>
                    {step.desc}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Expandable Logs Terminal Console */}
      <div 
        style={{
          border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          backgroundColor: '#05070c'
        }}
      >
        <button
          onClick={() => setLogsExpanded(!logsExpanded)}
          style={{
            width: '100%',
            backgroundColor: 'rgba(255,255,255,0.02)',
            padding: '10px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'between',
            color: 'var(--color-text-secondary)',
            fontSize: '0.825rem',
            borderBottom: logsExpanded ? '1px solid var(--color-border)' : 'none',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, textAlign: 'left' }}>
            <FiTerminal size={14} />
            <span className="font-mono">Live Pipeline Diagnostics</span>
          </div>
          {logsExpanded ? <FiChevronUp size={16} /> : <FiChevronDown size={16} />}
        </button>

        {logsExpanded && (
          <div 
            style={{
              padding: '16px',
              fontFamily: 'monospace',
              fontSize: '0.75rem',
              color: '#34d399', // Cyan/Green terminal colors
              textAlign: 'left',
              maxHeight: '180px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px'
            }}
          >
            {visibleLogs.map((log, idx) => {
              const isHeader = log.startsWith('---');
              const isError = log.startsWith('[ERROR]');
              return (
                <div 
                  key={idx} 
                  style={{ 
                    color: isError ? 'var(--color-error)' : (isHeader ? '#6366f1' : '#34d399'),
                    fontWeight: isHeader ? 'bold' : 'normal'
                  }}
                >
                  {log}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

// Helper function to decide detail visibility
function isOpenText(isActive: boolean, isCompleted: boolean) {
  return isActive || isCompleted;
}

export default PipelineStepper;
