import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FiVideo, FiUploadCloud, FiCpu, FiFileText, FiCheckCircle, FiArrowRight } from 'react-icons/fi';

const Instructions: React.FC = () => {
  const navigate = useNavigate();

  const handleContinue = () => {
    navigate('/');
  };

  const steps = [
    {
      icon: <FiVideo size={24} className="text-indigo-400" />,
      title: '1. Prepare Video Recording',
      description: 'Record a short video of the child in a well-lit environment. Ensure the child is fully in the camera frame, standing or moving naturally without obstructions.',
      tips: ['Good lighting', 'No other people in frame', 'Clear view of joints']
    },
    {
      icon: <FiUploadCloud size={24} className="text-cyan-400" />,
      title: '2. Upload & Transmit',
      description: 'Upload the video file on the dashboard. We support MP4, AVI, and MOV formats up to 500MB. The transmission is fully encrypted and HIPAA-compliant.',
      tips: ['Max 500MB size', 'MP4 / AVI / MOV', 'Secure transmission']
    },
    {
      icon: <FiCpu size={24} className="text-emerald-400" />,
      title: '3. Neural AI Pipeline',
      description: 'The system automatically extracts 2D skeletal coordinates, filters out adult joints, maps frames using VGG16 visual encoders, and evaluates sequences in a Bi-LSTM network.',
      tips: ['Skeletal joint tracking', 'Child bounding box filter', 'Bi-LSTM classification']
    },
    {
      icon: <FiFileText size={24} className="text-rose-400" />,
      title: '4. Clinical Review',
      description: 'Review the generated diagnostic reports, including probability scores, confidence metrics, and complete sequence logs. You can export results as JSON or CSV.',
      tips: ['Confidence percentages', 'Probability distributions', 'JSON / CSV export']
    }
  ];

  return (
    <div className="space-y-6" style={{ textAlign: 'left' }}>
      {/* Page Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight">Parent & Clinician Onboarding</h1>
        <p className="text-sm text-secondary">
          Welcome to the ASD Detector. Follow this clinical protocol to run objective deep learning screenings for autism spectrum indicators.
        </p>
      </div>

      {/* Grid of Step Cards */}
      <div 
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
          gap: '20px',
          marginTop: '8px'
        }}
        className="grid-cols-2"
      >
        {steps.map((step, idx) => (
          <div 
            key={idx} 
            className="premium-card flex flex-col justify-between"
            style={{
              padding: '24px',
              border: '1px solid var(--color-border)',
              backgroundColor: 'var(--color-surface)',
              borderRadius: 'var(--radius-lg)'
            }}
          >
            <div>
              {/* Card Header with Icon */}
              <div 
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  marginBottom: '16px'
                }}
              >
                <div 
                  style={{
                    height: '46px',
                    width: '46px',
                    borderRadius: '12px',
                    backgroundColor: 'var(--color-surface-hover)',
                    border: '1px solid var(--color-border)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  {step.icon}
                </div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                  {step.title}
                </h3>
              </div>

              {/* Card Description */}
              <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.85rem', lineHeight: '1.5', marginBottom: '16px' }}>
                {step.description}
              </p>
            </div>

            {/* Quick Guidelines / Bullets */}
            <div 
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
                paddingTop: '12px',
                borderTop: '1px solid var(--color-border)'
              }}
            >
              {step.tips.map((tip, tIdx) => (
                <div 
                  key={tIdx} 
                  style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '6px', 
                    fontSize: '0.75rem', 
                    color: 'var(--color-text-muted)' 
                  }}
                >
                  <FiCheckCircle size={12} className="text-emerald-500" />
                  <span>{tip}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Onboarding Confirmation & Action */}
      <div 
        className="premium-card flex flex-col md:flex-row items-center justify-between gap-6"
        style={{
          marginTop: '8px',
          padding: '24px',
          background: 'linear-gradient(135deg, rgba(99,102,241,0.05) 0%, rgba(6,182,212,0.05) 100%)',
          border: '1px solid rgba(99,102,241,0.15)',
        }}
      >
        <div style={{ flex: 1 }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: '4px' }}>
            Ready to perform screening?
          </h3>
          <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.825rem', maxWidth: '650px' }}>
            Ensure you have read and understood the checklist above before proceeding. If you have any clinical doubts, consult the technical guides in the reports tab.
          </p>
        </div>

        <button 
          onClick={handleContinue} 
          className="btn-primary"
          style={{ padding: '12px 28px', whiteSpace: 'nowrap' }}
        >
          <span>Continue to Dashboard</span>
          <FiArrowRight size={16} />
        </button>
      </div>
    </div>
  );
};

export default Instructions;
