import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiUser, FiLock, FiAlertCircle, FiActivity } from 'react-icons/fi';

const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    if (!username || !password) {
      setError('Please enter both username and password.');
      return;
    }

    setLoading(true);

    // Simulate clinical database verification delay
    setTimeout(() => {
      setLoading(false);
      // Store dummy token in localStorage to simulate login
      localStorage.setItem('authToken', 'fake-token');
      navigate('/instructions');
    }, 800);
  };

  return (
    <div className="flex flex-col min-h-screen items-center justify-center bg-gradient-to-br from-indigo-900 to-slate-900 px-4">
      {/* Decorative Glowing Elements */}
      <div 
        style={{
          position: 'absolute',
          width: '500px',
          height: '500px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(0, 0, 0, 0) 70%)',
          top: '10%',
          left: '15%',
          pointerEvents: 'none',
          zIndex: 1,
        }}
      />
      <div 
        style={{
          position: 'absolute',
          width: '500px',
          height: '500px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(6, 182, 212, 0.12) 0%, rgba(0, 0, 0, 0) 70%)',
          bottom: '10%',
          right: '15%',
          pointerEvents: 'none',
          zIndex: 1,
        }}
      />

      {/* Main Login Card */}
      <div 
        className="glass-panel p-8 rounded-xl shadow-xl w-full animate-fadeIn"
        style={{ 
          maxWidth: '400px', 
          zIndex: 10,
          border: '1px solid var(--glass-border)',
          animation: 'fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards'
        }}
      >
        {/* Portal Branding */}
        <div className="text-center mb-6">
          <div 
            className="mx-auto flex items-center justify-center mb-3"
            style={{
              height: '54px',
              width: '54px',
              borderRadius: '14px',
              background: 'linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%)',
              boxShadow: '0 8px 24px var(--color-primary-glow)',
              color: '#ffffff',
            }}
          >
            <FiActivity size={26} className="animate-pulse" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">ASD Detector</h1>
          <p className="text-xs text-secondary mt-1">Clinical AI Screening Portal</p>
        </div>

        {/* Error Alert Box */}
        {error && (
          <div 
            className="flex items-center gap-2 p-3 rounded-lg mb-4 text-xs"
            style={{
              backgroundColor: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.2)',
              color: 'var(--color-error)'
            }}
          >
            <FiAlertCircle size={14} style={{ flexShrink: 0 }} />
            <span>{error}</span>
          </div>
        )}

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-secondary mb-1.5" htmlFor="username">
              Username or Clinical ID
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-3 flex items-center text-muted">
                <FiUser size={15} />
              </span>
              <input
                id="username"
                type="text"
                placeholder="clinician@hospital.org"
                className="input-field w-full pl-10 text-sm"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={loading}
                autoComplete="username"
                style={{
                  border: '1px solid var(--color-border)',
                  backgroundColor: 'rgba(0,0,0,0.2)',
                }}
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-secondary mb-1.5" htmlFor="password">
              Security Password
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-3 flex items-center text-muted">
                <FiLock size={15} />
              </span>
              <input
                id="password"
                type="password"
                placeholder="••••••••••••"
                className="input-field w-full pl-10 text-sm"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
                autoComplete="current-password"
                style={{
                  border: '1px solid var(--color-border)',
                  backgroundColor: 'rgba(0,0,0,0.2)',
                }}
              />
            </div>
          </div>

          <button 
            type="submit" 
            className="btn-primary w-full mt-4" 
            disabled={loading}
            style={{ height: '44px' }}
          >
            {loading ? (
              <div className="flex items-center gap-2">
                <span className="animate-spin rounded-full border-2 border-white border-t-transparent h-4 w-4" />
                <span>Verifying credentials...</span>
              </div>
            ) : (
              'Sign In to Portal'
            )}
          </button>
        </form>

        {/* Footer info */}
        <div className="text-center mt-6 pt-4 border-t border-border">
          <p style={{ fontSize: '0.675rem', color: 'var(--color-text-muted)' }}>
            Authorized Clinical Use Only. Data encrypted under HIPAA guidelines.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
