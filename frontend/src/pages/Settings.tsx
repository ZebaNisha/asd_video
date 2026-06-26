import React, { useContext, useState, useEffect } from 'react';
import { AppContext } from '../context/AppContext';
import { useToast } from '../components/Toast';
import { FiSettings, FiSliders, FiServer, FiSave, FiSun, FiMoon } from 'react-icons/fi';
import { api } from '../api';

const Settings: React.FC = () => {
  const { state, dispatch } = useContext(AppContext);
  const toast = useToast();
  
  const [modelType, setModelType] = useState('vgg16_bilstm_v2');
  const [hiddenDim, setHiddenDim] = useState(128);
  const [dropout, setDropout] = useState(0.3);
  const [serverEndpoint, setServerEndpoint] = useState('/api');

  useEffect(() => {
    const loadSettings = async () => {
      try {
        const settings = await api.fetchSettings();
        setModelType(settings.modelType);
        setHiddenDim(settings.hiddenDim);
        setDropout(settings.dropout);
        setServerEndpoint(settings.serverEndpoint);
      } catch (e: any) {
        console.error(e);
        toast.addToast('Failed to load settings', 'error');
      }
    };
    loadSettings();
  }, []);

  const handleSave = async () => {
    try {
      await api.updateSettings({
        modelType,
        hiddenDim,
        dropout,
        serverEndpoint,
      });
      toast.addToast('System configuration saved successfully', 'success');
    } catch (e: any) {
      console.error(e);
      toast.addToast('Failed to save settings', 'error');
    }
  };

  return (
    <div className="space-y-6" style={{ textAlign: 'left' }}>
      <div>
        <h1 className="text-2xl font-bold tracking-tight">System Settings</h1>
        <p className="text-sm text-secondary">Configure clinical diagnostic thresholds, model weights, and connection endpoints.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }} className="grid-cols-2">
        {/* Model Configurations */}
        <div className="premium-card flex flex-col gap-4">
          <h3 style={{ fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '8px', borderBottom: '1px solid var(--color-border)', paddingBottom: '12px' }}>
            <FiSliders className="text-primary" />
            Neural Model Parameters
          </h3>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <label style={{ fontSize: '0.825rem', fontWeight: 500, color: 'var(--color-text-secondary)' }}>
                Active Classifier Weights
              </label>
              <select 
                className="input-field"
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                style={{ cursor: 'pointer' }}
              >
                <option value="vgg16_bilstm_v2">Bidirectional LSTM v2.1 (VGG16 visual features)</option>
                <option value="vgg16_lstm_v1">Standard LSTM v1.0 (VGG16 visual features)</option>
                <option value="openpose_gcn_legacy">OpenPose GCN v0.5 (Scale-invariant, Deprecated)</option>
              </select>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <label style={{ fontSize: '0.825rem', fontWeight: 500, color: 'var(--color-text-secondary)' }}>
                  LSTM Hidden Dimension
                </label>
                <input 
                  type="number" 
                  className="input-field" 
                  value={hiddenDim}
                  onChange={(e) => setHiddenDim(parseInt(e.target.value) || 0)}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <label style={{ fontSize: '0.825rem', fontWeight: 500, color: 'var(--color-text-secondary)' }}>
                  Dropout Coeff.
                </label>
                <input 
                  type="number" 
                  step="0.1" 
                  className="input-field" 
                  value={dropout}
                  onChange={(e) => setDropout(parseFloat(e.target.value) || 0)}
                />
              </div>
            </div>

            <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
              Note: Hidden dimensions and dropout variables are dynamically loaded at inference runtime and must match configurations trained in training script.
            </p>
          </div>
        </div>

        {/* Server & Preferences */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Server Config */}
          <div className="premium-card flex flex-col gap-4">
            <h3 style={{ fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '8px', borderBottom: '1px solid var(--color-border)', paddingBottom: '12px' }}>
              <FiServer className="text-cyan-500" />
              API Gateway Endpoints
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <label style={{ fontSize: '0.825rem', fontWeight: 500, color: 'var(--color-text-secondary)' }}>
                  Flask Backend Root Path
                </label>
                <input 
                  type="text" 
                  className="input-field" 
                  value={serverEndpoint}
                  onChange={(e) => setServerEndpoint(e.target.value)}
                />
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.825rem', marginTop: '4px' }}>
                <span style={{ color: 'var(--color-text-secondary)' }}>Connection Status:</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--color-success)', fontWeight: 600 }}>
                  <span style={{ height: '8px', width: '8px', borderRadius: '50%', backgroundColor: 'var(--color-success)', display: 'inline-block' }} />
                  Online
                </span>
              </div>
            </div>
          </div>

          {/* Theme Settings */}
          <div className="premium-card flex flex-col gap-4">
            <h3 style={{ fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '8px', borderBottom: '1px solid var(--color-border)', paddingBottom: '12px' }}>
              <FiSettings className="text-indigo-500" />
              Display Preferences
            </h3>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.875rem' }}>
              <div>
                <p style={{ fontWeight: 500 }}>Global App Interface Theme</p>
                <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.75rem' }}>Toggle dark mode priority styling</p>
              </div>
              <button 
                onClick={() => dispatch({ type: 'TOGGLE_DARK_MODE' })}
                className="btn-outline flex items-center gap-2"
                style={{ padding: '6px 12px' }}
              >
                {state.darkMode ? <FiSun /> : <FiMoon />}
                <span>{state.darkMode ? 'Light' : 'Dark'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'end', marginTop: '24px' }}>
        <button 
          onClick={handleSave}
          className="btn-indigo flex items-center gap-2"
          style={{ padding: '10px 24px' }}
        >
          <FiSave size={16} />
          Save Configurations
        </button>
      </div>
    </div>
  );
};

export default Settings;
