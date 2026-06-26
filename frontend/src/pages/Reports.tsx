import React, { useEffect, useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { AppContext } from '../context/AppContext';
import { useToast } from '../components/Toast';
import { FiDownload, FiSearch, FiFileText, FiEye, FiClock, FiActivity } from 'react-icons/fi';

type ReportItem = {
  id: string;
  predictionLabel: string;
  confidenceScore: number;
  processingTime: number;
  modelVersion: string;
  createdAt: string;
};

export const Reports: React.FC = () => {
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const { dispatch } = useContext(AppContext);
  const navigate = useNavigate();
  const toast = useToast();

  const loadReports = async () => {
    try {
      setLoading(true);
      const data = await api.fetchPredictions();
      // Filter for completed runs which have reports
      const completedRuns = data.filter((item: any) => item.status === 'completed');
      setReports(completedRuns);
    } catch (err) {
      toast.addToast('Failed to load completed reports', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadReports();
  }, []);

  const handleViewInDashboard = (report: ReportItem) => {
    dispatch({
      type: 'SET_ACTIVE_JOB',
      payload: { id: report.id, status: 'completed' }
    });
    dispatch({
      type: 'SET_PIPELINE_STEP',
      payload: 7
    });
    dispatch({
      type: 'SET_PREDICTION_RESULT',
      payload: {
        id: report.id,
        diagnosis: report.predictionLabel,
        confidence: report.confidenceScore,
        model: report.modelVersion,
        processingTime: `${report.processingTime} sec`
      }
    });
    navigate('/');
  };

  // Filter reports
  const filteredReports = reports.filter((r) => {
    return r.id.toLowerCase().includes(search.toLowerCase()) || 
           r.predictionLabel.toLowerCase().includes(search.toLowerCase()) ||
           r.modelVersion.toLowerCase().includes(search.toLowerCase());
  });

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full border-4 border-border border-t-indigo-500 h-10 w-10" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Clinical Diagnostic Reports</h1>
          <p className="text-sm text-secondary">Access and download technical diagnosis files for completed screening pipelines.</p>
        </div>
        <button onClick={loadReports} className="btn-outline flex items-center gap-2">
          <FiActivity className="animate-pulse" />
          Refresh
        </button>
      </div>

      <div className="premium-card flex items-center p-4">
        <div className="relative w-full md:w-80">
          <span className="absolute inset-y-0 left-3 flex items-center text-muted">
            <FiSearch size={16} />
          </span>
          <input
            type="text"
            className="input-field w-full pl-10"
            placeholder="Search reports by ID, label, model..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {filteredReports.length === 0 ? (
        <div className="premium-card text-center py-16 text-muted">
          <FiFileText size={48} className="mx-auto mb-4 text-muted opacity-50" />
          <p className="text-lg font-medium">No completed reports found</p>
          <p className="text-sm">Process a video stream on the Dashboard to generate diagnostic reports.</p>
        </div>
      ) : (
        <div 
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
            gap: '16px'
          }}
        >
          {filteredReports.map((report) => {
            const isASD = report.predictionLabel === 'ASD';
            return (
              <div 
                key={report.id} 
                className="premium-card flex flex-col justify-between"
                style={{
                  padding: 'calc(var(--space-unit) * 4)',
                  textAlign: 'left'
                }}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--color-border)', paddingBottom: '12px', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <FiFileText size={18} className="text-indigo-500" />
                      <span className="font-mono font-medium text-xs text-secondary">
                        ID: {report.id.slice(0, 8)}...
                      </span>
                    </div>
                    <span 
                      style={{ 
                        fontSize: '0.725rem',
                        fontWeight: 600,
                        padding: '2px 8px',
                        borderRadius: 'var(--radius-sm)',
                        backgroundColor: isASD ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                        color: isASD ? 'var(--color-error)' : 'var(--color-success)',
                        border: `1px solid ${isASD ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)'}`
                      }}
                    >
                      {report.predictionLabel} ({report.confidenceScore.toFixed(1)}%)
                    </span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.825rem', color: 'var(--color-text-secondary)' }}>
                      <FiClock size={14} className="text-muted" />
                      <span>Executed: {report.createdAt}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.825rem', color: 'var(--color-text-secondary)' }}>
                      <FiActivity size={14} className="text-muted" />
                      <span>Classifier: {report.modelVersion || 'Bi-LSTM v2.1'}</span>
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', borderTop: '1px solid var(--color-border)', paddingTop: '12px' }}>
                  <button 
                    onClick={() => handleViewInDashboard(report)}
                    className="btn-outline flex items-center gap-1.5"
                    style={{ padding: '6px 12px', fontSize: '0.825rem' }}
                  >
                    <FiEye size={14} />
                    <span>View Result</span>
                  </button>
                  <a 
                    href={`/reports/${report.id}/download/json`}
                    className="btn-indigo flex items-center gap-1.5"
                    style={{ padding: '6px 12px', fontSize: '0.825rem', textDecoration: 'none', marginLeft: 'auto' }}
                    download
                  >
                    <FiDownload size={14} />
                    <span>JSON</span>
                  </a>
                  <a 
                    href={`/reports/${report.id}/download/csv`}
                    className="btn-outline flex items-center gap-1.5"
                    style={{ padding: '6px 12px', fontSize: '0.825rem', textDecoration: 'none' }}
                    download
                  >
                    <FiDownload size={14} />
                    <span>CSV</span>
                  </a>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default Reports;
