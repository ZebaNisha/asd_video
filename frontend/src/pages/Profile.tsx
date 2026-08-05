import React, { useEffect, useState } from 'react';
import { api } from '../api';
import { useToast } from '../components/Toast';
import { FiMail, FiDownload, FiSearch, FiBriefcase, FiDatabase, FiFileText } from 'react-icons/fi';

type ReportItem = {
  id: string;
  predictionLabel: string;
  confidenceScore: number;
  processingTime: number;
  modelVersion: string;
  createdAt: string;
};

type ProfileData = {
  username: string;
  email: string;
  role: string;
  department: string;
  hospital: string;
  clinicalId: string;
};

const Profile: React.FC = () => {
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const toast = useToast();

  const loadData = async () => {
    try {
      setLoading(true);
      // Fetch profile info (or fallback to dummy)
      let profileInfo: ProfileData;
      try {
        profileInfo = await api.get('/profile');
      } catch (err) {
        // Fallback in case of network issue
        profileInfo = {
          username: 'doctor',
          email: 'doctor@hospital.org',
          role: 'Lead Clinical Specialist',
          department: 'Pediatric Neurology',
          hospital: 'Children\'s Health Medical Center',
          clinicalId: 'CLINIC-9428-ASD',
        };
      }
      setProfile(profileInfo);

      // Fetch completed reports
      const predictionData = await api.fetchPredictions();
      const completedRuns = predictionData.filter((item: any) => item.status === 'completed');
      setReports(completedRuns);
    } catch (err) {
      toast.addToast('Failed to load profile details', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

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
    <div className="space-y-6" style={{ textAlign: 'left' }}>
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Clinician Profile</h1>
        <p className="text-sm text-secondary">Manage your user profile details and access your generated diagnostic reports.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px' }} className="grid-cols-1 md:grid-cols-3">
        {/* Clinician Card */}
        <div className="premium-card flex flex-col items-center text-center p-6 gap-4">
          <div 
            style={{
              height: '96px',
              width: '96px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 8px 24px var(--color-primary-glow)',
              color: '#ffffff',
              fontSize: '2.5rem',
              fontWeight: 'bold',
            }}
          >
            {profile?.username.slice(0, 2).toUpperCase() || 'DR'}
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Dr. {profile?.username || 'Clinician'}</h2>
            <p className="text-xs text-secondary font-mono">{profile?.clinicalId}</p>
          </div>

          <div style={{ width: '100%', borderTop: '1px solid var(--color-border)', paddingTop: '16px' }} className="space-y-3 text-left">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.825rem' }}>
              <FiBriefcase className="text-indigo-500" />
              <div>
                <p style={{ fontWeight: 500, margin: 0 }}>Role & Dept</p>
                <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.75rem', margin: 0 }}>
                  {profile?.role} - {profile?.department}
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.825rem' }}>
              <FiMail className="text-indigo-500" />
              <div>
                <p style={{ fontWeight: 500, margin: 0 }}>Email Address</p>
                <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.75rem', margin: 0 }}>
                  {profile?.email}
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.825rem' }}>
              <FiDatabase className="text-indigo-500" />
              <div>
                <p style={{ fontWeight: 500, margin: 0 }}>Affiliated Hospital</p>
                <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.75rem', margin: 0 }}>
                  {profile?.hospital}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Reports Download list */}
        <div className="premium-card flex flex-col p-6 gap-4">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>My Diagnostic Reports</h3>
            <div className="relative w-full sm:w-64">
              <span className="absolute inset-y-0 left-3 flex items-center text-muted">
                <FiSearch size={15} />
              </span>
              <input
                type="text"
                className="input-field w-full pl-10 text-xs py-1.5"
                placeholder="Search reports..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </div>

          <div style={{ overflowX: 'auto', marginTop: '8px' }}>
            {filteredReports.length === 0 ? (
              <div className="text-center py-12 text-muted">
                <FiFileText size={36} className="mx-auto mb-2 opacity-50" />
                <p style={{ fontSize: '0.9rem' }}>No completed reports found</p>
              </div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.825rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--color-border)', color: 'var(--color-text-secondary)' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Report ID</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Date</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Result</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Model</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredReports.map((report) => {
                    const isASD = report.predictionLabel === 'ASD';
                    const dateStr = new Date(report.createdAt).toLocaleDateString(undefined, {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric'
                    });
                    return (
                      <tr 
                        key={report.id} 
                        style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}
                        className="hover:bg-white/5"
                      >
                        <td style={{ padding: '12px 12px', fontFamily: 'monospace' }}>
                          {report.id.slice(0, 8)}...
                        </td>
                        <td style={{ padding: '12px 12px' }}>{dateStr}</td>
                        <td style={{ padding: '12px 12px' }}>
                          <span 
                            style={{ 
                              fontSize: '0.7rem',
                              fontWeight: 600,
                              padding: '2px 6px',
                              borderRadius: '4px',
                              backgroundColor: isASD ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                              color: isASD ? 'var(--color-error)' : 'var(--color-success)',
                              border: `1px solid ${isASD ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)'}`
                            }}
                          >
                            {report.predictionLabel} ({report.confidenceScore.toFixed(1)}%)
                          </span>
                        </td>
                        <td style={{ padding: '12px 12px', color: 'var(--color-text-secondary)' }}>
                          {report.modelVersion || 'Bi-LSTM v2.1'}
                        </td>
                        <td style={{ padding: '12px 12px', textAlign: 'right' }}>
                          <div style={{ display: 'flex', gap: '8px', justifyContent: 'end' }}>
                            <a 
                              href={`/reports/${report.id}/download/json`}
                              className="btn-indigo flex items-center gap-1"
                              style={{ padding: '4px 8px', fontSize: '0.75rem', textDecoration: 'none' }}
                              download
                              title="Download JSON Report"
                            >
                              <FiDownload size={12} />
                              <span>JSON</span>
                            </a>
                            <a 
                              href={`/reports/${report.id}/download/csv`}
                              className="btn-outline flex items-center gap-1"
                              style={{ padding: '4px 8px', fontSize: '0.75rem', textDecoration: 'none' }}
                              download
                              title="Download CSV Report"
                            >
                              <FiDownload size={12} />
                              <span>CSV</span>
                            </a>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
