import React, { useEffect, useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppContext } from '../context/AppContext';
import { useToast } from '../components/Toast';
import { FiSearch, FiDownload, FiEye, FiFilter, FiActivity, FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { api } from '../api';

interface PredictionRecord {
  id: string;
  status: string;
  predictionLabel: string;
  confidenceScore: number;
  processingTime: number;
  modelVersion: string;
  createdAt: string;
}

const History: React.FC = () => {
  const [records, setRecords] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [labelFilter, setLabelFilter] = useState('all');
  const [sortField, setSortField] = useState<keyof PredictionRecord>('createdAt');
  const [sortAsc, setSortAsc] = useState(false);
  
  const { dispatch } = useContext(AppContext);
  const navigate = useNavigate();
  const toast = useToast();

  const fetchHistory = async () => {
  try {
    setLoading(true);
    const data = await api.fetchPredictions();
    setRecords(data);
  } catch (e: any) {
    console.error(e);
    toast.addToast('Failed to fetch prediction history', 'error');
  } finally {
    setLoading(false);
  }
};

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleViewInDashboard = (record: PredictionRecord) => {
    dispatch({
      type: 'SET_ACTIVE_JOB',
      payload: { id: record.id, status: record.status }
    });
    dispatch({
      type: 'SET_PIPELINE_STEP',
      payload: record.status === 'completed' ? 7 : (record.status === 'failed' ? -1 : 2)
    });
    
    if (record.status === 'completed') {
      dispatch({
        type: 'SET_PREDICTION_RESULT',
        payload: {
          id: record.id,
          diagnosis: record.predictionLabel,
          confidence: record.confidenceScore,
          model: record.modelVersion,
          processingTime: `${record.processingTime} sec`
        }
      });
    }
    
    navigate('/');
  };

  const handleSort = (field: keyof PredictionRecord) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  // Filter records
  const filteredRecords = records.filter((r) => {
    const matchesSearch = r.id.toLowerCase().includes(search.toLowerCase()) || 
                          r.predictionLabel.toLowerCase().includes(search.toLowerCase()) ||
                          r.modelVersion.toLowerCase().includes(search.toLowerCase());
    
    const matchesStatus = statusFilter === 'all' || r.status === statusFilter;
    const matchesLabel = labelFilter === 'all' || r.predictionLabel === labelFilter;

    return matchesSearch && matchesStatus && matchesLabel;
  });

  // Sort records
  const sortedRecords = [...filteredRecords].sort((a, b) => {
    let valA = a[sortField];
    let valB = b[sortField];

    if (typeof valA === 'string') {
      valA = (valA as string).toLowerCase();
      valB = (valB as string).toLowerCase();
    }

    if (valA < valB) return sortAsc ? -1 : 1;
    if (valA > valB) return sortAsc ? 1 : -1;
    return 0;
  });

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Prediction History</h1>
          <p className="text-sm text-secondary">View and query all historical autism screenings conducted on the platform.</p>
        </div>
        <button onClick={fetchHistory} className="btn-outline flex items-center gap-2">
          <FiActivity className="animate-pulse" />
          Refresh
        </button>
      </div>

      <div className="premium-card flex flex-col md:flex-row gap-4 justify-between items-center p-4">
        {/* Search */}
        <div className="relative w-full md:w-80">
          <span className="absolute inset-y-0 left-3 flex items-center text-muted">
            <FiSearch size={16} />
          </span>
          <input
            type="text"
            className="input-field w-full pl-10"
            placeholder="Search by ID, label, or model..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-2 w-full md:w-auto">
          {/* Status filter */}
          <div className="flex items-center gap-1 bg-surface-hover rounded-lg px-3 py-1.5 border border-border">
            <FiFilter className="text-muted" size={14} />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-transparent border-none text-sm text-primary-text outline-none cursor-pointer"
            >
              <option value="all">All Statuses</option>
              <option value="completed">Completed</option>
              <option value="processing">Processing</option>
              <option value="queued">Queued</option>
              <option value="failed">Failed</option>
            </select>
          </div>

          {/* Label filter */}
          <div className="flex items-center gap-1 bg-surface-hover rounded-lg px-3 py-1.5 border border-border">
            <FiFilter className="text-muted" size={14} />
            <select
              value={labelFilter}
              onChange={(e) => setLabelFilter(e.target.value)}
              className="bg-transparent border-none text-sm text-primary-text outline-none cursor-pointer"
            >
              <option value="all">All Diagnoses</option>
              <option value="ASD">ASD</option>
              <option value="TD">TD</option>
            </select>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full border-4 border-border border-t-indigo-500 h-10 w-10" />
        </div>
      ) : sortedRecords.length === 0 ? (
        <div className="premium-card text-center py-16 text-muted">
          <FiXCircle size={48} className="mx-auto mb-4 text-muted opacity-50" />
          <p className="text-lg font-medium">No screenings found</p>
          <p className="text-sm">Try adjusting your filters or upload a new child video analysis.</p>
        </div>
      ) : (
        <div className="premium-card overflow-hidden p-0 border border-border">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-hover border-b border-border text-xs font-semibold text-secondary uppercase tracking-wider">
                  <th className="p-4 cursor-pointer select-none" onClick={() => handleSort('id')}>
                    Job ID {sortField === 'id' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th className="p-4 cursor-pointer select-none" onClick={() => handleSort('status')}>
                    Status {sortField === 'status' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th className="p-4 cursor-pointer select-none" onClick={() => handleSort('predictionLabel')}>
                    Prediction {sortField === 'predictionLabel' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th className="p-4 cursor-pointer select-none" onClick={() => handleSort('confidenceScore')}>
                    Confidence {sortField === 'confidenceScore' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th className="p-4 cursor-pointer select-none" onClick={() => handleSort('processingTime')}>
                    Duration {sortField === 'processingTime' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th className="p-4 cursor-pointer select-none" onClick={() => handleSort('modelVersion')}>
                    Model {sortField === 'modelVersion' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th className="p-4 cursor-pointer select-none" onClick={() => handleSort('createdAt')}>
                    Date Run {sortField === 'createdAt' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th className="p-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border text-sm">
                {sortedRecords.map((record) => {
                  const isASD = record.predictionLabel === 'ASD';
                  return (
                    <tr key={record.id} className="hover:bg-surface-hover transition-colors">
                      <td className="p-4 font-mono font-medium text-xs text-secondary">
                        {record.id.slice(0, 8)}...
                      </td>
                      <td className="p-4">
                        {record.status === 'completed' && (
                          <span className="flex items-center gap-1 text-emerald-500 font-medium">
                            <FiCheckCircle size={14} /> Completed
                          </span>
                        )}
                        {record.status === 'processing' && (
                          <span className="flex items-center gap-1 text-amber-500 font-medium">
                            <span className="h-2 w-2 rounded-full bg-amber-500 animate-ping" /> Processing
                          </span>
                        )}
                        {record.status === 'queued' && (
                          <span className="flex items-center gap-1 text-slate-400 font-medium">
                            <span className="h-2 w-2 rounded-full bg-slate-400" /> Queued
                          </span>
                        )}
                        {record.status === 'failed' && (
                          <span className="flex items-center gap-1 text-red-500 font-medium">
                            <FiXCircle size={14} /> Failed
                          </span>
                        )}
                      </td>
                      <td className="p-4">
                        {record.status === 'completed' ? (
                          <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                            isASD ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
                          }`}>
                            {record.predictionLabel === 'ASD' ? 'Autism Spectrum Disorder' : 'Typical Development'}
                          </span>
                        ) : '-'}
                      </td>
                      <td className="p-4 font-semibold text-primary-text">
                        {record.status === 'completed' ? `${record.confidenceScore.toFixed(1)}%` : '-'}
                      </td>
                      <td className="p-4 text-secondary">
                        {record.processingTime ? `${record.processingTime.toFixed(1)}s` : '-'}
                      </td>
                      <td className="p-4 text-xs font-medium text-secondary">
                        {record.modelVersion || '-'}
                      </td>
                      <td className="p-4 text-secondary">
                        {record.createdAt}
                      </td>
                      <td className="p-4 text-right flex items-center justify-end gap-2">
                        <button
                          onClick={() => handleViewInDashboard(record)}
                          className="p-1.5 rounded hover:bg-surface-hover text-indigo-400 hover:text-indigo-300"
                          title="View on Dashboard"
                        >
                          <FiEye size={16} />
                        </button>
                        {record.status === 'completed' && (
                          <>
                            <a
                              href={`/reports/${record.id}/download/json`}
                              download
                              className="p-1.5 rounded hover:bg-surface-hover text-cyan-400 hover:text-cyan-300"
                              title="Download JSON Report"
                            >
                              <FiDownload size={16} />
                            </a>
                          </>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default History;
