// src/hooks/useJobPolling.ts
import { useEffect, useState, useCallback } from 'react';
import { mapJobStatusToStep, mapPredictionPayload } from '../api/adapter';
import type { PredictionResult } from '../api/adapter';

interface JobData {
  id: string;
  status: string;
  step: number;
  result?: PredictionResult;
}

/**
 * Poll the backend job endpoint until it reaches a terminal state.
 * Updates are returned via the hook's state so consuming components can
 * render progress, results, and handle errors.
 */
export const useJobPolling = (jobId: string, pollIntervalMs: number = 2000) => {
  const [job, setJob] = useState<JobData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const resp = await fetch(`/api/job/${jobId}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const step = mapJobStatusToStep(data.status);
      const result = data.status === 'completed' ? mapPredictionPayload(data) : undefined;
      setJob({ id: jobId, status: data.status, step, result });
      setLoading(data.status !== 'completed' && data.status !== 'failed');
    } catch (e: any) {
      setError(e.message);
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    fetchStatus();
    const interval = setInterval(fetchStatus, pollIntervalMs);
    return () => clearInterval(interval);
  }, [jobId, fetchStatus, pollIntervalMs]);

  return { job, loading, error };
};
