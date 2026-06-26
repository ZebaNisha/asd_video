// src/api.ts
// Thin wrapper around Flask backend endpoints. No authentication needed for now.

const getHeaders = (extra?: Record<string, string>) => {
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...(extra || {}) };
  return headers;
};

export const api = {
  get: async (path: string) => {
    const res = await fetch(`/api${path}`, { method: 'GET', headers: getHeaders() });
    if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
    return res.json();
  },
  post: async (path: string, data: any) => {
    const res = await fetch(`/api${path}`, {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
    return res.json();
  },
  
  // Flask native endpoints
  fetchDashboardStats: () => api.get('/dashboard/stats'),
  fetchJobStatus: (id: string) => fetch(`/api/job/${id}`).then((res) => {
    if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
    return res.json();
  }),
  
  // JSON endpoint for predictions
  fetchPredictions: () => api.get('/predictions'),
  fetchSettings: () => api.get('/settings'),
  // Update server settings (placeholder)
  updateSettings: (data: any) => api.post('/settings', data),

  // Upload video route
  uploadFile: async (file: File, _onProgress?: (ev: ProgressEvent) => void) => {
    const formData = new FormData();
    formData.append('video', file); // Backend expects 'video'

    const res = await fetch('/upload', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);

    // Parse JSON response containing job information
    const data = await res.json();
    return { id: data.id, status: data.status };
  },
};

export const uploadFile = api.uploadFile;
export const fetchDashboardStats = api.fetchDashboardStats;
