// processing.js – polls job status and updates timeline UI

document.addEventListener('DOMContentLoaded', () => {
  const jobId = document.body.dataset.jobId || null;
  if (!jobId) return;

  const stepQueued = document.getElementById('step-queued');
  const stepProcessing = document.getElementById('step-processing');
  const stepCompleted = document.getElementById('step-completed');
  const stepFailed = document.getElementById('step-failed');

  const updateUI = (status) => {
    // Hide all steps first
    [stepQueued, stepProcessing, stepCompleted, stepFailed].forEach(el => el.style.display = 'none');
    if (status === 'queued') {
      stepQueued.style.display = 'block';
    } else if (status === 'processing') {
      stepProcessing.style.display = 'block';
    } else if (status === 'completed') {
      stepCompleted.style.display = 'block';
    } else if (status === 'failed') {
      stepFailed.style.display = 'block';
    }
  };

  const poll = async () => {
    try {
      const response = await fetch(`/api/job/${jobId}`);
      if (!response.ok) throw new Error('Network response was not ok');
      const data = await response.json();
      updateUI(data.status);
      if (data.status === 'completed') {
        // Redirect after a short delay to let user see the completed step
        setTimeout(() => {
          window.location.href = `/prediction/${jobId}`;
        }, 1500);
        return; // stop polling
      }
      if (data.status === 'failed') {
        // Stay on page; user can retry upload
        return;
      }
      // Continue polling
      setTimeout(poll, 3000);
    } catch (err) {
      console.error('Polling error:', err);
      // Retry after a longer interval
      setTimeout(poll, 5000);
    }
  };

  // Initial UI state based on presumed status
  updateUI('queued');
  poll();
});
