// upload.js – handles video upload form submission

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('upload-form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        // Show a Bootstrap toast (assumes toast container exists)
        const toastEl = document.getElementById('upload-success-toast');
        if (toastEl) new bootstrap.Toast(toastEl).show();
        // Redirect to processing page
        window.location.href = `/processing/${data.job_id}`;
      } else {
        alert(data.error || 'Upload failed');
      }
    } catch (err) {
      console.error('Upload error:', err);
      alert('An unexpected error occurred');
    }
  });
});
