// main.js - Client side logic for ASD Vision Dashboard and Patient Management

document.addEventListener('DOMContentLoaded', () => {
  // Check if we are on the dashboard page
  if (document.getElementById('total-patients')) {
    loadDashboardStats();
  }

  // Check if we are on the patients page
  if (document.getElementById('patientsTable')) {
    loadPatients();
    setupPatientForm();
  }
});

// --- DASHBOARD FUNCTIONS ---
async function loadDashboardStats() {
  try {
    const response = await fetch('/api/dashboard/stats');
    if (!response.ok) throw new Error('Failed to fetch dashboard stats');
    const data = await response.json();

    // Update metric cards
    document.getElementById('total-patients').innerText = data.total_patients;
    document.getElementById('videos-uploaded').innerText = data.videos_uploaded;
    document.getElementById('completed-analyses').innerText = data.completed_analyses;
    document.getElementById('processing-queue').innerText = data.processing_queue;

    // Populate recent predictions table
    const tbody = document.getElementById('recent-predictions-body');
    tbody.innerHTML = '';

    if (data.recent_predictions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">No recent predictions found.</td></tr>`;
      return;
    }

    data.recent_predictions.forEach(job => {
      const row = document.createElement('tr');
      const formattedDate = job.created_at ? new Date(job.created_at).toLocaleString() : '-';
      const badgeClass = job.prediction_label === 'ASD' ? 'bg-danger' : job.prediction_label === 'TD' ? 'bg-primary' : 'bg-secondary';
      const confidence = job.status === 'completed' ? `${(job.confidence_score * 100).toFixed(1)}%` : '-';
      const procTime = job.status === 'completed' ? `${job.processing_time}s` : '-';

      row.innerHTML = `
        <td><code>${job.id.substring(0, 8)}...</code></td>
        <td>General Patient</td>
        <td><span class="badge ${badgeClass}">${job.prediction_label}</span></td>
        <td>${confidence}</td>
        <td>${procTime}</td>
        <td>
          <a href="/prediction/${job.id}" class="btn btn-sm btn-outline-primary">Results</a>
          <a href="/reports/${job.id}" class="btn btn-sm btn-outline-secondary">Report</a>
        </td>
      `;
      tbody.appendChild(row);
    });
  } catch (err) {
    console.error('Error loading dashboard stats:', err);
  }
}

// --- PATIENTS FUNCTIONS ---
async function loadPatients() {
  try {
    const response = await fetch('/patients', {
      headers: { 'Accept': 'application/json' }
    });
    if (!response.ok) throw new Error('Failed to fetch patients');
    const patients = await response.json();

    const tbody = document.querySelector('#patientsTable tbody');
    tbody.innerHTML = '';

    if (patients.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No patients registered. Click "Add Patient" to start.</td></tr>`;
      return;
    }

    patients.forEach(p => {
      const row = document.createElement('tr');
      const addedDate = p.created_at ? new Date(p.created_at).toLocaleDateString() : '-';
      row.innerHTML = `
        <td><code>PT-${p.id}</code></td>
        <td><strong>${p.name}</strong></td>
        <td>${p.age || '-'}</td>
        <td>${p.gender || '-'}</td>
        <td>${p.guardian_name || '-'}</td>
        <td>${addedDate}</td>
        <td>
          <button class="btn btn-sm btn-outline-primary" onclick="editPatient(${p.id})"><i class="bi bi-pencil-fill"></i> Edit</button>
          <button class="btn btn-sm btn-outline-danger" onclick="deletePatient(${p.id})"><i class="bi bi-trash-fill"></i> Delete</button>
        </td>
      `;
      tbody.appendChild(row);
    });
  } catch (err) {
    console.error('Error loading patients:', err);
  }
}

function setupPatientForm() {
  const saveBtn = document.getElementById('savePatientBtn');
  const form = document.getElementById('patientForm');

  // Since the submit button is inside the modal footer, trigger form submit manually
  saveBtn.addEventListener('click', (e) => {
    e.preventDefault();
    if (form.reportValidity()) {
      savePatient();
    }
  });

  // Reset form when modal is closed
  const modalEl = document.getElementById('patientModal');
  modalEl.addEventListener('hidden.bs.modal', () => {
    form.reset();
    document.getElementById('patientId').value = '';
    document.getElementById('patientModalLabel').innerText = 'Add Patient';
  });
}

async function savePatient() {
  const form = document.getElementById('patientForm');
  const formData = new FormData(form);
  const data = {};
  formData.forEach((value, key) => data[key] = value);

  try {
    const response = await fetch('/patients', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (!response.ok) throw new Error('Failed to save patient');

    // Hide Modal
    const modalEl = document.getElementById('patientModal');
    const modal = bootstrap.Modal.getInstance(modalEl);
    if (modal) modal.hide();

    // Reload patients table
    loadPatients();
  } catch (err) {
    console.error('Error saving patient:', err);
    alert('Error saving patient data. Please try again.');
  }
}

async function editPatient(id) {
  try {
    const response = await fetch(`/patient/${id}`, {
      headers: { 'Accept': 'application/json' }
    });
    if (!response.ok) throw new Error('Failed to fetch patient details');
    const p = await response.json();

    // Fill form fields
    document.getElementById('patientId').value = p.id;
    document.getElementById('patientName').value = p.name;
    document.getElementById('patientAge').value = p.age || '';
    document.getElementById('patientGender').value = p.gender || '';
    document.getElementById('patientGuardian').value = p.guardian_name || '';
    document.getElementById('patientNotes').value = p.notes || '';

    document.getElementById('patientModalLabel').innerText = 'Edit Patient';

    // Show modal
    const modalEl = document.getElementById('patientModal');
    const modal = new bootstrap.Modal(modalEl);
    modal.show();
  } catch (err) {
    console.error('Error editing patient:', err);
  }
}

async function deletePatient(id) {
  if (!confirm('Are you sure you want to delete this patient?')) return;

  try {
    const response = await fetch(`/patient/${id}`, {
      method: 'DELETE'
    });
    if (!response.ok) throw new Error('Failed to delete patient');
    loadPatients();
  } catch (err) {
    console.error('Error deleting patient:', err);
    alert('Error deleting patient. Please try again.');
  }
}
