// prediction.js – Native premium canvas rendering for autism prediction probabilities

document.addEventListener('DOMContentLoaded', () => {
  const container = document.querySelector('.container[data-asd-prob]');
  if (!container) return;

  const asdProb = parseFloat(container.dataset.asdProb) || 0;
  const tdProb = parseFloat(container.dataset.tdProb) || 0;

  const canvas = document.getElementById('probabilityChart');
  if (!canvas) return;

  // Set sizing
  canvas.width = 600;
  canvas.height = 200;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Draw chart
  const drawChart = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Styling configuration
    const padding = 40;
    const barHeight = 35;
    const barSpacing = 30;
    const startX = 150;
    const maxWidth = canvas.width - startX - padding;

    // Draw background grid lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.font = '12px Outfit, Inter, sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';

    for (let i = 0; i <= 5; i++) {
      const pct = i * 20;
      const x = startX + (pct / 100) * maxWidth;
      // line
      ctx.beginPath();
      ctx.moveTo(x, 15);
      ctx.lineTo(x, 140);
      ctx.stroke();
      // text
      ctx.fillText(`${pct}%`, x, 155);
    }

    // Draw Bars
    const categories = [
      { label: 'ASD (Autism)', value: asdProb, color1: '#ef4444', color2: '#b91c1c' },
      { label: 'TD (Typical)', value: tdProb, color1: '#3b82f6', color2: '#1d4ed8' }
    ];

    categories.forEach((cat, index) => {
      const y = padding + index * (barHeight + barSpacing);
      const width = cat.value * maxWidth;

      // Label text
      ctx.fillStyle = '#1e293b';
      ctx.font = 'bold 13px Outfit, Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(cat.label, 15, y + barHeight / 2 + 5);

      // Bar rounded container path
      ctx.beginPath();
      const radius = 6;
      ctx.roundRect(startX, y, width > radius ? width : radius, barHeight, radius);
      
      // Gradient fill
      const grad = ctx.createLinearGradient(startX, 0, startX + width, 0);
      grad.addColorStop(0, cat.color1);
      grad.addColorStop(1, cat.color2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Confidence label inside or outside the bar
      ctx.fillStyle = '#1e293b';
      ctx.font = 'bold 13px Outfit, Inter, sans-serif';
      ctx.fillText(`${(cat.value * 100).toFixed(1)}%`, startX + width + 10, y + barHeight / 2 + 5);
    });
  };

  drawChart();
});
