document.addEventListener('DOMContentLoaded', () => {
    const batterySelect = document.getElementById('battery-select');
    
    // UI Elements
    const metricCycles = document.getElementById('metric-cycles');
    const metricSoh = document.getElementById('metric-soh');
    const metricFade = document.getElementById('metric-fade');
    const metricStatus = document.getElementById('metric-status');

    // Chart instances
    let sohChart = null;
    let fadeChart = null;

    // Common Chart.js options for aesthetic dark mode glass style
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Outfit', sans-serif";
    
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#e2e8f0',
                    usePointStyle: true,
                    boxWidth: 8
                }
            },
            tooltip: {
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                titleColor: '#f8fafc',
                bodyColor: '#cbd5e1',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                padding: 12,
                cornerRadius: 8,
                displayColors: false
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255,255,255,0.05)',
                    drawBorder: false
                }
            },
            y: {
                grid: {
                    color: 'rgba(255,255,255,0.05)',
                    drawBorder: false
                }
            }
        },
        interaction: {
            mode: 'index',
            intersect: false,
        }
    };

    // Load Battery List
    fetch('/api/batteries')
        .then(res => res.json())
        .then(data => {
            batterySelect.innerHTML = '';
            data.forEach(id => {
                const opt = document.createElement('option');
                opt.value = id;
                opt.textContent = id;
                batterySelect.appendChild(opt);
            });
            
            // Load initial battery
            if (data.length > 0) {
                loadBatteryData(data[0]);
            }
        });

    batterySelect.addEventListener('change', (e) => {
        loadBatteryData(e.target.value);
    });

    function loadBatteryData(batteryId) {
        fetch(`/api/battery/${batteryId}`)
            .then(res => res.json())
            .then(data => {
                updateMetrics(data);
                updateCharts(data, batteryId);
            });
    }

    function updateMetrics(data) {
        if (!data || data.length === 0) return;
        
        const lastCycle = data[data.length - 1];
        const maxFade = Math.max(...data.map(d => d.capacity_fade_pct));

        // Animate numbers
        animateValue(metricCycles, 0, lastCycle.discharge_cycle, 1000, '');
        animateValue(metricSoh, 0, lastCycle.soh_pct, 1000, '%');
        animateValue(metricFade, 0, maxFade, 1000, '%');

        // Status logic
        if (lastCycle.is_end_of_life) {
            metricStatus.textContent = "EOL Reached";
            metricStatus.style.color = "var(--status-bad)";
        } else if (lastCycle.is_near_eol) {
            metricStatus.textContent = "Warning (Near EOL)";
            metricStatus.style.color = "var(--status-warn)";
        } else {
            metricStatus.textContent = "Healthy";
            metricStatus.style.color = "var(--status-good)";
        }
    }

    function updateCharts(data, batteryId) {
        const labels = data.map(d => `Cycle ${d.discharge_cycle}`);
        const sohData = data.map(d => d.soh_pct);
        const rollingSohData = data.map(d => d.rolling_soh_7);
        const fadeData = data.map(d => d.capacity_fade_pct);

        // 1. SOH Chart
        const sohCtx = document.getElementById('sohChart').getContext('2d');
        
        // Gradient for SOH area
        const sohGradient = sohCtx.createLinearGradient(0, 0, 0, 350);
        sohGradient.addColorStop(0, 'rgba(59, 130, 246, 0.4)');
        sohGradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

        if (sohChart) sohChart.destroy();
        sohChart = new Chart(sohCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Raw SOH (%)',
                        data: sohData,
                        borderColor: 'rgba(255,255,255,0.2)',
                        borderWidth: 1,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'Rolling SOH (%)',
                        data: rollingSohData,
                        borderColor: '#3b82f6',
                        backgroundColor: sohGradient,
                        borderWidth: 3,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: {
                ...chartOptions,
                animation: {
                    duration: 1500,
                    easing: 'easeOutQuart'
                }
            }
        });

        // 2. Capacity Fade Chart
        const fadeCtx = document.getElementById('fadeChart').getContext('2d');
        
        const fadeGradient = fadeCtx.createLinearGradient(0, 0, 0, 350);
        fadeGradient.addColorStop(0, 'rgba(139, 92, 246, 0.4)');
        fadeGradient.addColorStop(1, 'rgba(139, 92, 246, 0.0)');

        if (fadeChart) fadeChart.destroy();
        fadeChart = new Chart(fadeCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Capacity Fade (%)',
                    data: fadeData,
                    borderColor: '#8b5cf6',
                    backgroundColor: fadeGradient,
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...chartOptions,
                animation: {
                    duration: 1500,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    // Number animation helper
    function animateValue(obj, start, end, duration, suffix) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            
            // easeOutQuad
            const easeOut = progress * (2 - progress);
            
            let val = (start + easeOut * (end - start)).toFixed(1);
            if (end % 1 === 0) val = Math.round(val); // remove decimals for integers
            
            obj.innerHTML = val + suffix;
            
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                let finalVal = Number(end).toFixed(1);
                if (end % 1 === 0) finalVal = end;
                obj.innerHTML = finalVal + suffix;
            }
        };
        window.requestAnimationFrame(step);
    }
});
