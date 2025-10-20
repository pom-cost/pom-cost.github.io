// Cleaned-up JavaScript for index.html
// Include this file in index.html (replace the inline <script> block) with: <script src="assets/main.js"></script>

(function () {
  'use strict';

  // Configuration
  const CSV_PATH = 'data/individual_data.csv';
  const CLIM_IFRAME_ID = 'climatologicalIframe';
  const CLIM_OVERLAY_ID = 'climatologicalOverlay';
  const MAP_CENTER_DESKTOP = [43.44, -3.8];
  const MAP_CENTER_MOBILE = [43.5, -3.78];
  const MAP_ZOOM_DESKTOP = 11;
  const MAP_ZOOM_MOBILE = 9.5;
  const MOBILE_BREAKPOINT = 768; // px
  const COLOR_MIN = -3;
  const COLOR_MAX = 3;

  // State
  let map = null;
  let markerLayer = null; // L.LayerGroup for markers (easy clear/add)
  let allMarkers = [];    // array of { marker: L.CircleMarker, month: 'MM' }
  let resizeDebounceTimer = null;
  let climResizeDebounceTimer = null;

  // Utility: debounce
  function debounce(fn, wait) {
    let t = null;
    return function () {
      const ctx = this, args = arguments;
      clearTimeout(t);
      t = setTimeout(() => fn.apply(ctx, args), wait);
    };
  }

  // Utility: safe parse month from several date formats
  // Returns '01' .. '12' or '' if unknown
  function getMonthString(dateStr) {
    if (!dateStr) return '';
    // Try ISO-ish parsing first
    const isoMatch = dateStr.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (isoMatch) return isoMatch[2];
    // Try common US format MM/DD/YYYY or M/D/YYYY
    const usMatch = dateStr.match(/^(\d{1,2})\/(\d{1,2})\/(\d{2,4})/);
    if (usMatch) {
      const mm = String(parseInt(usMatch[1], 10)).padStart(2, '0');
      return mm;
    }
    // Fallback: try Date parsing (may be locale dependent)
    const d = new Date(dateStr);
    if (!isNaN(d.getTime())) {
      return String(d.getMonth() + 1).padStart(2, '0');
    }
    return '';
  }

  // Utility: color interpolation blue -> white -> red for range COLOR_MIN..COLOR_MAX
  function getColor(tempAnomRaw) {
    let tempAnom = parseFloat(tempAnomRaw);
    if (!isFinite(tempAnom)) tempAnom = 0;
    const min = COLOR_MIN, max = COLOR_MAX;
    const mid = (min + max) / 2;

    const blue = [0, 0, 255];
    const white = [255, 255, 255];
    const red = [255, 0, 0];

    let startColor, endColor, ratio;
    if (tempAnom < mid) {
      startColor = blue;
      endColor = white;
      ratio = (tempAnom - min) / (mid - min);
    } else {
      startColor = white;
      endColor = red;
      ratio = (tempAnom - mid) / (max - mid);
    }
    ratio = Math.max(0, Math.min(1, ratio));

    const r = Math.round(startColor[0] + ratio * (endColor[0] - startColor[0]));
    const g = Math.round(startColor[1] + ratio * (endColor[1] - startColor[1]));
    const b = Math.round(startColor[2] + ratio * (endColor[2] - startColor[2]));
    return `rgb(${r},${g},${b})`;
  }

  // Init map and controls
  function initMap() {
    const isMobile = window.innerWidth < MOBILE_BREAKPOINT;
    const center = isMobile ? MAP_CENTER_MOBILE : MAP_CENTER_DESKTOP;
    const zoom = isMobile ? MAP_ZOOM_MOBILE : MAP_ZOOM_DESKTOP;

    map = L.map('map').setView(center, zoom);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    // Marker layer group for easy filtering/clearing
    markerLayer = L.layerGroup().addTo(map);

    // Add legend / color bar
    const colorBarControl = L.control({ position: 'bottomleft' });
    colorBarControl.onAdd = function () {
      const div = L.DomUtil.create('div', 'color-bar-container');
      div.innerHTML = '<div class="color-bar-title">Relative Temperature to Sardinero (70s)</div>' +
                      '<div class="color-bar">-3°C to 3°C</div>';
      return div;
    };
    colorBarControl.addTo(map);

    // Add month dropdown
    const dropdownControl = L.control({ position: 'bottomleft' });
    dropdownControl.onAdd = function () {
      const div = L.DomUtil.create('div', 'dropdown-container');
      div.innerHTML = '<label for="month-select" style="display:none">Month</label>' +
                      '<select id="month-select" aria-label="Filter by month">' +
                      '<option value="all">All Months</option>' +
                      '<option value="01">January</option>' +
                      '<option value="02">February</option>' +
                      '<option value="03">March</option>' +
                      '<option value="04">April</option>' +
                      '<option value="05">May</option>' +
                      '<option value="06">June</option>' +
                      '<option value="07">July</option>' +
                      '<option value="08">August</option>' +
                      '<option value="09">September</option>' +
                      '<option value="10">October</option>' +
                      '<option value="11">November</option>' +
                      '<option value="12">December</option>' +
                      '</select>';
      // Avoid stopping map interactions when interacting with the select
      L.DomEvent.disableClickPropagation(div);
      return div;
    };
    dropdownControl.addTo(map);

    // Wire up change handler (expose globally because HTML used inline onchange previously)
    document.addEventListener('change', function (e) {
      if (e.target && e.target.id === 'month-select') {
        filterMarkersByMonth();
      }
    });
    // Also expose globally so inline attributes still work
    window.filterMarkersByMonth = filterMarkersByMonth;
  }

  // Fetch CSV and add markers
  function loadCSVAndAddMarkers() {
    fetch(CSV_PATH)
      .then(response => {
        if (!response.ok) throw new Error('CSV fetch failed: ' + response.statusText);
        return response.text();
      })
      .then(csvText => {
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          complete: function (results) {
            if (!results || !results.data) {
              console.error('No CSV rows parsed');
              return;
            }
            results.data.forEach((row, idx) => {
              try {
                const lat = parseFloat(row.Latitude);
                const lon = parseFloat(row.Longitude);
                if (!isFinite(lat) || !isFinite(lon)) return; // skip invalid rows

                const temp = row.Temperature || '';
                const tempAnom = parseFloat(row.Temperature_Anomaly);
                const date = row.Date || '';
                const month = getMonthString(date);

                const fillColor = getColor(tempAnom);

                const popupContent = `<div style="width: 200px; font-family: Montserrat, sans-serif;">` +
                                     `<strong>Sea Temperature:</strong> ${temp}°C<br>` +
                                     `<strong>Date:</strong> ${date}` +
                                     `</div>`;

                const popupOptions = {
                  maxWidth: 260,
                  // allow some vertical scrolling if content tall
                  className: 'data-popup'
                };

                const marker = L.circleMarker([lat, lon], {
                  radius: 6,
                  color: 'gray',
                  weight: 1,
                  fillColor: fillColor,
                  fillOpacity: 0.6
                });

                marker.bindPopup(popupContent, popupOptions);

                // store marker with metadata, but add to layer only once
                allMarkers.push({ marker: marker, month: month });
                markerLayer.addLayer(marker);
              } catch (errInner) {
                console.error('Error processing CSV row', idx, errInner);
              }
            });
          },
          error: function (err) {
            console.error('PapaParse error:', err);
          }
        });
      })
      .catch(err => {
        console.error('Error loading CSV:', err);
      });
  }

  // Filter markers by month (uses markerLayer for efficiency)
  function filterMarkersByMonth() {
    const select = document.getElementById('month-select');
    const selectedMonth = select ? select.value : 'all';
    markerLayer.clearLayers();
    if (selectedMonth === 'all') {
      allMarkers.forEach(obj => markerLayer.addLayer(obj.marker));
      return;
    }
    allMarkers.forEach(obj => {
      if (obj.month === selectedMonth) {
        markerLayer.addLayer(obj.marker);
      }
    });
  }

  // Overlay sizing with cross-origin-safe fallbacks
  function adjustOverlaySizeById(id) {
    const overlay = document.getElementById(id);
    if (!overlay) return;
    const iframe = overlay.querySelector('iframe');
    if (!iframe) return;

    function resize() {
      try {
        const doc = iframe.contentDocument || iframe.contentWindow.document;
        if (doc && doc.body) {
          const h = doc.body.scrollHeight;
          if (h && h > 0) {
            iframe.style.height = h + 'px';
            return;
          }
        }
      } catch (e) {
        // cross-origin or access denied
      }
      // fallback: base on viewport height with a minimum
      const fallback = Math.max(window.innerHeight * 0.6, 600);
      iframe.style.height = fallback + 'px';
    }

    // Resize when iframe loads
    iframe.addEventListener('load', function () {
      // small delay to let internal resources render
      setTimeout(resize, 250);
    });

    // also attempt an immediate resize
    setTimeout(resize, 350);
  }

  // Set climatological iframe src based on screen size, but avoid unnecessary reloads
  function setClimatologicalSrcIfNeeded() {
    const iframe = document.getElementById(CLIM_IFRAME_ID);
    if (!iframe) return;
    const isMobile = window.innerWidth < MOBILE_BREAKPOINT;
    const desired = isMobile ? 'climatological-year/climatological_plot_mbl.html' : 'climatological-year/climatological_plot.html';
    if (iframe.src && iframe.getAttribute('data-src') === desired) {
      // already set to desired file
      return;
    }
    iframe.setAttribute('data-src', desired);
    // Set src (this will load the iframe)
    iframe.src = desired;
  }

  // Expose toggleOverlay and closeOverlay for inline onclick usage in HTML
  window.toggleOverlay = function (id) {
    // close other overlays
    document.querySelectorAll('.overlay.active').forEach(o => o.classList.remove('active'));

    const overlay = document.getElementById(id);
    if (!overlay) return;

    overlay.classList.add('active');
    adjustOverlaySizeById(id);

    // Special case: if climatological overlay opened, set appropriate iframe src
    if (id === CLIM_OVERLAY_ID) {
      setClimatologicalSrcIfNeeded();
    }
  };

  window.closeOverlay = function (id) {
    const overlay = document.getElementById(id);
    if (!overlay) return;
    overlay.classList.remove('active');
  };

  // Toggle mobile menu
  window.toggleMenu = function () {
    const menu = document.querySelector('.map-menu');
    if (menu) menu.classList.toggle('active');
  };

  // Initialization runner
  function init() {
    initMap();
    loadCSVAndAddMarkers();

    // Add fixed site markers (existing code adapted)
    const watericonblue = L.icon({
      iconUrl: 'assets/water_icon3.png',
      iconSize: [32, 32],
      iconAnchor: [16, 32],
      popupAnchor: [0, -32]
    });

    const sites = [
      { lat: 43.4756, lon: -3.7847, popup: 'sardinero_plot.html', w: 400, h: 200 },
      { lat: 43.4704, lon: -3.7653, popup: 'magdalena_plot.html', w: 400, h: 200 },
      { lat: 43.4508, lon: -3.8215, popup: 'miguel_plot.html', w: 420, h: 250 },
      { lat: 43.4634, lon: -3.7827, popup: 'IEO_plot.html', w: 420, h: 250 }
    ];

    sites.forEach(s => {
      const content = `<div style="width:${s.w}px; height:${s.h}px; overflow:hidden;">` +
                      `<iframe src="${s.popup}" style="border:none; width:100%; height:100%;" loading="lazy"></iframe>` +
                      `</div>`;
      L.marker([s.lat, s.lon], { icon: watericonblue }).addTo(map).bindPopup(content);
    });

    // Debounced window resize: adjust overlays and (if climatological overlay open) update src
    const onResize = debounce(function () {
      // reposition map for mobile/desktop if needed (optional)
      const isMobile = window.innerWidth < MOBILE_BREAKPOINT;
      if (map) {
        // adjust the map view if desired (same logic as initial)
        if (isMobile) {
          map.setView(MAP_CENTER_MOBILE, MAP_ZOOM_MOBILE);
        } else {
          map.setView(MAP_CENTER_DESKTOP, MAP_ZOOM_DESKTOP);
        }
      }

      // resize active overlays
      document.querySelectorAll('.overlay.active').forEach(overlay => {
        const id = overlay.id;
        adjustOverlaySizeById(id);
      });

      // update climatological iframe only if overlay open
      const climOverlay = document.getElementById(CLIM_OVERLAY_ID);
      if (climOverlay && climOverlay.classList.contains('active')) {
        setClimatologicalSrcIfNeeded();
      }
    }, 200);

    window.addEventListener('resize', onResize);

    // Ensure any existing close-buttons wired with data-overlay work too
    document.querySelectorAll('.close-btn').forEach(function (button) {
      const dataOverlay = button.getAttribute('data-overlay');
      // if data-overlay attr exists, use it; otherwise keep existing click handlers in DOM
      if (dataOverlay) {
        button.addEventListener('click', function () {
          window.closeOverlay(dataOverlay);
        });
      }
    });

    // expose some helpers for debugging if needed
    window._pomcost = {
      allMarkers,
      markerLayer,
      reloadCSV: loadCSVAndAddMarkers
    };
  }

  // Run init when DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
