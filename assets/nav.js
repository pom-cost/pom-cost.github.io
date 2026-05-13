// nav.js — Navbar scroll shrink, mobile toggle, climatological iframe
(function () {
  'use strict';

  const navbar   = document.getElementById('navbar');
  const toggle   = document.getElementById('navToggle');
  const navLinks = document.getElementById('navLinks');

  // ── Navbar shrink on scroll ──────────────────
  window.addEventListener('scroll', function () {
    if (window.scrollY > 60) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  }, { passive: true });

  // ── Mobile hamburger toggle ──────────────────
  toggle.addEventListener('click', function () {
    navLinks.classList.toggle('open');
  });

  // Close menu when a link is clicked
  navLinks.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') {
      navLinks.classList.remove('open');
    }
  });

  // ── Climatological iframe — lazy load on scroll into view ──
  const climSection = document.getElementById('climatologia');
  const climIframe  = document.getElementById('climatologicalIframe');

  if (climSection && climIframe) {
    let loaded = false;

    function loadClimIframe() {
      if (loaded) return;
      climIframe.src = 'plots/climatology_plot.html';
      loaded = true;
    }

    // Use IntersectionObserver when available; fall back to scroll
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver(function (entries) {
        if (entries[0].isIntersecting) {
          loadClimIframe();
          observer.disconnect();
        }
      }, { rootMargin: '200px' });
      observer.observe(climSection);
    } else {
      window.addEventListener('scroll', function onScroll() {
        const rect = climSection.getBoundingClientRect();
        if (rect.top < window.innerHeight + 200) {
          loadClimIframe();
          window.removeEventListener('scroll', onScroll);
        }
      }, { passive: true });
    }

    // Also reload on resize if needed
    window.addEventListener('resize', function () {
      if (!loaded) return;
      const desired = 'plots/climatology_plot.html';
      if (climIframe.src !== desired) {
        climIframe.src = desired;
      }
    }, { passive: true });
  }

})();
