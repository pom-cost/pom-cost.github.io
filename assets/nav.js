// nav.js — Navbar scroll shrink, mobile toggle, climatological iframe
(function () {
  'use strict';

  const navbar   = document.getElementById('navbar');
  const toggle   = document.getElementById('navToggle');
  const navLinks = document.getElementById('navLinks');
  const MOBILE   = 768;

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
      const isMobile = window.innerWidth < MOBILE;
      climIframe.src = isMobile
        ? 'climatological-year/climatological_plot_mbl.html'
        : 'climatological-year/climatological_plot.html';
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

    // Also reload with correct src on resize (e.g. rotate phone)
    window.addEventListener('resize', function () {
      if (!loaded) return;
      const isMobile = window.innerWidth < MOBILE;
      const desired = isMobile
        ? 'climatological-year/climatological_plot_mbl.html'
        : 'climatological-year/climatological_plot.html';
      if (climIframe.src !== desired) {
        climIframe.src = desired;
      }
    }, { passive: true });
  }

})();
