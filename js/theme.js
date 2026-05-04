// Theme toggle: light · dark · system. Stored in localStorage as 'theme'.
// The FOUC-prevention snippet at the top of each <head> applies the saved
// theme before paint; this file injects the toggle buttons and wires clicks.
(function () {
  const STORAGE_KEY = 'theme';
  const VALID = ['light', 'dark', 'system'];

  const ICONS = {
    light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>',
    dark:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
    system:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>',
  };

  function getTheme() {
    const t = localStorage.getItem(STORAGE_KEY);
    return VALID.includes(t) ? t : 'system';
  }

  function applyTheme(theme) {
    if (theme === 'system') {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', theme);
    }
  }

  function setActive(theme) {
    document.querySelectorAll('.theme-toggle button[data-set-theme]').forEach(btn => {
      btn.setAttribute('aria-pressed', String(btn.dataset.setTheme === theme));
    });
  }

  function setTheme(theme) {
    if (!VALID.includes(theme)) return;
    localStorage.setItem(STORAGE_KEY, theme);
    applyTheme(theme);
    setActive(theme);
  }

  function buildToggle(el) {
    if (el.querySelector('button')) return; // already built
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', 'Theme');
    el.innerHTML = VALID.map(t =>
      `<button type="button" data-set-theme="${t}" aria-label="${t[0].toUpperCase() + t.slice(1)} theme" title="${t[0].toUpperCase() + t.slice(1)}">${ICONS[t]}</button>`
    ).join('');
    el.addEventListener('click', e => {
      const btn = e.target.closest('button[data-set-theme]');
      if (btn) setTheme(btn.dataset.setTheme);
    });
  }

  function init() {
    const current = getTheme();
    applyTheme(current);
    document.querySelectorAll('.theme-toggle').forEach(buildToggle);
    setActive(current);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
