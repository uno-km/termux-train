/**
 * AMEVA Documentation i18n & Utility Engine
 */

(function () {
  const STORAGE_KEY = 'ameva_docs_lang';
  const THEME_KEY = 'ameva_docs_theme';

  function getPreferredLanguage() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && window.AMEVA_TRANSLATIONS && window.AMEVA_TRANSLATIONS[saved]) {
      return saved;
    }
    const navLang = (navigator.language || 'en').slice(0, 2).toLowerCase();
    if (window.AMEVA_TRANSLATIONS && window.AMEVA_TRANSLATIONS[navLang]) {
      return navLang;
    }
    return 'en';
  }

  function setLanguage(lang) {
    if (!window.AMEVA_TRANSLATIONS || !window.AMEVA_TRANSLATIONS[lang]) return;
    localStorage.setItem(STORAGE_KEY, lang);
    const dict = window.AMEVA_TRANSLATIONS[lang];

    document.querySelectorAll('[data-i18n]').forEach((el) => {
      const key = el.getAttribute('data-i18n');
      if (dict[key]) {
        if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
          el.placeholder = dict[key];
        } else {
          el.innerHTML = dict[key];
        }
      }
    });

    const selector = document.getElementById('lang-selector');
    if (selector) selector.value = lang;
    document.documentElement.lang = lang;
  }

  function initTheme() {
    const savedTheme = localStorage.getItem(THEME_KEY) || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
      themeBtn.innerHTML = savedTheme === 'dark' ? '☀️' : '🌙';
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem(THEME_KEY, next);
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
      themeBtn.innerHTML = next === 'dark' ? '☀️' : '🌙';
    }
  }

  function initCopyButtons() {
    document.querySelectorAll('.copy-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const container = btn.closest('.code-container');
        if (!container) return;
        const code = container.querySelector('pre code') || container.querySelector('pre');
        if (!code) return;
        
        navigator.clipboard.writeText(code.innerText.trim()).then(() => {
          const orig = btn.innerText;
          btn.innerText = 'Copied! ✓';
          btn.style.color = '#00f5d4';
          setTimeout(() => {
            btn.innerText = orig;
            btn.style.color = '';
          }, 2000);
        });
      });
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    const currentLang = getPreferredLanguage();
    setLanguage(currentLang);

    const selector = document.getElementById('lang-selector');
    if (selector) {
      selector.addEventListener('change', (e) => setLanguage(e.target.value));
    }

    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
      themeBtn.addEventListener('click', toggleTheme);
    }

    initCopyButtons();
  });

  window.setLanguage = setLanguage;
})();
