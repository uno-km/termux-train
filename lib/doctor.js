/**
 * termux-train: Hardware & Training Capacity Diagnostic Doctor
 * Production-grade heuristics with free RAM safety margin calculation.
 * Open-Source under Apache License 2.0.
 */

'use strict';

const os = require('os');
const { spawnSync } = require('child_process');

// Cache holds { cmd: string, resolvedAt: number } or null on failure
// Cache TTL: 60 seconds — allows re-detection after pyenv / venv switches
const PYTHON_CACHE_TTL_MS = 60_000;
let _pythonCache = null;

function resolvePythonCmd() {
  const now = Date.now();

  // Return cached result if fresh
  if (_pythonCache !== null && (now - _pythonCache.resolvedAt) < PYTHON_CACHE_TTL_MS) {
    return _pythonCache.cmd; // May be null if Python was not found at last check
  }

  const candidates = ['python3', 'python', 'py'];
  for (const cmd of candidates) {
    try {
      const res = spawnSync(cmd, ['--version'], { stdio: 'pipe', encoding: 'utf-8' });
      if (!res.error && res.status === 0) {
        _pythonCache = { cmd, resolvedAt: now };
        return cmd;
      }
    } catch (_) {}
  }

  // Cache the negative result with TTL so repeat calls don't thrash disk
  _pythonCache = { cmd: null, resolvedAt: now };
  return null;
}

/** Force re-detection on the next call (useful after Python installation). */
function invalidatePythonCache() {
  _pythonCache = null;
}

function runDoctor(options = {}) {
  const isTermux = Boolean(process.env.TERMUX_VERSION || (process.env.PREFIX && process.env.PREFIX.includes('com.termux')));
  const isAndroid = process.platform === 'android' || isTermux;
  const totalRamMb = typeof os.totalmem === 'function' ? Math.round(os.totalmem() / (1024 * 1024)) : 4096;
  const freeRamMb = typeof os.freemem === 'function' ? Math.round(os.freemem() / (1024 * 1024)) : 1024;
  const cpus = os.cpus();
  const cpuCores = Array.isArray(cpus) && cpus.length > 0 ? cpus.length : 4;

  // Usable capacity = 60% of currently available free RAM (40% safety headroom)
  const usableFreeRamMb = Math.max(0, Math.floor(freeRamMb * 0.6));

  let tier;
  let recLoraRank;
  let recBatchSize;

  if (usableFreeRamMb >= 4096) {
    tier = 'High-End Mobile / Workstation';
    recLoraRank = 16;
    recBatchSize = 32;
  } else if (usableFreeRamMb >= 1024) {
    tier = 'Standard Mobile';
    recLoraRank = 8;
    recBatchSize = 16;
  } else if (usableFreeRamMb >= 256) {
    tier = 'Low-Memory Mobile';
    recLoraRank = 4;
    recBatchSize = 4;
  } else {
    tier = 'Critical Low-Memory (<256MB Free)';
    recLoraRank = 2;
    recBatchSize = 1;
  }

  const pyCmd = resolvePythonCmd();
  let pythonAvailable = false;
  let pythonVersion = 'Not Installed';
  let termuxTrainPyInstalled = false;
  let vulkanDetected = false;

  if (pyCmd) {
    try {
      const pyVer = spawnSync(pyCmd, ['--version'], { stdio: 'pipe', encoding: 'utf-8' });
      if (!pyVer.error && pyVer.status === 0) {
        pythonAvailable = true;
        pythonVersion = (pyVer.stdout || pyVer.stderr).trim();
      }

      const checkMod = spawnSync(
        pyCmd,
        ['-c', 'import termux_train as tt; b = tt.get_backend("vulkan") if "vulkan" in tt.available_backends() else None; print(getattr(b, "is_vulkan_active", False))'],
        { stdio: 'pipe', encoding: 'utf-8' }
      );
      if (!checkMod.error && checkMod.status === 0) {
        termuxTrainPyInstalled = true;
        vulkanDetected = checkMod.stdout.trim().toLowerCase() === 'true';
      }
    } catch (_) {}
  }

  return {
    framework: 'termux-train',
    version: require('../package.json').version,
    platform: {
      system: process.platform,
      arch: process.arch,
      isTermux,
      isAndroid
    },
    hardware: {
      cpuCores,
      totalRamMb,
      freeRamMb,
      usableFreeRamMb,
      tier
    },
    python: {
      available: pythonAvailable,
      version: pythonVersion,
      binary: pyCmd,
      moduleInstalled: termuxTrainPyInstalled
    },
    vulkan: {
      gpuAcceleration: vulkanDetected,
      status: vulkanDetected ? 'AVAILABLE' : 'CPU_FALLBACK'
    },
    recommendedTrainingConfig: {
      loraRank: recLoraRank,
      batchSize: recBatchSize,
      seqLen: 512,
      safeTensorsMmap: true
    },
    status: pythonAvailable && termuxTrainPyInstalled ? 'READY' : 'SETUP_REQUIRED'
  };
}

module.exports = {
  resolvePythonCmd,
  invalidatePythonCache,
  runDoctor
};
