/**
 * termux-train: Multimodal & Distributed Training Node.js Bridge
 * Provides asynchronous JavaScript SDK bindings for Diffusion, Vision VLM,
 * Whisper STT, TTS Speaker Adaptation, and Cluster Orchestration.
 * Open-Source under Apache License 2.0.
 */

'use strict';

const { spawn } = require('child_process');
const { resolvePythonCmd } = require('./doctor');
const { PythonRuntimeNotFoundError, TrainingExecutionError } = require('./errors');

function _executePythonCli(subcommand, optArgs = {}) {
  return new Promise((resolve, reject) => {
    const pyCmd = resolvePythonCmd();
    if (!pyCmd) {
      return reject(new PythonRuntimeNotFoundError('Python runtime required for multimodal training execution.'));
    }

    const cliArgs = ['-m', 'termux_train.cli', subcommand];
    for (const [k, v] of Object.entries(optArgs)) {
      if (v === null || v === undefined) continue;
      const flagName = k.replace(/([A-Z])/g, '-$1').toLowerCase();
      if (typeof v === 'boolean') {
        if (v) cliArgs.push(`--${flagName}`);
      } else {
        cliArgs.push(`--${flagName}`, String(v));
      }
    }

    const child = spawn(pyCmd, cliArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdoutData = '';
    let stderrData = '';

    child.stdout.on('data', (d) => {
      stdoutData += d.toString();
    });

    child.stderr.on('data', (d) => {
      stderrData += d.toString();
    });

    child.on('error', (err) => {
      reject(err);
    });

    child.on('close', (code) => {
      if (code !== 0) {
        const err = new TrainingExecutionError(`Execution of '${subcommand}' failed with exit code ${code}:\n${stderrData || stdoutData}`);
        err.exitCode = code;
        err.stderr = stderrData;
        return reject(err);
      }
      resolve({
        status: 'SUCCESS',
        subcommand,
        stdout: stdoutData.trim(),
        stderr: stderrData.trim()
      });
    });
  });
}

function trainDiffusion(options = {}) {
  return _executePythonCli('diffusion-train', options);
}

function trainVision(options = {}) {
  return _executePythonCli('vision-train', options);
}

function trainSTT(options = {}) {
  return _executePythonCli('stt-train', options);
}

function trainTTS(options = {}) {
  return _executePythonCli('tts-train', options);
}

function runClusterProbe(options = {}) {
  return _executePythonCli('cluster-probe', options);
}

module.exports = {
  trainDiffusion,
  trainVision,
  trainSTT,
  trainTTS,
  runClusterProbe,
};
