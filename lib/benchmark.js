/**
 * termux-train: Benchmark Suite
 * High-performance GEMM & DAG Autograd Latency & Throughput Benchmark.
 * All benchmark execution uses stdin JSON IPC via termux_train.runtime.benchmarker
 * — zero python -c inline script injection.
 * Open-Source under Apache License 2.0.
 */

'use strict';

const { spawn, spawnSync } = require('child_process');
const { resolvePythonCmd } = require('./doctor');
const { PythonRuntimeNotFoundError, TermuxTrainModuleNotFoundError } = require('./errors');

function _validateBenchmarkOptions(options = {}) {
  const rawDim = options.dim !== undefined ? options.dim : 256;
  const rawIters = options.iters !== undefined ? options.iters : 10;

  if (!Number.isInteger(rawDim) || rawDim < 2 || rawDim > 4096) {
    throw new TypeError(`Invalid dimension: ${rawDim}. Must be an integer between 2 and 4096.`);
  }

  if (!Number.isInteger(rawIters) || rawIters < 1 || rawIters > 1000) {
    throw new TypeError(`Invalid iterations: ${rawIters}. Must be an integer between 1 and 1000.`);
  }

  return {
    dim: Math.floor(rawDim),
    iters: Math.floor(rawIters)
  };
}

async function runBenchmarkAsync(options = {}) {
  const { dim, iters } = _validateBenchmarkOptions(options);
  const pyCmd = resolvePythonCmd();

  if (!pyCmd) {
    throw new PythonRuntimeNotFoundError();
  }

  const cfg = JSON.stringify({ dim, iters });

  return new Promise((resolve, reject) => {
    const proc = spawn(pyCmd, ['-m', 'termux_train.runtime.benchmarker', '--stdin-json'], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    proc.stdin.write(cfg);
    proc.stdin.end();

    let stdoutAcc = '';
    let stderrAcc = '';

    proc.stdout.on('data', (c) => { stdoutAcc += c.toString('utf-8'); });
    proc.stderr.on('data', (c) => { stderrAcc += c.toString('utf-8'); });

    proc.on('close', (code) => {
      if (code !== 0) {
        if (stderrAcc.includes('No module named termux_train')) {
          return reject(new TermuxTrainModuleNotFoundError());
        }
        return reject(new Error(`Benchmark process failed (exit code ${code}): ${stderrAcc}`));
      }

      try {
        const parsed = JSON.parse(stdoutAcc.trim());
        resolve(parsed);
      } catch (e) {
        reject(new Error(`Failed to parse benchmark JSON output: ${stdoutAcc}`));
      }
    });

    proc.on('error', (err) => {
      reject(err);
    });
  });
}

function runBenchmark(options = {}) {
  const { dim, iters } = _validateBenchmarkOptions(options);
  const pyCmd = resolvePythonCmd();

  if (!pyCmd) {
    throw new PythonRuntimeNotFoundError();
  }

  const cfg = JSON.stringify({ dim, iters });

  const res = spawnSync(pyCmd, ['-m', 'termux_train.runtime.benchmarker', '--stdin-json'], {
    input: cfg,
    stdio: ['pipe', 'pipe', 'pipe'],
    encoding: 'utf-8'
  });

  if (res.status !== 0) {
    const err = res.stderr || res.stdout || '';
    if (err.includes('No module named termux_train')) {
      throw new TermuxTrainModuleNotFoundError();
    }
    throw new Error(`Benchmark failed: ${err}`);
  }

  try {
    return JSON.parse(res.stdout.trim());
  } catch (e) {
    throw new Error(`Failed to parse benchmark output: ${res.stdout}`);
  }
}

module.exports = {
  runBenchmark,
  runBenchmarkAsync,
  benchmark: runBenchmarkAsync
};
