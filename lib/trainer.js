/**
 * termux-train: Production-Grade On-Device Training Orchestrator
 * True Dual-Engine LoRA, Transformer & MLP Training Pipeline with SafeTensors I/O.
 * Open-Source under Apache License 2.0.
 */

'use strict';

const { EventEmitter } = require('events');
const { spawn, spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const { resolvePythonCmd } = require('./doctor');
const {
  PythonRuntimeNotFoundError,
  TrainingExecutionError,
  InvalidCheckpointError
} = require('./errors');

// Maximum stderr accumulation per session: 1MB cap to prevent heap exhaustion
const MAX_STDERR_BYTES = 1024 * 1024;

class TermuxTrainer extends EventEmitter {
  constructor(config = {}) {
    super();
    // Default max listener count raised to support multi-epoch long training
    this.setMaxListeners(50);

    const modelType = typeof config.modelType === 'string'
      ? config.modelType.toLowerCase().trim()
      : 'mlp';

    const dim = Number.isInteger(config.dim) && config.dim >= 4 && config.dim <= 4096
      ? config.dim
      : 32;

    const hiddenDim = Number.isInteger(config.hiddenDim) && config.hiddenDim >= 4
      ? config.hiddenDim
      : dim * 2;

    const outDim = Number.isInteger(config.outDim) && config.outDim >= 1
      ? config.outDim
      : (modelType.includes('lora') ? dim : 1);

    const maxRank = Math.min(dim, outDim);
    const loraRank = Number.isInteger(config.loraRank) && config.loraRank >= 1
      ? Math.min(config.loraRank, maxRank)
      : Math.min(4, maxRank);

    const loraAlpha = typeof config.loraAlpha === 'number' && isFinite(config.loraAlpha) && config.loraAlpha > 0
      ? config.loraAlpha
      : 1.0;

    const vocabSize = Number.isInteger(config.vocabSize) && config.vocabSize >= 2
      ? config.vocabSize
      : 256;

    const layers = Number.isInteger(config.layers) && config.layers >= 1 && config.layers <= 32
      ? config.layers
      : 2;

    // Determine heads: use explicit user configuration, or choose adaptive default that divides dim evenly
    let heads;
    if (Number.isInteger(config.heads) && config.heads >= 1) {
      if (dim % config.heads !== 0) {
        throw new TypeError(
          `config.heads=${config.heads} does not evenly divide dim=${dim}. ` +
          `Choose heads that evenly divide dim (e.g., ${this._suggestHeads(dim)}).`
        );
      }
      heads = config.heads;
    } else {
      // Default: 2 for even dim, 1 for odd dim
      heads = (dim % 2 === 0) ? 2 : 1;
    }

    if (dim % heads !== 0) {
      throw new TypeError(
        `Resolved heads=${heads} does not evenly divide dim=${dim}. ` +
        `Choose heads that evenly divide dim (e.g., ${this._suggestHeads(dim)}).`
      );
    }

    // Validate RoPE Transformer Invariants: head_dim = dim / heads must be an even integer
    if (modelType.includes('transformer') || modelType.includes('rope')) {
      if (dim % 2 !== 0) {
        throw new TypeError(
          `Transformer models with RoPE rotary embeddings require an even dimension, got dim=${dim}. ` +
          `RoPE operates on 2D coordinate pairs. Choose an even dimension (e.g., ${dim + 1}).`
        );
      }
      const headDim = Math.floor(dim / heads);
      if (headDim % 2 !== 0) {
        throw new TypeError(
          `head_dim = dim / heads (${dim} / ${heads} = ${headDim}) must be even for RoPE embeddings. ` +
          `Choose heads such that dim / heads is even.`
        );
      }
    }

    const lr = typeof config.lr === 'number' && isFinite(config.lr) && config.lr > 0 && config.lr <= 10.0
      ? config.lr
      : 1e-3;

    const batchSize = Number.isInteger(config.batchSize) && config.batchSize >= 1 && config.batchSize <= 1024
      ? config.batchSize
      : 16;

    const seqLen = Number.isInteger(config.seqLen) && config.seqLen >= 2 && config.seqLen <= 2048
      ? config.seqLen
      : 32;

    const backend = typeof config.backend === 'string'
      ? config.backend.replace(/[^a-zA-Z0-9_-]/g, '').toLowerCase()
      : 'auto';

    this.config = {
      modelType,
      dim,
      hiddenDim,
      outDim,
      loraRank,
      loraAlpha,
      vocabSize,
      layers,
      heads,
      lr,
      batchSize,
      seqLen,
      backend
    };

    this.process = null;
    this.isRunning = false;
    this._lineBuffer = '';
    this._stoppedEmitted = false;
  }

  _suggestHeads(dim) {
    const candidates = [1, 2, 4, 8, 16];
    return candidates.filter(h => dim % h === 0).join(', ');
  }

  async fit(options = {}) {
    if (this.isRunning) {
      throw new Error('Training session is already running. Concurrent execution on single device is restricted.');
    }

    const pyCmd = resolvePythonCmd();
    if (!pyCmd) {
      throw new PythonRuntimeNotFoundError();
    }

    const rawEpochs = options.epochs !== undefined ? options.epochs : 5;
    const rawLr = options.lr !== undefined ? options.lr : this.config.lr;
    const checkpointPath = options.checkpointPath ? path.resolve(String(options.checkpointPath)) : null;
    const dataPath = options.dataPath ? path.resolve(String(options.dataPath)) : null;

    if (!Number.isInteger(rawEpochs) || rawEpochs <= 0 || rawEpochs > 100000) {
      throw new TypeError(`Invalid epochs: ${rawEpochs}. Must be an integer between 1 and 100000.`);
    }

    if (typeof rawLr !== 'number' || !isFinite(rawLr) || rawLr <= 0 || rawLr > 10.0) {
      throw new TypeError(`Invalid learning rate: ${rawLr}. Must be a finite positive number <= 10.0.`);
    }

    if (dataPath && !fs.existsSync(dataPath)) {
      throw new Error(`Specified dataPath does not exist: "${dataPath}"`);
    }

    if (checkpointPath) {
      const parent = path.dirname(checkpointPath);
      try {
        if (!fs.existsSync(parent)) {
          fs.mkdirSync(parent, { recursive: true });
        }
      } catch (mkdirErr) {
        throw new InvalidCheckpointError(checkpointPath, `Cannot create parent directory: ${mkdirErr.message}`);
      }
    }

    const safeEpochs = Math.floor(rawEpochs);
    const safeLr = Number(rawLr.toPrecision(8));

    const sessionConfig = {
      ...this.config,
      epochs: safeEpochs,
      lr: safeLr,
      checkpointPath,
      dataPath
    };

    return new Promise((resolve, reject) => {
      this.isRunning = true;
      this._lineBuffer = '';
      this._stoppedEmitted = false;

      this.process = spawn(pyCmd, ['-m', 'termux_train.runtime.runner', '--stdin-json'], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.process.stdin.write(JSON.stringify(sessionConfig));
      this.process.stdin.end();

      let stderrBytes = 0;
      let stderrAcc = '';
      let checkpointInfo = null;

      this.process.stdout.on('data', (chunk) => {
        this._lineBuffer += chunk.toString('utf-8');
        let newlineIdx;
        while ((newlineIdx = this._lineBuffer.indexOf('\n')) !== -1) {
          const line = this._lineBuffer.slice(0, newlineIdx).trim();
          this._lineBuffer = this._lineBuffer.slice(newlineIdx + 1);

          if (line.startsWith('__METRICS__:')) {
            try {
              const data = JSON.parse(line.slice('__METRICS__:'.length));
              if (data.event === 'checkpoint') {
                checkpointInfo = data;
                this.emit('checkpoint', data);
              } else {
                this.emit('step', data);
                if (data.epoch) {
                  this.emit('epoch', data);
                }
              }
            } catch (parseErr) {
              this.emit('warning', `Failed to parse telemetry line: ${parseErr.message}`);
            }
          } else if (line.startsWith('__ERROR__:')) {
            this.emit('warning', `Remote backend error: ${line}`);
          }
        }
      });

      this.process.stderr.on('data', (chunk) => {
        const chunkStr = chunk.toString('utf-8');
        stderrBytes += Buffer.byteLength(chunkStr, 'utf-8');
        if (stderrBytes <= MAX_STDERR_BYTES) {
          stderrAcc += chunkStr;
        }
        // Beyond cap: silently drop to prevent heap exhaustion. Error is still propagated via exit code.
      });

      this.process.on('close', (code) => {
        this.isRunning = false;
        this.process = null;
        this._lineBuffer = '';

        if (code === 0) {
          const result = {
            status: 'SUCCESS',
            epochs: safeEpochs,
            modelType: this.config.modelType,
            checkpoint: checkpointInfo
          };
          this.emit('complete', result);
          resolve(result);
        } else {
          const err = new TrainingExecutionError(
            `Training process terminated with exit code ${code}`,
            code,
            stderrAcc
          );
          if (this.listenerCount('error') > 0) {
            this.emit('error', err);
          }
          reject(err);
        }
      });

      this.process.on('error', (procErr) => {
        this.isRunning = false;
        this.process = null;
        if (this.listenerCount('error') > 0) {
          this.emit('error', procErr);
        }
        reject(procErr);
      });
    });
  }

  async stop() {
    if (!this.process || !this.isRunning) {
      return;
    }

    const proc = this.process;
    const pid = proc.pid;

    // Guard: pid must be a valid positive integer before any kill attempt
    if (typeof pid !== 'number' || !Number.isInteger(pid) || pid <= 0) {
      this.isRunning = false;
      this.process = null;
      this._emitStopped();
      return;
    }

    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        try {
          if (process.platform === 'win32') {
            spawnSync('taskkill', ['/pid', String(pid), '/f', '/t']);
          } else {
            proc.kill('SIGKILL');
          }
        } catch (_) {}
        this.isRunning = false;
        this.process = null;
        this._lineBuffer = '';
        this._emitStopped();
        resolve();
      }, 2000);

      proc.once('close', () => {
        clearTimeout(timer);
        this.isRunning = false;
        this.process = null;
        this._lineBuffer = '';
        this._emitStopped();
        resolve();
      });

      try {
        if (process.platform === 'win32') {
          spawn('taskkill', ['/pid', String(pid), '/f', '/t']);
        } else {
          proc.kill('SIGTERM');
        }
      } catch (_) {
        try { proc.kill('SIGKILL'); } catch (_2) {}
      }
    });
  }

  /** Ensures 'stopped' is emitted exactly once per stop() call. */
  _emitStopped() {
    if (!this._stoppedEmitted) {
      this._stoppedEmitted = true;
      this.emit('stopped');
    }
  }

  /** Deterministically remove all event listeners after a session. */
  dispose() {
    this.removeAllListeners();
  }
}

module.exports = {
  TermuxTrainer
};
