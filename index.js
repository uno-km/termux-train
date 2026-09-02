/**
 * termux-train: Native On-Device Deep Learning & LoRA Training Framework
 * Dual-Engine (Python & Node.js/TypeScript) Native Module for Android Termux & ARM64.
 * Open-Source under Apache License 2.0.
 */

'use strict';

const errors = require('./lib/errors');
const doctor = require('./lib/doctor');
const benchmark = require('./lib/benchmark');
const trainer = require('./lib/trainer');
const packageJson = require('./package.json');

const version = packageJson.version;

module.exports = {
  version,
  __version__: version,
  errors,
  doctor: doctor.runDoctor,
  runDoctor: doctor.runDoctor,
  invalidatePythonCache: doctor.invalidatePythonCache,
  benchmark: benchmark.runBenchmarkAsync,
  runBenchmark: benchmark.runBenchmark,
  runBenchmarkAsync: benchmark.runBenchmarkAsync,
  TermuxTrainer: trainer.TermuxTrainer,
  Trainer: trainer.TermuxTrainer
};
