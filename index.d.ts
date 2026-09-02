/**
 * Type declarations for termux-train Node.js SDK
 * Open-Source under Apache License 2.0.
 */

import { EventEmitter } from 'events';

export const version: string;
export const __version__: string;

export interface DoctorPlatformInfo {
  system: string;
  arch: string;
  isTermux: boolean;
  isAndroid: boolean;
}

export interface DoctorHardwareInfo {
  cpuCores: number;
  totalRamMb: number;
  freeRamMb: number;
  tier: string;
}

export interface DoctorPythonInfo {
  available: boolean;
  version: string;
  binary: string | null;
  moduleInstalled: boolean;
}

export interface DoctorVulkanInfo {
  gpuAcceleration: boolean;
  status: 'AVAILABLE' | 'CPU_FALLBACK';
}

export interface RecommendedTrainingConfig {
  loraRank: number;
  batchSize: number;
  seqLen: number;
  safeTensorsMmap: boolean;
}

export interface DoctorReport {
  framework: 'termux-train';
  version: string;
  platform: DoctorPlatformInfo;
  hardware: DoctorHardwareInfo;
  python: DoctorPythonInfo;
  vulkan: DoctorVulkanInfo;
  recommendedTrainingConfig: RecommendedTrainingConfig;
  status: 'READY' | 'SETUP_REQUIRED';
}

export interface BenchmarkOptions {
  dim?: number;
  iters?: number;
}

export interface BenchmarkResult {
  dimension: string;
  iterations: number;
  backend: string;
  gemmLatencyMs: number;
  autogradStepLatencyMs: number;
  throughputGflops: number;
}

export interface TrainerConfig {
  modelType?: 'mlp' | 'transformer' | 'transformer-lm' | 'lora' | 'linear-lora' | 'rope';
  dim?: number;
  hiddenDim?: number;
  outDim?: number;
  loraRank?: number;
  loraAlpha?: number;
  vocabSize?: number;
  layers?: number;
  heads?: number;
  lr?: number;
  batchSize?: number;
  seqLen?: number;
  backend?: 'auto' | 'cpu' | 'numpy' | 'vulkan';
}

export interface FitOptions {
  epochs?: number;
  lr?: number;
  checkpointPath?: string;
  dataPath?: string;
}

export interface TrainingMetrics {
  event: 'step' | 'epoch';
  epoch: number;
  totalEpochs: number;
  loss: number;
  latencyMs: number;
}

export interface CheckpointInfo {
  event: 'checkpoint';
  path: string;
  tensorsSaved: number;
  sizeBytes: number;
}

export interface FitResult {
  status: 'SUCCESS' | 'FAILED';
  epochs: number;
  modelType: string;
  checkpoint: CheckpointInfo | null;
}

export class TermuxTrainer extends EventEmitter {
  constructor(config?: TrainerConfig);
  fit(options?: FitOptions): Promise<FitResult>;
  stop(): Promise<void>;

  on(event: 'step', listener: (metrics: TrainingMetrics) => void): this;
  on(event: 'epoch', listener: (metrics: TrainingMetrics) => void): this;
  on(event: 'checkpoint', listener: (info: CheckpointInfo) => void): this;
  on(event: 'complete', listener: (res: FitResult) => void): this;
  on(event: 'warning', listener: (msg: string) => void): this;
  on(event: 'error', listener: (err: Error) => void): this;
  on(event: 'stopped', listener: () => void): this;
}

export const Trainer: typeof TermuxTrainer;

export function doctor(options?: Record<string, unknown>): DoctorReport;
export function runDoctor(options?: Record<string, unknown>): DoctorReport;
export function benchmark(options?: BenchmarkOptions): Promise<BenchmarkResult>;
export function runBenchmark(options?: BenchmarkOptions): BenchmarkResult;
export function runBenchmarkAsync(options?: BenchmarkOptions): Promise<BenchmarkResult>;

export namespace errors {
  export class TermuxTrainError extends Error {}
  export class PythonRuntimeNotFoundError extends TermuxTrainError {}
  export class TermuxTrainModuleNotFoundError extends TermuxTrainError {}
  export class TrainingExecutionError extends TermuxTrainError {
    exitCode: number;
    stderr: string;
  }
  export class VulkanNotAvailableError extends TermuxTrainError {}
  export class InvalidCheckpointError extends TermuxTrainError {
    path: string;
  }
}
