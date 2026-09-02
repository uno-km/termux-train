/**
 * termux-train: Standard Error Definitions
 * Open-Source under Apache License 2.0.
 */

'use strict';

class TermuxTrainError extends Error {
  constructor(message) {
    super(message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, this.constructor);
  }
}

class PythonRuntimeNotFoundError extends TermuxTrainError {
  constructor(msg = 'Python runtime (python3 or python) was not found in PATH.') {
    super(msg);
  }
}

class TermuxTrainModuleNotFoundError extends TermuxTrainError {
  constructor(msg = 'termux_train Python module is not installed or importable. Run "pip install termux-train".') {
    super(msg);
  }
}

class TrainingExecutionError extends TermuxTrainError {
  constructor(message, exitCode = 1, stderr = '') {
    super(message);
    this.exitCode = exitCode;
    this.stderr = stderr;
  }
}

class VulkanNotAvailableError extends TermuxTrainError {
  constructor(msg = 'Vulkan GPU acceleration is requested but not available on this device.') {
    super(msg);
  }
}

class InvalidCheckpointError extends TermuxTrainError {
  constructor(path, reason = 'Corrupted or unreadable SafeTensors checkpoint.') {
    super(`Invalid checkpoint at "${path}": ${reason}`);
    this.path = path;
  }
}

module.exports = {
  TermuxTrainError,
  PythonRuntimeNotFoundError,
  TermuxTrainModuleNotFoundError,
  TrainingExecutionError,
  VulkanNotAvailableError,
  InvalidCheckpointError
};
