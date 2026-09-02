#!/usr/bin/env node

/**
 * termux-train: Node.js Global CLI Binary.
 * High-performance on-device Deep Learning & LoRA Training Framework.
 * Open-Source under Apache License 2.0.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const { version } = require('../package.json');
const { runDoctor, resolvePythonCmd } = require('../lib/doctor');
const { runBenchmark } = require('../lib/benchmark');
const { TermuxTrainer } = require('../lib/trainer');
const {
  PythonRuntimeNotFoundError,
  TermuxTrainModuleNotFoundError
} = require('../lib/errors');

const args = process.argv.slice(2);
const command = args[0] || '--help';

function getArg(flag, alias = null) {
  for (let i = 0; i < args.length; i++) {
    const item = args[i];
    if (item.startsWith(`${flag}=`)) {
      return item.slice(flag.length + 1);
    }
    if (alias && item.startsWith(`${alias}=`)) {
      return item.slice(alias.length + 1);
    }
    if (item === flag || (alias && item === alias)) {
      if (i + 1 < args.length && !args[i + 1].startsWith('-')) {
        return args[i + 1];
      }
    }
  }
  return null;
}

function hasFlag(flag, alias = null) {
  return args.includes(flag) || (alias && args.includes(alias));
}

function printHelp() {
  console.log(`termux-train CLI v${version} (Node.js Engine)`);
  console.log('Usage: termux-train <command> [options]\n');
  console.log('Commands:');
  console.log('  doctor                           Inspect device hardware, RAM tier, and Vulkan GPU');
  console.log('  check                            Run self-diagnostic mathematical checks across backends');
  console.log('  score                            Run 0-point baseline granular audit scoring system');
  console.log('  benchmark [options]              Run on-device GEMM & Autograd latency benchmark');
  console.log('  train [options]                  Run on-device training / LoRA loop');
  console.log('  demo <1..8>                      Execute one of 8 canonical example demos\n');
  console.log('Options:');
  console.log('  --json                           Output results in standard JSON format');
  console.log('  --dim <num>                      Matrix dimension for benchmark (default: 256)');
  console.log('  --data <path>                    Path to dataset file (.safetensors, .jsonl, .txt)');
  console.log('  --epochs <num>                   Number of training epochs (default: 5)');
  console.log('  --lr <val>                       Learning rate (default: 0.001)');
  console.log('  -v, --version                    Display version information');
  console.log('  -h, --help                       Show this help message');
}

async function main() {
  if (hasFlag('-v') || hasFlag('--version')) {
    console.log(`termux-train ${version}`);
    process.exit(0);
  }

  if (command === '--help' || command === '-h' || command === 'help') {
    printHelp();
    process.exit(0);
  }

  if (command === 'doctor') {
    const isJson = hasFlag('--json');
    const rep = runDoctor();

    if (isJson) {
      console.log(JSON.stringify(rep, null, 2));
    } else {
      console.log('=== termux-train Diagnostic Doctor (Node.js Engine) ===');
      console.log(`  Platform       : ${rep.platform.system} (${rep.platform.arch}) | Android/Termux: ${rep.platform.isAndroid || rep.platform.isTermux}`);
      console.log(`  Hardware Tier  : ${rep.hardware.tier} (RAM: ${rep.hardware.totalRamMb}MB | Cores: ${rep.hardware.cpuCores})`);
      console.log(`  Python Runtime : ${rep.python.available ? rep.python.version : 'NOT FOUND'} (${rep.python.moduleInstalled ? 'termux_train installed' : 'module missing'})`);
      console.log(`  Vulkan GPU     : ${rep.vulkan.status}`);
      console.log(`  Preset Config  : Max LoRA Rank r=${rep.recommendedTrainingConfig.loraRank} | Batch Size=${rep.recommendedTrainingConfig.batchSize}`);
      console.log(`  Overall Status : ${rep.status}`);
    }
    process.exit(0);
  }

  if (command === 'benchmark') {
    const isJson = hasFlag('--json');
    const dim = parseInt(getArg('--dim') || '256', 10);
    try {
      const res = runBenchmark({ dim });
      if (isJson) {
        console.log(JSON.stringify(res, null, 2));
      } else {
        console.log(`=== termux-train Benchmark (Dimension: ${res.dimension}) ===`);
        console.log(`  Backend                   : ${res.backend}`);
        console.log(`  Forward GEMM Latency      : ${res.gemmLatencyMs} ms`);
        console.log(`  Full Autograd Step (F+B)  : ${res.autogradStepLatencyMs} ms`);
        console.log(`  Throughput                : ${res.throughputGflops} GFLOPS`);
      }
      process.exit(0);
    } catch (err) {
      console.error(`[ERROR] Benchmark failed: ${err.message}`);
      process.exit(1);
    }
  }

  if (command === 'train') {
    const epochs = parseInt(getArg('--epochs') || '5', 10);
    const lr = parseFloat(getArg('--lr') || '0.001');
    const modelType = getArg('--model') || 'mlp';
    const loraRank = parseInt(getArg('--rank') || '8', 10);
    const dim = parseInt(getArg('--dim') || '32', 10);
    const checkpoint = getArg('--checkpoint') || null;
    const data = getArg('--data') || getArg('--dataPath') || null;

    if (data && !fs.existsSync(path.resolve(data))) {
      console.error(`[ERROR] Specified dataset path does not exist: "${data}"`);
      process.exit(1);
    }

    console.log(`[*] Initializing TermuxTrainer (model=${modelType}, dim=${dim}, rank=${loraRank}, epochs=${epochs}, lr=${lr}, data=${data || 'synthetic'})...`);
    const trainer = new TermuxTrainer({
      modelType,
      dim,
      loraRank,
      lr
    });

    trainer.on('step', (m) => {
      console.log(`  [Epoch ${m.epoch}/${m.totalEpochs}] Loss: ${m.loss.toFixed(6)} (${m.latencyMs} ms)`);
    });

    trainer.on('checkpoint', (ckpt) => {
      console.log(`  [Checkpoint] Saved SafeTensors to: ${ckpt.path} (${ckpt.sizeBytes} bytes)`);
    });

    try {
      const res = await trainer.fit({
        epochs,
        checkpointPath: checkpoint,
        dataPath: data ? path.resolve(data) : null
      });
      console.log(`[+] Training completed successfully in ${res.epochs} epochs.`);
      trainer.dispose();
      process.exit(0);
    } catch (err) {
      console.error(`[ERROR] Training failed: ${err.message}`);
      trainer.dispose();
      process.exit(1);
    }
  }

  if (command === 'check' || command === 'score' || command === 'demo') {
    const pyCmd = resolvePythonCmd();
    if (!pyCmd) {
      console.error('[ERROR] Python runtime is required to execute native self-tests.');
      process.exit(1);
    }
    const pyArgs = ['-m', 'termux_train.cli', command, ...args.slice(1)];
    const res = spawnSync(pyCmd, pyArgs, { stdio: 'inherit' });
    process.exit(res.status || 0);
  }

  console.error(`[ERROR] Unknown command: '${command}'`);
  printHelp();
  process.exit(2);
}

main().catch((err) => {
  console.error(`[FATAL] ${err.message}`);
  process.exit(1);
});
