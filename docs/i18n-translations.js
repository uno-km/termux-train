/**
 * termux-train - Official Documentation Translation Dictionary (AMEVA Ecosystem Design)
 * Languages: English (en), Korean (ko), Japanese (ja), Chinese (zh), Spanish (es), Hindi (hi)
 */

(function(global) {
  'use strict';

  const translations = {
    "en": {
      "common": {
        "brand": "termux-train",
        "releaseTag": "v0.1.0-alpha (Native)",
        "pypiBtn": "PyPI Package",
        "githubBtn": "GitHub Repo",
        "nav": {
          "overview": "Overview",
          "home": "Home / Architecture",
          "installation": "Installation Guide (pip)",
          "quickstart": "Quickstart & Recipes",
          "models": "Tiny Models & LoRA Hub",
          "guide": "Training Manual & Recipes",
          "apiReference": "100% Full API Reference",
          "advancedParams": "Mobile Memory & INT8",
          "benchmarks": "Benchmarks & Hardware",
          "versions": "Version Archive",
          "advanced": "AI Agent Specifications"
        },
        "footerText": "© 2026 termux-train Project (uno-km). Released under Apache License 2.0."
      },
      "home": {
        "title": "termux-train",
        "subtitle": "Native On-Device Deep Learning & LoRA Training Framework for Android Termux (ARM64 Bionic)",
        "whyTitle": "The Mobile Deep Learning Challenge",
        "whyText": "Standard PyTorch binaries fail on Android Termux due to GNU Glibc vs Android Bionic Libc mismatch, while PRoot Linux containers add 40% memory overhead and trigger Android LMK (Low Memory Killer) aborts.",
        "solTitle": "The Zero-Dependency Breakthrough",
        "solText": "termux-train runs natively on Android Termux with a pure Python DAG Autograd core, pluggable NumPy/OpenBLAS ARM NEON SIMD vectorization, RoPE Transformers, SafeTensors zero-copy, and low-rank LoRA fine-tuning.",
        "capTitle": "Key Capabilities & Mobile Architecture",
        "codeExampleTitle": "Canonical 10-Line Training Demo (Python)"
      },
      "installation": {
        "title": "Installation Guide (PyPI & Termux)",
        "subtitle": "Step-by-step setup instructions for Android Termux, Linux, Windows, and macOS."
      },
      "quickstart": {
        "title": "Quickstart & Practical Recipes",
        "subtitle": "Build, train, and recover neural networks with crash-resilient checkpoints."
      },
      "models": {
        "title": "Tiny Models & LoRA Hub",
        "subtitle": "Pre-configured architectures for on-device Transformers, Whisper speech recognition, and low-rank adapters."
      },
      "guide": {
        "title": "On-Device Tiny Model & Small LLM Training Manual",
        "subtitle": "Comprehensive engineering guide for training RoPE Transformers, Whisper LoRA, and DocFold models on mobile hardware."
      },
      "api": {
        "title": "100% Full API Reference",
        "subtitle": "Complete specification of Tensor, Module, Optimizer, Checkpoint, and Tokenizer classes."
      },
      "advancedParams": {
        "title": "Mobile Memory Management & INT8 Quantization",
        "subtitle": "Guidelines for MMap disk streaming, SafeTensors zero-copy serialization, and LMK defense."
      },
      "benchmarks": {
        "title": "Performance & Hardware Benchmarks",
        "subtitle": "Empirical latency, throughput, and memory consumption across mobile CPUs."
      },
      "versions": {
        "title": "Version Archive & Changelog",
        "subtitle": "Release history, changelog, and roadmap milestones."
      }
    },
    "ko": {
      "common": {
        "brand": "termux-train",
        "releaseTag": "v0.1.0-alpha (네이티브)",
        "pypiBtn": "PyPI 패키지",
        "githubBtn": "GitHub 저장소",
        "nav": {
          "overview": "문서 목차",
          "home": "홈 / 아키텍처",
          "installation": "설치 가이드 (pip)",
          "quickstart": "퀵스타트 & 레시피",
          "models": "소형 모델 & LoRA 허브",
          "guide": "훈련 매뉴얼 & 실전 레시피",
          "apiReference": "100% 전수 API 레퍼런스",
          "advancedParams": "모바일 메모리 & INT8",
          "benchmarks": "성능 벤치마크",
          "versions": "버전 릴리즈 아카이브",
          "advanced": "AI 에이전트 명세"
        },
        "footerText": "© 2026 termux-train 프로젝트 (uno-km). Apache License 2.0 라이선스로 배포됩니다."
      },
      "home": {
        "title": "termux-train",
        "subtitle": "안드로이드 Termux 환경을 위한 네이티브 온디바이스 딥러닝 & LoRA 훈련 프레임워크 (ARM64 Bionic)",
        "whyTitle": "모바일 딥러닝 환경의 한계와 도전",
        "whyText": "공식 파이토치는 Glibc 기반으로 빌드되어 안드로이드 Bionic Libc에서 심볼 오류로 실행되지 않으며, PRoot 가상화는 40% 이상의 메모리 낭비와 LMK(Low Memory Killer) 강제 종료를 유발합니다.",
        "solTitle": "Zero-Dependency 기술 혁신",
        "solText": "termux-train은 순수 파이썬 DAG 오토그래드 코어와 C-가속 NumPy(OpenBLAS NEON), RoPE 트랜스포머, SafeTensors 제로-카피, LoRA 저순위 어댑터로 순정 Termux에서 100% 네이티브로 동작합니다.",
        "capTitle": "핵심 기능 및 모바일 최적화 아키텍처",
        "codeExampleTitle": "대표 10줄 훈련 코드 예제 (Python)"
      },
      "installation": {
        "title": "설치 가이드 (PyPI & Termux)",
        "subtitle": "안드로이드 Termux, 리눅스, 윈도우, 맥 환경별 단계별 설치 안내."
      },
      "quickstart": {
        "title": "5분 퀵스타트 & 실전 레시피",
        "subtitle": "원자적 체크포인트와 함께 신경망을 즉시 구축하고 훈련하는 실전 예제."
      },
      "models": {
        "title": "소형 모델 & LoRA 허브",
        "subtitle": "RoPE 트랜스포머, 음성인식 Tiny Whisper, 저순위 어댑터 프리셋."
      },
      "guide": {
        "title": "소형 모델 & LLM 훈련 종합 매뉴얼",
        "subtitle": "스마트폰 환경에서 RoPE 트랜스포머 언어모델, 위스퍼 LoRA, DocFold를 훈련하는 전 과정 가이드."
      },
      "api": {
        "title": "100% 전수 API 레퍼런스",
        "subtitle": "Tensor, nn, optim, checkpoint, tokenization 전수 API 명세."
      },
      "advancedParams": {
        "title": "모바일 메모리 관리 & INT8 양자화",
        "subtitle": "MMap 디스크 스트리밍, SafeTensors 직렬화 및 LMK 방어 전략."
      },
      "benchmarks": {
        "title": "성능 & 하드웨어 벤치마크",
        "subtitle": "0점 기준 감사 스코어카드(100점 만점) 및 가속 백엔드 실측 지표."
      },
      "versions": {
        "title": "버전 릴리즈 아카이브 & 체인지로그",
        "subtitle": "v0.1.0-alpha 릴리즈 노트 및 스프린트 개발 이력."
      }
    },
    "ja": {
      "common": {
        "brand": "termux-train",
        "releaseTag": "v0.1.0-alpha (ネイティブ)",
        "pypiBtn": "PyPI パッケージ",
        "githubBtn": "GitHub",
        "nav": {
          "overview": "目次",
          "home": "ホーム / アーキテクチャ",
          "installation": "インストールガイド (pip)",
          "quickstart": "クイックスタート",
          "models": "軽量モデル & LoRA",
          "guide": "訓練マニュアル & レシピ",
          "apiReference": "100% 完全 API リファレンス",
          "advancedParams": "メモリ管理 & INT8",
          "benchmarks": "ベンチマーク",
          "versions": "バージョン履歴",
          "advanced": "AI エージェント仕様"
        },
        "footerText": "© 2026 termux-train プロジェクト (uno-km). Apache License 2.0。"
      }
    },
    "zh": {
      "common": {
        "brand": "termux-train",
        "releaseTag": "v0.1.0-alpha (原生)",
        "pypiBtn": "PyPI 软件包",
        "githubBtn": "GitHub 仓库",
        "nav": {
          "overview": "概览",
          "home": "首页 / 架构",
          "installation": "安装指南 (pip)",
          "quickstart": "快速上手与示例",
          "models": "轻量模型与 LoRA",
          "guide": "端侧训练指南与手册",
          "apiReference": "100% 完整 API 参考",
          "advancedParams": "内存管理与 INT8",
          "benchmarks": "性能基准测试",
          "versions": "版本发布归档",
          "advanced": "AI 代理协议"
        },
        "footerText": "© 2026 termux-train 项目 (uno-km). 遵循 Apache License 2.0 协议。"
      }
    },
    "es": {
      "common": {
        "brand": "termux-train",
        "releaseTag": "v0.1.0-alpha (Nativo)",
        "pypiBtn": "Paquete PyPI",
        "githubBtn": "GitHub",
        "nav": {
          "overview": "Visión General",
          "home": "Inicio / Arquitectura",
          "installation": "Guía de Instalación (pip)",
          "quickstart": "Guía Rápida",
          "models": "Modelos y LoRA",
          "guide": "Manual de Entrenamiento",
          "apiReference": "Referencia de API 100%",
          "advancedParams": "Memoria e INT8",
          "benchmarks": "Rendimiento",
          "versions": "Versiones",
          "advanced": "Protocolo IA"
        },
        "footerText": "© 2026 termux-train Proyecto (uno-km). Apache License 2.0."
      }
    },
    "hi": {
      "common": {
        "brand": "termux-train",
        "releaseTag": "v0.1.0-alpha (नेटिव)",
        "pypiBtn": "PyPI पैकेज",
        "githubBtn": "गिटहब",
        "nav": {
          "overview": "अवलोकन",
          "home": "होम / आर्किटेक्चर",
          "installation": "स्थापना (pip)",
          "quickstart": "त्वरित शुरुआत",
          "models": "मॉडल और LoRA",
          "guide": "प्रशिक्षण मैनुअल",
          "apiReference": "100% संपूर्ण API संदर्भ",
          "advancedParams": "मेमोरी और INT8",
          "benchmarks": "बेंचमार्क",
          "versions": "रिलीज़ आर्काइव",
          "advanced": "AI एजेंट प्रोटोकॉल"
        },
        "footerText": "© 2026 termux-train परियोजना (uno-km). Apache License 2.0।"
      }
    }
  };

  if (global.I18n) {
    global.I18n.registerTranslations(translations);
  }
})(window);
