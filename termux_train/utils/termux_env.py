"""
termux_train.utils.termux_env
=============================
Platform and Operating Environment Identification for Android Termux.

변경 이력:
- [PATCH] vulkan_available 체크 추가: /system/lib64/libvulkan.so 존재 여부 탐지.
  ameva-runtime 설치 전 사전 환경 점검 시 사용됩니다.
- [PATCH] get_cpu_info() / get_storage_info() 의 except:pass 묵음 패턴 제거.
  모든 실패는 [termux-train] 태그와 함께 logging.debug 에 기록됩니다.
"""

import logging
import os
import platform
import shutil
import sys
from typing import Any, Dict

logger = logging.getLogger("termux_train.utils.termux_env")


# [B방안] Platform SSOT: ameva-runtime.platform 에서 공유 구현을 가져옵니다.
try:
    from ameva_runtime.vulkan.platform import (
        is_android as _ameva_is_android,
        is_termux as _ameva_is_termux,
    )
    _AMEVA_PLATFORM_AVAILABLE = True
except ImportError:
    _AMEVA_PLATFORM_AVAILABLE = False


def is_android() -> bool:
    """Check if the current runtime is running on Android.

    [B방안] ameva-runtime.platform.is_android() 를 SSOT 로 사용합니다.
    """
    if _AMEVA_PLATFORM_AVAILABLE:
        return _ameva_is_android()
    if os.path.exists("/system/build.prop"):
        return True
    if "ANDROID_ROOT" in os.environ or "ANDROID_DATA" in os.environ:
        return True
    return False


def is_termux() -> bool:
    """Check if running specifically inside the Termux native environment.

    [B방안] ameva-runtime.platform.is_termux() 를 SSOT 로 사용합니다.
    """
    if _AMEVA_PLATFORM_AVAILABLE:
        return _ameva_is_termux()
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix or "TERMUX_VERSION" in os.environ:
        return True
    if os.path.exists("/data/data/com.termux"):
        return True
    return False


def check_vulkan_available() -> bool:
    """Android Bionic ICD 기반 Vulkan 드라이버 가용성을 확인합니다.

    탐색 순서는 ameva-runtime 의 ICD 정책과 동일합니다:
    시스템 ICD 우선, Termux Mesa fallback.

    Returns:
        True — /system/lib64/libvulkan.so 또는 /vendor/lib64/libvulkan.so 존재.
        False — Vulkan ICD 파일 미존재 또는 접근 불가.
    """
    # [중요] Termux Mesa($PREFIX/lib/libvulkan.so)는 여기서 체크하지 않습니다.
    # Bionic ICD 와 Mesa 를 동시에 로드하면 SIGABRT 충돌이 발생합니다.
    # 시스템 ICD 만 확인하고, 없으면 Vulkan 불가로 판단합니다.
    system_icd_paths = [
        "/system/lib64/libvulkan.so",
        "/system/lib/libvulkan.so",
        "/vendor/lib64/libvulkan.so",
        "/vendor/lib/libvulkan.so",
    ]
    for path in system_icd_paths:
        try:
            if os.path.isfile(path):
                # 파일 크기 최소 검증 (1KB 미만은 스텁 파일로 간주)
                if os.path.getsize(path) >= 1024:
                    return True
        except OSError as e:
            logger.debug(
                "[termux-train] Vulkan ICD 파일 접근 실패: path=%s, error=%s — SELinux 제한일 수 있습니다.",
                path, e
            )
    return False


def get_cpu_info() -> Dict[str, Any]:
    """Inspect CPU architecture, core count, and processor model."""
    info: Dict[str, Any] = {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cores_logical": os.cpu_count() or 1,
        "is_arm64": platform.machine().lower() in ("aarch64", "arm64"),
    }

    # Try reading /proc/cpuinfo on Linux/Android
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                info["cpuinfo_snippet"] = content[:300].strip()
        except Exception as e:
            logger.debug(
                "[termux-train] /proc/cpuinfo 읽기 실패 (비 Linux 환경에서 정상): %s", e
            )
    return info


def get_storage_info() -> Dict[str, Any]:
    """Inspect disk space in current working directory."""
    try:
        usage = shutil.disk_usage(os.getcwd())
        return {
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "used_gb": round(usage.used / (1024 ** 3), 2),
            "free_gb": round(usage.free / (1024 ** 3), 2),
        }
    except Exception as e:
        logger.debug(
            "[termux-train] 디스크 사용량 조회 실패 (제한된 환경에서 정상): %s", e
        )
        return {"total_gb": None, "free_gb": None}


def get_device_info() -> Dict[str, Any]:
    """Platform and OS environment metadata dictionary.

    반환 키:
        is_android          : Android OS 여부
        is_termux           : Termux 네이티브 환경 여부
        platform            : 플랫폼 문자열
        python_version      : Python 버전
        python_compiler     : Python 컴파일러
        cpu                 : CPU 정보 dict (architecture, cores, is_arm64 등)
        storage             : 저장소 정보 dict (total_gb, used_gb, free_gb)
        opencl_available    : OpenCL ICD 파일 존재 여부
        vulkan_available    : Android Bionic Vulkan ICD 파일 존재 여부
                              (ameva-runtime 설치 전 사전 환경 점검용)
    """
    return {
        "is_android": is_android(),
        "is_termux": is_termux(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_compiler": platform.python_compiler(),
        "cpu": get_cpu_info(),
        "storage": get_storage_info(),
        "opencl_available": (
            os.path.exists("/system/vendor/lib64/libOpenCL.so")
            or os.path.exists("/vendor/lib64/libOpenCL.so")
        ),
        # [신규] Vulkan ICD 가용성 — ameva-runtime 연동 전제조건 확인
        "vulkan_available": check_vulkan_available(),
    }
