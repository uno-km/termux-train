# 📱 Android Termux SSH Connection Guide

본 가이드는 Android 스마트폰/태블릿 기기의 Termux를 PC와 무선으로 연결하여 개발 및 학습을 진행하기 위한 **SSH 접속 세팅 절차**입니다.

---

## 🛠️ 1. 스마트폰(Android Termux) 세팅

1. **Termux 실행 후 SSH 및 빌드 툴 설치**:
   ```bash
   pkg update && pkg upgrade -y
   pkg install openssh git python -y
   ```
2. **비밀번호 설정 (최초 1회)**:
   ```bash
   passwd
   ```
3. **SSH 서버 실행**:
   ```bash
   sshd
   ```
   *(기본 포트는 `8022`)*
4. **사용자명 및 IP 확인**:
   ```bash
   whoami
   # 출력 예: u0_a123
   
   ifconfig
   # wlan0 항목의 inet 주소 확인 (예: 192.168.0.50)
   ```

---

## 💻 2. PC(Windows / VSCode / PowerShell) 접속

1. **PowerShell 또는 터미널에서 접속**:
   ```bash
   ssh -p 8022 u0_a123@192.168.0.50
   ```
2. **VSCode Remote - SSH 연동**:
   - `~/.ssh/config`에 아래 항목 추가:
     ```text
     Host termux-device
         HostName 192.168.0.50
         Port 8022
         User u0_a123
     ```
   - VSCode에서 `Remote-SSH: Connect to Host...` $\rightarrow$ `termux-device` 선택 시 실시간 원격 개발 가능.
