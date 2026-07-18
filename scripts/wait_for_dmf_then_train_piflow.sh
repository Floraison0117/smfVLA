#!/usr/bin/env bash
# wait_for_dmf_then_train_piflow.sh
#
# 监听 DMF training 进程，结束后自动启动 Pi-Flow training。
#
#   1. 轮询 DMF 进程 (pgrep "run_train.py.*dmf_libero_plus")，直到进程消失
#   2. 等待 GPU 显存释放 (< 5GB)，防止 JAX 预分配冲突
#   3. 删除旧 Pi-Flow checkpoint (step_* 目录，保留 config.yaml)
#   4. 清理旧 piflow_train tmux session
#   5. 在新 piflow_train tmux session 中启动 Pi-Flow training
#
# 用法:
#   tmux new-session -d -s piflow_monitor \
#     "bash /root/autodl-tmp/scripts/wait_for_dmf_then_train_piflow.sh"
#   tmux attach -t piflow_monitor   # 查看监控进度
#
# 日志: logs/wait_for_dmf_then_train_piflow.log

set -euo pipefail

# ── 路径常量 ──────────────────────────────────────────────────
PROJECT_ROOT="/root/autodl-tmp"
PIFLOW_DIR="${PROJECT_ROOT}/piflow"
PIFLOW_CKPT_DIR="${PROJECT_ROOT}/checkpoints/piflow_finetuned"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/wait_for_dmf_then_train_piflow.log"

DMF_PROCESS_PATTERN="run_train.py.*dmf_libero_plus"
GPU_MEM_THRESHOLD_MB=5000   # 5GB 以下认为 GPU 已释放
DMF_POLL_INTERVAL_SEC=60    # DMF 进程轮询间隔
GPU_POLL_INTERVAL_SEC=15     # GPU 显存轮询间隔
GPU_POLL_TIMEOUT_SEC=300     # GPU 释放最多等 5min

NEW_TMUX_SESSION="piflow_train"

mkdir -p "${LOG_DIR}"

# ── 日志辅助函数 ────────────────────────────────────────────────
log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${ts}] $*" | tee -a "${LOG_FILE}"
}

# ── 获取 GPU 已用显存 (MB) ─────────────────────────────────────
gpu_mem_used_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' '
}

# ── 检查 DMF 进程是否存活 ───────────────────────────────────────
dmf_running() {
    pgrep -f "${DMF_PROCESS_PATTERN}" >/dev/null 2>&1
}

# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

log "============================================================"
log "Pi-Flow 训练监控脚本启动"
log "  DMF 进程模式: ${DMF_PROCESS_PATTERN}"
log "  GPU 显存阈值: ${GPU_MEM_THRESHOLD_MB} MB"
log "  日志文件: ${LOG_FILE}"
log "============================================================"

# ── 阶段 1: 等待 DMF 训练进程结束 ───────────────────────────────
log "[阶段 1/4] 等待 DMF 训练进程结束..."

if dmf_running; then
    log "  DMF 训练正在运行，开始轮询 (每 ${DMF_POLL_INTERVAL_SEC}s)..."

    while dmf_running; do
        local_dmf_pid="$(pgrep -f "${DMF_PROCESS_PATTERN}" | head -n1 || true)"
        local_gpu_mem="$(gpu_mem_used_mb || echo 'N/A')"
        log "  DMF 仍在运行 (PID=${local_dmf_pid:-未知}, GPU mem=${local_gpu_mem} MB)，等待..."
        sleep "${DMF_POLL_INTERVAL_SEC}"
    done

    log "  DMF 训练进程已结束。"
else
    log "  DMF 训练进程未检测到，跳过等待。"
fi

# ── 阶段 2: 等待 GPU 显存释放 ───────────────────────────────────
log "[阶段 2/4] 等待 GPU 显存释放 (< ${GPU_MEM_THRESHOLD_MB} MB)..."

gpu_wait_elapsed=0
while true; do
    mem_mb="$(gpu_mem_used_mb || echo 0)"
    if [ "${mem_mb}" -lt "${GPU_MEM_THRESHOLD_MB}" ]; then
        log "  GPU 显存已释放: ${mem_mb} MB < ${GPU_MEM_THRESHOLD_MB} MB"
        break
    fi

    if [ "${gpu_wait_elapsed}" -ge "${GPU_POLL_TIMEOUT_SEC}" ]; then
        log "  ⚠️  GPU 显存释放超时 (${gpu_wait_elapsed}s)，当前 ${mem_mb} MB。"
        log "  强制继续启动 Pi-Flow 训练 (可能 OOM)。"
        break
    fi

    log "  GPU 显存仍占用: ${mem_mb} MB，等待释放 (已等 ${gpu_wait_elapsed}s)..."
    sleep "${GPU_POLL_INTERVAL_SEC}"
    gpu_wait_elapsed=$((gpu_wait_elapsed + GPU_POLL_INTERVAL_SEC))
done

# ── 阶段 3: 删除旧 Pi-Flow checkpoint ───────────────────────────
log "[阶段 3/4] 删除旧 Pi-Flow checkpoint..."

if [ -d "${PIFLOW_CKPT_DIR}" ]; then
    # 删除所有 step_* 目录 (旧 checkpoint)，保留 config.yaml
    deleted_count=0
    shopt -s nullglob
    for d in "${PIFLOW_CKPT_DIR}"/step_*; do
        if [ -d "$d" ]; then
            log "  删除: $(basename "$d")"
            rm -rf "$d"
            deleted_count=$((deleted_count + 1))
        fi
    done
    shopt -u nullglob

    if [ "${deleted_count}" -eq 0 ]; then
        log "  无旧 checkpoint 目录需要删除。"
    else
        log "  共删除 ${deleted_count} 个旧 checkpoint 目录。"
    fi
else
    log "  checkpoint 目录不存在: ${PIFLOW_CKPT_DIR}，跳过删除。"
fi

# ── 阶段 4: 启动 Pi-Flow 训练 (tmux) ───────────────────────────
log "[阶段 4/4] 启动 Pi-Flow 训练..."

# 清理旧 tmux session
if tmux has-session -t "${NEW_TMUX_SESSION}" 2>/dev/null; then
    log "  旧 tmux session '${NEW_TMUX_SESSION}' 存在，先清理。"
    tmux kill-session -t "${NEW_TMUX_SESSION}"
    log "  已清理旧 session。"
fi

# 启动新训练 session
tmux new-session -d -s "${NEW_TMUX_SESSION}" -c "${PIFLOW_DIR}" \
    "bash scripts/train.sh 2>&1 | tee ${LOG_DIR}/train_piflow_state_norm_fix.log"

log "  Pi-Flow 训练已在新 tmux session '${NEW_TMUX_SESSION}' 中启动。"
log "  训练日志: ${LOG_DIR}/train_piflow_state_norm_fix.log"
log "  查看训练: tmux attach -t ${NEW_TMUX_SESSION}"
log ""
log "============================================================"
log "监控脚本完成。Pi-Flow 训练正在后台运行。"
log "============================================================"

# 脚本退出，piflow_train session 继续在后台运行
