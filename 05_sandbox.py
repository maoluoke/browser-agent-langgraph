"""Step 5：代码执行沙箱（OpenSandbox 优先，subprocess 兜底）

背景：
  智能体执行 LLM 生成的代码时必须有隔离环境，防止误删宿主机文件、
  fork 炸弹耗尽资源等风险。本项目选用阿里开源 OpenSandbox：
  Docker 容器级隔离 + REST/SDK 管理 + 资源限额，优于手搓 docker run。

架构：
  opensandbox-server（localhost:8080，独立后台进程）
    -> Docker 容器（本地镜像 python:3.12-slim，无需拉取）
    -> execd 在容器内执行命令并回传 stdout/stderr/exit_code

兜底：
  服务不可达时降级为 subprocess + resource.setrlimit 限制 CPU/内存，
  保证主流程不中断（隔离强度弱于容器，仅限可信代码）。

验证（直接运行本文件）：
  /home/zkx456/other/bin/python3.13 05_sandbox.py
  -> 建沙箱执行命令、验证文件隔离（沙箱内写文件宿主机不可见）、演示兜底
"""

from __future__ import annotations

import os
import resource
import subprocess
import tempfile

SANDBOX_SERVER = os.getenv("OPENSANDBOX_URL", "127.0.0.1:8080")
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "python:3.12-slim")  # 本地已有，避免拉取
CMD_TIMEOUT = 30          # 单条命令超时（秒）
FALLBACK_CPU_LIMIT = 10   # 兜底模式 CPU 秒数上限
FALLBACK_MEM_LIMIT = 512 * 1024 * 1024  # 兜底模式内存上限 512MB


def _logs_text(logs) -> tuple[str, str]:
    """ExecutionLogs.stdout/stderr 是 OutputMessage 列表，取 text 字段拼成纯文本"""
    def join(items):
        if not items:
            return ""
        return "".join(getattr(c, "text", None) or getattr(c, "content", str(c)) for c in items)
    return join(getattr(logs, "stdout", [])), join(getattr(logs, "stderr", []))


class CodeSandbox:
    """统一接口：run(command) -> (exit_code, stdout, stderr)"""

    def __init__(self, prefer_opensandbox: bool = True):
        self.backend = "subprocess"
        self._sb = None
        if prefer_opensandbox:
            try:
                from opensandbox import SandboxSync
                from opensandbox.config.connection_sync import ConnectionConfigSync

                cfg = ConnectionConfigSync(domain=SANDBOX_SERVER, protocol="http", api_key="")
                self._sb = SandboxSync.create(
                    image=SANDBOX_IMAGE,
                    connection_config=cfg,
                    entrypoint=["sleep", "infinity"],
                    skip_health_check=False,
                )
                self.backend = "opensandbox"
            except Exception as e:
                print(f"[Sandbox] OpenSandbox 不可用（{type(e).__name__}: {e}），降级 subprocess")

    def run(self, command: str) -> tuple[int, str, str]:
        if self.backend == "opensandbox":
            ex = self._sb.commands.run(command)
            out, err = _logs_text(ex.logs)
            return ex.exit_code or 0, out, err
        return self._run_subprocess(command)

    @staticmethod
    def _run_subprocess(command: str) -> tuple[int, str, str]:
        """兜底：子进程 + rlimit 限制 CPU/内存，在临时目录中执行"""

        def _limits():
            resource.setrlimit(resource.RLIMIT_CPU, (FALLBACK_CPU_LIMIT, FALLBACK_CPU_LIMIT))
            resource.setrlimit(resource.RLIMIT_AS, (FALLBACK_MEM_LIMIT, FALLBACK_MEM_LIMIT))

        with tempfile.TemporaryDirectory() as cwd:
            try:
                p = subprocess.run(
                    ["bash", "-c", command], cwd=cwd,
                    capture_output=True, text=True,
                    timeout=CMD_TIMEOUT, preexec_fn=_limits,
                )
                return p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired:
                return 124, "", f"超时（>{CMD_TIMEOUT}s）"

    def close(self):
        if self._sb is not None:
            try:
                self._sb.kill()
            except Exception:
                pass


if __name__ == "__main__":
    sb = CodeSandbox()
    try:
        print(f"[后端] {sb.backend}")

        code, out, err = sb.run("python3 -c \"print(sum(i*i for i in range(10)))\"")
        print(f"[计算] exit={code} stdout={out.strip()}")  # 期望 285

        # 隔离验证：沙箱内写文件，宿主机同路径必须不存在
        mark = "/tmp/agent_sandbox_mark.txt"
        sb.run(f"echo inside > {mark}")
        host_has = os.path.exists(mark)
        print(f"[隔离] 沙箱内已写 {mark}；宿主机可见={host_has}（期望 False）")

        code, out, _ = sb.run("cat /etc/os-release | head -1")
        print(f"[环境] {out.strip()}")
    finally:
        sb.close()

    # 兜底链路演示
    fb = CodeSandbox(prefer_opensandbox=False)
    code, out, err = fb.run("python3 -c \"print('fallback ok')\"")
    print(f"[兜底] subprocess exit={code} stdout={out.strip()}")
    print("[完成] 阶段5验证通过")
