import logging
import os
import platform
import re
import time
import uuid
import docker
import psutil
import requests
from filelock import FileLock
from pathlib import Path
import io
import os
import tarfile
from pathlib import Path

import getpass
from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.docker.DockerProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
RETRY_INTERVAL = 1
# 16 actors per node start simultaneously; each Docker start can take up to 60s,
# so give the queue enough headroom for all of them.
LOCK_TIMEOUT = 120


class PortAllocationError(Exception):
    pass


class DockerProvider(Provider):
    def __init__(self, region: str):
        self.client = docker.from_env()
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.container = None
        self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}  # Modify if needed

        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        self.lock_file = temp_dir / "docker_port_allocations.lck"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_container_labels(self) -> dict[str, str]:
        labels = {"osworld.managed": "true"}
        tag = os.getenv("OSWORLD_DOCKER_TAG", "").strip()
        if tag:
            labels["osworld.tag"] = tag
        # run_group is a stable cross-run label (e.g. "replay-verify-youruser").
        # Used to cleanup all containers belonging to this pipeline/user on startup
        # and after Ctrl+C, while leaving other users' osworld containers untouched.
        run_group = os.getenv("OSWORLD_RUN_GROUP", "").strip()
        if run_group:
            labels["osworld.run_group"] = run_group
        return labels

    def _get_container_name(self) -> str:
        tag = os.getenv("OSWORLD_DOCKER_TAG", "").strip()
        safe_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "-", tag).strip("-").lower()
        safe_tag = safe_tag[:40] if safe_tag else ""
        parts = ["osworld"]
        if safe_tag:
            parts.append(safe_tag)
        parts.append(uuid.uuid4().hex[:8])
        return "-".join(parts)

    def _get_used_ports(self):
        """Get all currently used ports (both system and Docker)."""
        system_ports = set(conn.laddr.port for conn in psutil.net_connections())

        docker_ports = set()
        try:
            raw_containers = self.client.api.containers(all=True)
        except Exception:
            logger.warning("Failed to list Docker containers for port scan; using system ports only")
            return system_ports

        for c in raw_containers:
            try:
                for port_entry in (c.get("Ports") or []):
                    public_port = port_entry.get("PublicPort")
                    if public_port:
                        docker_ports.add(int(public_port))
            except Exception:
                continue

        return system_ports | docker_ports

    def _allocate_ports(self) -> tuple:
        """Allocate VNC, server, chromium, VLC ports in a single scan."""
        used = self._get_used_ports()

        def _find(start: int) -> int:
            port = start
            while port < 65354:
                if port not in used:
                    used.add(port)
                    return port
                port += 1
            raise PortAllocationError(f"No available ports found starting from {start}")

        return _find(8006), _find(5000), _find(9222), _find(8080)

    def _wait_for_vm_ready(self, timeout: int = 300):
        """Wait for VM to be ready by checking screenshot endpoint."""
        start_time = time.time()
        
        def check_screenshot():
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10)
                )
                return response.status_code == 200
            except Exception:
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            time.sleep(RETRY_INTERVAL)
        
        raise TimeoutError("VM failed to become ready within timeout period")
    def _copy_to_container(self, local_path: str, container_parent_dir: str):
        """
        Copy a local file or directory into container_parent_dir.
        Example:
            local_path=/host/workspace
            container_parent_dir=/
        Result:
            /host/workspace -> /workspace
        """
        local_path = os.path.abspath(local_path)
        src = Path(local_path)

        if not src.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")

        self.container.exec_run(f"mkdir -p {container_parent_dir}")

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(local_path, arcname=src.name)

        tar_stream.seek(0)
        ok = self.container.put_archive(container_parent_dir, tar_stream.read())
        if not ok:
            raise RuntimeError(f"Failed to copy {local_path} to container:{container_parent_dir}")
        
    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        # Fast path: container already exists (restarted by revert_to_snapshot)
        if self.container is not None:
            try:
                self.container.reload()
                if self.container.status == "running":
                    logger.info("Reusing restarted container (Server: %s, VNC: %s, Chrome: %s, VLC: %s)",
                                self.server_port, self.vnc_port, self.chromium_port, self.vlc_port)
                    self._wait_for_vm_ready()
                    return
            except Exception as e:
                logger.warning("Restarted container unavailable (%s), creating new one", e)
                try:
                    self.container.remove(force=True)
                except Exception:
                    pass
                self.container = None
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None

        lock = FileLock(str(self.lock_file), timeout=LOCK_TIMEOUT)
        
        try:
            with lock:
                self.vnc_port, self.server_port, self.chromium_port, self.vlc_port = self._allocate_ports()

                devices = []
                if os.path.exists("/dev/kvm"):
                    devices.append("/dev/kvm")
                    logger.info("KVM device found, using hardware acceleration")
                else:
                    self.environment["KVM"] = "N"
                    logger.warning("KVM device not found, running without hardware acceleration (will be slower)")

                self.container = self.client.containers.run(
                    "happysixd/osworld-docker",
                    name=self._get_container_name(),
                    environment=self.environment,
                    cap_add=["NET_ADMIN"],
                    devices=devices,
                    labels=self._get_container_labels(),
                    volumes={
                        os.path.abspath(path_to_vm): {
                            "bind": "/System.qcow2",
                            "mode": "ro"
                        }
                    },
                    ports={
                        8006: self.vnc_port,
                        5000: self.server_port,
                        9222: self.chromium_port,
                        8080: self.vlc_port
                    },
                    detach=True
                )
            
            logger.info(f"Started container with ports - VNC: {self.vnc_port}, "
                       f"Server: {self.server_port}, Chrome: {self.chromium_port}, VLC: {self.vlc_port}")

            self._wait_for_vm_ready()

        except Exception as e:
            if self.container:
                try:
                    self.container.remove(force=True)
                except Exception:
                    logger.exception("Failed to remove partially created Docker container")
                finally:
                    self.container = None
            raise e

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for Docker provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        if self.container is not None:
            logger.info("Restarting Docker container for VM reset (ports preserved)...")
            try:
                self.container.restart(timeout=30)
                logger.info("Container restarted successfully, QEMU will boot fresh from base image")
                return
            except Exception as e:
                logger.warning("Container restart failed: %s; falling back to destroy+recreate", e)
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        if self.container:
            logger.info("Stopping VM...")
            try:
                self.container.remove(force=True)
            except docker.errors.NotFound:
                logger.info("Container already removed")
            except Exception as e:
                logger.error(f"Error stopping container: {e}")
            finally:
                self.container = None
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None
