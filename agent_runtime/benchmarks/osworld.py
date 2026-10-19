from __future__ import annotations

import base64
import json as _json
import logging
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any

from agent_runtime.backend import (
    BackendProvider,
    BackendSession,
    ProxyConfig,
    UploadIncompleteError,
)
from agent_runtime.benchmark import Benchmark
from agent_runtime.env import Env
from agent_runtime.evaluator import Evaluator
from agent_runtime.harness import Harness
from agent_runtime.registry import register_benchmark
from agent_runtime.types import (
    Action,
    ActionType,
    EnvCapabilities,
    EvaluationResult,
    HarnessContext,
    Modality,
    ModalityKind,
    Observation,
    RunTaskResult,
    Task,
    Trace,
)

logger = logging.getLogger(__name__)

_script_logger = logging.getLogger(f"{__name__}.scripts")

def _log_script_result(
    result: dict[str, Any],
    label: str,
    elapsed: float | None = None,
    debug: bool = False,
) -> None:

    rc = result.get("returncode", "?")
    status = result.get("status", "unknown")
    output = result.get("output", "") or ""
    error = result.get("error", "") or ""
    output_lines = output.rstrip().split("\n") if output.strip() else []
    n_lines = len(output_lines)

    time_str = f" ({elapsed:.1f}s)" if elapsed is not None else ""
    summary = f"[{label}] rc={rc} status={status} output={n_lines} lines{time_str}"

    if (rc == 0 and not debug) or rc == 0:
        _script_logger.info(summary)
    else:
        _script_logger.warning(summary)

    should_show_output = debug or rc != 0 or _script_logger.isEnabledFor(logging.DEBUG)
    if should_show_output and output_lines:
        _script_logger.info("[%s] ─── output ───", label)
        for line in output_lines:
            _script_logger.info("[%s]   %s", label, line)
        _script_logger.info("[%s] ─── end ───", label)
    if error and (debug or rc != 0):
        _script_logger.warning("[%s] stderr: %s", label, error[:500])

def _fixup_bash_result(result: dict[str, Any] | None) -> dict[str, Any]:

    if result is None:
        return {}
    if (result.get("status") == "error"
            and "_append_event" in result.get("output", "")):
        logger.debug("run_bash_script hit _append_event bug (script may have run)")
    return result

_osworld_imported = False

def _ensure_osworld_on_path(osworld_path: str) -> None:

    global _osworld_imported
    if _osworld_imported:
        return
    osworld_root = os.path.abspath(osworld_path)
    if osworld_root not in sys.path:
        sys.path.insert(0, osworld_root)
    _osworld_imported = True

def _import_desktop_env(osworld_path: str) -> Any:
    _ensure_osworld_on_path(osworld_path)
    from desktop_env.desktop_env import DesktopEnv as _DesktopEnv
    return _DesktopEnv

_DOMAINS = [
    "chrome", "gimp", "libreoffice_calc", "libreoffice_impress",
    "libreoffice_writer", "multi_apps", "os", "thunderbird", "vlc", "vs_code",
]

def _load_task_configs(
    osworld_path: str,
    split: str = "osworld_all",
    domains: list[str] | None = None,
) -> list[dict[str, Any]]:

    examples_dir = Path(osworld_path) / "evaluation_examples"
    split_file = examples_dir / f"{split}.json"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. "
            f"Available: {[f.stem for f in examples_dir.glob('*.json')]}"
        )

    with open(split_file) as f:
        split_meta: dict[str, list[str]] = _json.load(f)

    target_domains = domains or list(split_meta.keys())
    configs: list[dict[str, Any]] = []

    for domain in target_domains:
        if domain not in split_meta:
            logger.warning("Domain '%s' not found in split '%s', skipping.", domain, split)
            continue
        task_ids = split_meta[domain]
        for task_id in task_ids:
            task_file = examples_dir / "examples" / domain / f"{task_id}.json"
            if not task_file.exists():
                logger.warning("Task file not found: %s, skipping.", task_file)
                continue
            with open(task_file) as f:
                config = _json.load(f)
            config["_domain"] = domain
            configs.append(config)

    return configs

def _task_config_to_task(config: dict[str, Any]) -> Task:

    domain = config.get("_domain", "unknown")
    related_apps = config.get("related_apps", [])
    suite = domain if domain != "unknown" else (
        related_apps[0] if len(related_apps) == 1 else ",".join(sorted(related_apps))
    )

    return Task(
        task_id=config["id"],
        benchmark="osworld",
        suite=suite,
        instruction=config.get("instruction"),
        native_ref={
            "task_config": config,
            "domain": domain,
            "snapshot": config.get("snapshot", "init_state"),
        },
        resource_requirements={
            "qemu_vm": 1,
            "kvm": True,
            "ram_gb": 4,
            "cpu_cores": 2,
        },
        metadata={
            "domain": domain,
            "related_apps": related_apps,
            "proxy": config.get("proxy", False),
            "snapshot": config.get("snapshot", "init_state"),
        },
        tags=[domain] + related_apps,
    )

_PLATFORM_TO_OSWORLD: dict[ActionType, str] = {
    ActionType.CLICK: "CLICK",
    ActionType.TAP: "CLICK",
    ActionType.DOUBLE_CLICK: "DOUBLE_CLICK",
    ActionType.RIGHT_CLICK: "RIGHT_CLICK",
    ActionType.SCROLL: "SCROLL",
    ActionType.TYPE: "TYPING",
    ActionType.KEY_PRESS: "PRESS",
    ActionType.HOTKEY: "HOTKEY",
    ActionType.DRAG: "DRAG_TO",
    ActionType.HOVER: "MOVE_TO",
    ActionType.NOOP: "WAIT",
}

def action_to_pyautogui(action: Action) -> str:

    args = action.arguments

    if action.type == ActionType.PYTHON:
        return args.get("code", args.get("command", ""))

    if action.type in (ActionType.CLICK, ActionType.TAP):
        x, y = args.get("x"), args.get("y")
        button = args.get("button", "left")
        if x is not None and y is not None:
            if button != "left":
                return f"pyautogui.click(x={x}, y={y}, button='{button}')"
            return f"pyautogui.click(x={x}, y={y})"
        return "pyautogui.click()"

    if action.type == ActionType.DOUBLE_CLICK:
        x, y = args.get("x"), args.get("y")
        if x is not None and y is not None:
            return f"pyautogui.doubleClick(x={x}, y={y})"
        return "pyautogui.doubleClick()"

    if action.type == ActionType.RIGHT_CLICK:
        x, y = args.get("x"), args.get("y")
        if x is not None and y is not None:
            return f"pyautogui.rightClick(x={x}, y={y})"
        return "pyautogui.rightClick()"

    if action.type == ActionType.SCROLL:
        dx = args.get("dx", 0)
        dy = args.get("dy", args.get("amount", 0))
        direction = args.get("direction")
        if direction:
            scroll_map = {"up": 3, "down": -3, "left": -3, "right": 3}
            amount = scroll_map.get(direction, 0)
            if direction in ("left", "right"):
                return f"pyautogui.hscroll({amount})"
            return f"pyautogui.vscroll({amount})"
        parts = []
        if dx:
            parts.append(f"pyautogui.hscroll({dx})")
        if dy:
            parts.append(f"pyautogui.vscroll({dy})")
        return "; ".join(parts) if parts else "pyautogui.vscroll(-3)"

    if action.type == ActionType.TYPE:
        text = args.get("text", "")
        return f"pyautogui.typewrite({text!r})"

    if action.type == ActionType.KEY_PRESS:
        keys_list = args.get("keys")
        if keys_list:
            if len(keys_list) > 1:
                keys_repr = ", ".join(repr(k) for k in keys_list)
                return f"pyautogui.hotkey({keys_repr})"
            key = keys_list[0]
        else:
            key = args.get("key", "")
        return f"pyautogui.press({key!r})"

    if action.type == ActionType.HOTKEY:
        keys = args.get("keys", [])
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split("+")]
        keys_repr = ", ".join(repr(k) for k in keys)
        return f"pyautogui.hotkey({keys_repr})"

    if action.type == ActionType.DRAG:
        x, y = args.get("x", args.get("end_x")), args.get("y", args.get("end_y"))
        if x is not None and y is not None:
            return f"pyautogui.dragTo({x}, {y}, duration=1.0, button='left', mouseDownUp=True)"
        return "pass"

    if action.type == ActionType.HOVER:
        x, y = args.get("x"), args.get("y")
        if x is not None and y is not None:
            return f"pyautogui.moveTo({x}, {y})"
        return "pass"

    if action.type == ActionType.BASH:
        command = args.get("command", "")
        return f"import subprocess; subprocess.run({command!r}, shell=True)"

    if action.type == ActionType.SCREENSHOT:
        return "pass"

    if action.type in (ActionType.TERMINATE, ActionType.SUBMIT):
        return ""

    if action.type == ActionType.NOOP:
        return ""

    if action.type == ActionType.CUSTOM:
        return args.get("command", args.get("code", "pass"))

    return "pass"

def action_to_computer13(action: Action) -> dict[str, Any] | str:

    args = action.arguments

    if action.type in (ActionType.CLICK, ActionType.TAP):
        result: dict[str, Any] = {"action_type": "CLICK"}
        if "x" in args:
            result["x"] = args["x"]
        if "y" in args:
            result["y"] = args["y"]
        if "button" in args:
            result["button"] = args["button"]
        return result

    if action.type == ActionType.DOUBLE_CLICK:
        result = {"action_type": "DOUBLE_CLICK"}
        if "x" in args:
            result["x"] = args["x"]
        if "y" in args:
            result["y"] = args["y"]
        return result

    if action.type == ActionType.RIGHT_CLICK:
        result = {"action_type": "RIGHT_CLICK"}
        if "x" in args:
            result["x"] = args["x"]
        if "y" in args:
            result["y"] = args["y"]
        return result

    if action.type == ActionType.SCROLL:
        dx = args.get("dx", 0)
        dy = args.get("dy", 0)
        direction = args.get("direction")
        if direction:
            scroll_map = {"up": (0, 3), "down": (0, -3), "left": (-3, 0), "right": (3, 0)}
            dx, dy = scroll_map.get(direction, (0, 0))
        return {"action_type": "SCROLL", "dx": dx, "dy": dy}

    if action.type == ActionType.TYPE:
        return {"action_type": "TYPING", "text": args.get("text", "")}

    if action.type == ActionType.KEY_PRESS:
        keys_list = args.get("keys")
        key = keys_list[0] if keys_list else args.get("key", "")
        return {"action_type": "PRESS", "key": key}

    if action.type == ActionType.HOTKEY:
        keys = args.get("keys", [])
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split("+")]
        return {"action_type": "HOTKEY", "keys": keys}

    if action.type == ActionType.DRAG:
        return {
            "action_type": "DRAG_TO",
            "x": args.get("x", args.get("end_x", 0)),
            "y": args.get("y", args.get("end_y", 0)),
        }

    if action.type == ActionType.HOVER:
        return {
            "action_type": "MOVE_TO",
            "x": args.get("x", 0),
            "y": args.get("y", 0),
        }

    if action.type == ActionType.TERMINATE:
        reason = args.get("reason", "")
        if reason.upper() == "FAIL":
            return "FAIL"
        return "DONE"

    if action.type == ActionType.SUBMIT:
        return "DONE"

    if action.type == ActionType.NOOP:
        return "WAIT"

    if action.type == ActionType.PYTHON:
        return args.get("code", args.get("command", "pass"))

    return "WAIT"

def _is_terminal_action(action: Action) -> bool:

    if action.type in (ActionType.TERMINATE, ActionType.SUBMIT):
        return True
    raw = (action.raw or "").strip().upper()
    if raw in ("DONE", "FAIL"):
        return True
    return False

def _get_terminal_marker(action: Action) -> str:

    if action.type == ActionType.TERMINATE:
        reason = action.arguments.get("reason", "DONE").upper()
        return reason if reason in ("DONE", "FAIL") else "DONE"
    if action.type == ActionType.SUBMIT:
        return "DONE"
    raw = (action.raw or "").strip().upper()
    return raw if raw in ("DONE", "FAIL") else "DONE"

def make_a11y_desktop_verifier(
    *,
    timeout: float = 8.0,
    poll: float = 0.7,
):

    def _verifier(env: OSWorldEnv) -> None:
        expected = env._desktop_filenames_to_check()
        if not expected:
            return
        t0 = time.time()
        missing = env._wait_for_desktop_icons(expected, timeout=timeout, poll=poll)
        elapsed = time.time() - t0
        if missing:
            logger.warning(
                "Desktop a11y did not show %s after %.1fs — files are on "
                "disk but icons may not be rendered yet",
                sorted(missing), elapsed,
            )
        else:
            logger.info(
                "Desktop a11y confirmed %d uploaded file(s) visible after %.1fs",
                len(expected), elapsed,
            )

    return _verifier

def make_desktop_icon_count_verifier(
    *,
    min_delta: int = 1,
    timeout: float = 8.0,
    poll: float = 0.7,
):

    def _verifier(env: OSWorldEnv) -> None:
        baseline = (env._pre_upload_state or {}).get("gjs_icon_count")
        if baseline is None:
            logger.debug(
                "No gjs icon baseline captured; skipping count verifier",
            )
            return
        target = baseline + min_delta
        t0 = time.time()
        deadline = t0 + timeout
        current: int | None = None
        while time.time() < deadline:
            current = env._count_gjs_desktop_icons()
            if current is not None and current >= target:
                logger.info(
                    "Desktop icons settled: %d (baseline=%d, target>=%d, "
                    "elapsed=%.1fs)",
                    current, baseline, target, time.time() - t0,
                )
                return
            time.sleep(poll)
        logger.warning(
            "Desktop icon count did not reach %d within %.1fs "
            "(baseline=%d, current=%s) — icons may not be rendered",
            target, timeout, baseline, current,
        )

    return _verifier

class OSWorldEnv(Env):

    def __init__(self, task: Task, session: Any = None, **env_options: Any) -> None:
        super().__init__(task, session, **env_options)
        self._capabilities = EnvCapabilities(
            step=True,
            screenshot=True,
            a11y_tree=True,
        )
        self._task_config: dict[str, Any] | None = (task.native_ref or {}).get("task_config") or None
        self._action_space: str = env_options.get("action_space", "pyautogui")
        self._step_count = 0
        self._terminated = False
        self._last_obs: Observation | None = None
        self._screenshot_dir: str | None = env_options.get("screenshot_dir")
        self._sleep_after_execution: float = env_options.get("sleep_after_execution", 2.0)
        self._initial_observation_delay: float = env_options.get("initial_observation_delay", 3.0)
        self._settle_time: float = env_options.get("settle_time", 20.0)

        self._upload_refresh_settle_s: float = float(
            env_options.get("upload_refresh_settle_s", 1.5)
        )
        self._upload_refresh_hotkey: str | None = env_options.get(
            "upload_refresh_hotkey", None
        )
        self._upload_refresh_verify = env_options.get(
            "upload_refresh_verify", None
        )

        self._pre_upload_state: dict[str, Any] = {}

        if session is not None and hasattr(session, "desktop_env"):
            self.desktop_env = session.desktop_env
        else:
            raise ValueError(
                "OSWorldEnv requires an OSWorldSession with a .desktop_env attribute. "
                "Use OSWorldBackendProvider.create_session() to obtain one."
            )

    def upload_file(self, local_path: str, vm_path: str) -> None:

        OSWorldBackendProvider._upload_file_to_vm(
            self.desktop_env, local_path, vm_path,
        )

    def _upload_task_files(self) -> None:

        if not self.task.env_setup:
            return
        files = self.task.env_setup.get("files")
        if not files:
            return
        for i, entry in enumerate(files):
            local_path = entry.get("local_path", "")
            vm_path = entry.get("vm_path", "")
            if not local_path or not vm_path:
                logger.warning("Skipping files[%d]: missing local_path or vm_path", i)
                continue
            t0 = time.time()
            try:
                self.upload_file(local_path, vm_path)
                elapsed = time.time() - t0
                logger.info("File upload [%d] done in %.1fs: %s → %s",
                            i, elapsed, local_path, vm_path)
            except Exception as exc:
                logger.error("File upload [%d] failed: %s → %s: %s",
                             i, local_path, vm_path, exc)
                raise

    def run_setup_script(self, script_content: str, timeout: int = 300) -> dict[str, Any]:

        wrapped = OSWorldBackendProvider._wrap_with_proxy_source(script_content)
        result = self.desktop_env.controller.run_bash_script(wrapped, timeout=timeout)
        return _fixup_bash_result(result)

    def _run_task_hook(self, hook_key: str) -> None:

        if not self.task.env_setup:
            return
        script_path = self.task.env_setup.get(hook_key)
        if not script_path:
            return
        p = Path(script_path)
        if not p.exists():
            logger.warning("Task hook %s not found: %s", hook_key, script_path)
            return
        label = f"task_hook:{hook_key}:{p.name}"
        logger.info("Running task hook %s: %s", hook_key, script_path)
        t0 = time.time()
        result = self.run_setup_script(p.read_text())
        elapsed = time.time() - t0
        _log_script_result(result, label=label, elapsed=elapsed, debug=True)
        rc = result.get("returncode", -1)
        if rc != 0:
            logger.warning("Task hook %s exited with code %d", hook_key, rc)

    _DESKTOP_PREFIXES: tuple[str, ...] = (
        "~/Desktop/",
        "/home/user/Desktop/",
        "$HOME/Desktop/",
    )

    def _desktop_filenames_to_check(self) -> list[str]:

        if not self.task.env_setup:
            return []
        files = self.task.env_setup.get("files") or []
        names: list[str] = []
        for entry in files:
            vm_path = (entry.get("vm_path") or "").strip()
            if not vm_path:
                continue
            if any(vm_path.startswith(pfx) for pfx in self._DESKTOP_PREFIXES):
                names.append(Path(vm_path).name)
        return names

    def _send_refresh_hotkey(self, key: str) -> None:

        cmd = f"import pyautogui; pyautogui.press({key!r})"
        try:
            self.desktop_env.controller.execute_python_command(cmd)
        except Exception as exc:
            logger.debug("Refresh hotkey %r failed (non-fatal): %s", key, exc)

    def _a11y_tree_enabled(self) -> bool:
        return bool(getattr(self.desktop_env, "require_a11y_tree", True))

    def _wait_for_desktop_icons(
        self,
        expected_names: list[str],
        timeout: float = 8.0,
        poll: float = 0.7,
    ) -> set[str]:

        if not self._a11y_tree_enabled():
            return set(expected_names)
        deadline = time.time() + timeout
        missing: set[str] = set(expected_names)
        while time.time() < deadline and missing:
            tree = None
            try:
                tree = self.desktop_env.controller.get_accessibility_tree()
            except Exception as exc:
                logger.debug("a11y tree fetch failed during refresh poll: %s", exc)
            if tree:
                tree_text = tree if isinstance(tree, str) else str(tree)
                missing = {n for n in missing if n not in tree_text}
                if not missing:
                    return missing
            time.sleep(poll)
        return missing

    def _count_gjs_desktop_icons(self) -> int | None:

        if not self._a11y_tree_enabled():
            return None
        try:
            tree = self.desktop_env.controller.get_accessibility_tree()
        except Exception as exc:
            logger.debug("a11y tree fetch failed during icon count: %s", exc)
            return None
        if not tree:
            return None
        text = tree if isinstance(tree, str) else str(tree)
        start = text.find('application name="gjs"')
        if start < 0:
            return 0

        end = text.find('<application name=', start + len('application name="gjs"'))
        if end < 0:
            end = len(text)
        subtree = text[start:end]
        return subtree.count('<icon ')

    def _capture_pre_upload_state(self) -> None:

        try:
            count = self._count_gjs_desktop_icons()
        except Exception as exc:
            logger.debug("Pre-upload state capture failed: %s", exc)
            count = None
        self._pre_upload_state = {"gjs_icon_count": count}

    def _refresh_desktop_after_upload(self) -> None:

        if self._upload_refresh_settle_s > 0:
            time.sleep(self._upload_refresh_settle_s)

        if self._upload_refresh_hotkey:
            self._send_refresh_hotkey(self._upload_refresh_hotkey)

        if self._upload_refresh_verify is not None:
            try:
                self._upload_refresh_verify(self)
            except Exception:
                logger.exception(
                    "upload_refresh_verify raised (non-fatal, step 0 may "
                    "not reflect the upload)",
                )

    def _save_screenshot(self, png_bytes: bytes, step_index: int) -> str | None:
        if not self._screenshot_dir:
            return None
        os.makedirs(self._screenshot_dir, exist_ok=True)
        filename = f"step_{step_index:03d}.png"
        filepath = os.path.join(self._screenshot_dir, filename)
        with open(filepath, "wb") as f:
            f.write(png_bytes)
        return filepath

    def _native_obs_to_observation(
        self,
        native_obs: dict[str, Any],
        step_index: int,
        terminal: bool = False,
    ) -> Observation:

        parts: list[Modality] = []

        screenshot = native_obs.get("screenshot")
        if screenshot and isinstance(screenshot, bytes):
            screenshot_b64 = base64.b64encode(screenshot).decode("ascii")
            self._save_screenshot(screenshot, step_index)
            parts.append(Modality(
                kind=ModalityKind.IMAGE,
                content=screenshot_b64,
                mime_type="image/png",
            ))

        a11y_tree = native_obs.get("accessibility_tree")
        if a11y_tree is not None:
            parts.append(Modality(
                kind=ModalityKind.A11Y_TREE,
                content=a11y_tree,
            ))

        terminal_output = native_obs.get("terminal")
        if terminal_output is not None:
            parts.append(Modality(
                kind=ModalityKind.TEXT,
                content=terminal_output,
                name="terminal",
            ))

        return Observation(
            parts=parts,
            instruction=native_obs.get("instruction") or self.task.instruction,
            step_index=step_index,
            terminal=terminal,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        self._step_count = 0
        self._terminated = False

        native_obs = self.desktop_env.reset(task_config=self._task_config)

        time.sleep(self._sleep_after_execution)
        if native_obs is None or not native_obs.get("screenshot"):
            native_obs = self.desktop_env._get_obs()

        if self.session and hasattr(self.session, "_provider") and self.session._provider:
            self.session._provider.provision_vm(self.desktop_env)
        else:
            OSWorldBackendProvider._patch_vm_server(self.desktop_env)
        if self.task.env_setup and self.task.env_setup.get("files"):
            self._capture_pre_upload_state()
        self._upload_task_files()
        self._run_task_hook("pre_task")
        if self.task.env_setup and (
            self.task.env_setup.get("pre_task") or self.task.env_setup.get("files")
        ):
            if self._initial_observation_delay > 0:
                time.sleep(self._initial_observation_delay)
            self._refresh_desktop_after_upload()
            native_obs = self.desktop_env._get_obs()

        obs = self._native_obs_to_observation(native_obs, step_index=0)
        self._last_obs = obs
        self.trace.record_step(observation=obs)

        info: dict[str, Any] = {
            "task_id": (self._task_config or {}).get("id", self.task.task_id),
            "domain": (self._task_config or {}).get("_domain", ""),
            "instruction": self.task.instruction,
        }
        return obs, info

    def step(
        self, action: Action
    ) -> tuple[Observation, float | None, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("Episode already terminated. Call reset() first.")

        self._step_count += 1
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        native_obs: dict[str, Any] | None = None

        if _is_terminal_action(action):
            terminal_marker = _get_terminal_marker(action)
            native_obs, _, _, _ = self.desktop_env.step(terminal_marker, pause=0)
            terminated = True
            info["terminal_marker"] = terminal_marker
        else:
            native_action: Any
            if self._action_space == "pyautogui":
                if action.type == ActionType.PYTHON:
                    native_action = action.arguments.get("code", action.arguments.get("command", "pass"))
                else:
                    native_action = action_to_pyautogui(action)
                if native_action:
                    info["native_action"] = native_action
            else:
                native_action = action_to_computer13(action)
                info["native_action"] = native_action

            if native_action:
                try:
                    native_obs, reward, done, step_info = self.desktop_env.step(
                        native_action, pause=self._sleep_after_execution
                    )
                    info["step_result"] = step_info
                    if done:
                        terminated = True
                except Exception as exc:
                    logger.warning("DesktopEnv.step() failed: %s", exc)
                    info["step_error"] = str(exc)

        if native_obs is None:
            try:
                native_obs = self.desktop_env._get_obs()
            except Exception:
                native_obs = {"screenshot": None, "accessibility_tree": None, "terminal": None, "instruction": self.task.instruction}

        obs = self._native_obs_to_observation(
            native_obs,
            step_index=self._step_count,
            terminal=terminated,
        )
        self._last_obs = obs

        self.trace.record_step(
            observation=obs,
            action=action,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        if terminated:
            self._terminated = True

        return obs, None, terminated, truncated, info

    def run(
        self,
        harness: Harness,
        evaluator: Evaluator | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> RunTaskResult:
        if self._last_obs is None:
            self.reset()

        effective_max_steps = max_steps or self.task.max_steps or 15
        start_time = time.time()

        obs = self._last_obs
        assert obs is not None

        try:
            self.desktop_env.controller.start_recording()
        except Exception:
            logger.debug("Video recording not available, skipping.")

        eval_events: list = []

        for step_idx in range(effective_max_steps):
            if timeout and (time.time() - start_time) > timeout:
                logger.warning("Timeout reached after %d steps", step_idx)
                break

            context = HarnessContext(task=self.task, step_index=step_idx)
            try:
                action = harness.act(obs, context)
            except Exception:
                logger.exception("Harness.act() failed at step %d", step_idx)
                break

            prev_obs = obs
            obs, _, terminated, truncated, info = self.step(action)

            if evaluator is not None:
                try:
                    step_events = evaluator.on_step(
                        env=self,
                        observation=prev_obs,
                        action=action,
                        next_observation=obs,
                        info=info,
                    )
                    if step_events:
                        eval_events.extend(step_events)
                except Exception:
                    logger.debug("evaluator.on_step() failed at step %d", step_idx, exc_info=True)

            if terminated or truncated:
                break

        time.sleep(self._settle_time)
        self._run_task_hook("post_task")

        evaluation = None
        if evaluator is not None:
            evaluation = evaluator.finalize(env=self, trace=self.trace)
            if eval_events and not evaluation.events:
                evaluation = evaluation.model_copy(update={"events": eval_events})
            self.trace.record_evaluation(evaluation)

        self.trace.finalize()

        return RunTaskResult(trace=self.trace, evaluation=evaluation)

    def save_recording(self, path: str) -> bool:

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.desktop_env.controller.end_recording(path)
            return True
        except Exception:
            logger.debug("Failed to save recording.")
            return False

    def close(self) -> None:
        super().close()

class OSWorldEvaluator(Evaluator):

    def __init__(self, task: Task | None = None) -> None:
        self.task = task

    def finalize(self, env: Env | None, trace: Trace) -> EvaluationResult:
        if env is None or not isinstance(env, OSWorldEnv):
            return EvaluationResult(
                success=False,
                score=0.0,
                reason="OSWorldEvaluator requires an OSWorldEnv",
                error_type="evaluator.no_env",
            )

        task_config = getattr(env, "_task_config", None)
        if not task_config or "evaluator" not in task_config:
            return EvaluationResult(
                success=None,
                score=None,
                reason=(
                    "custom task has no native OSWorld evaluator config — "
                    "native scoring skipped (pass a custom Evaluator to score it)"
                ),
                native_result={"evaluated": False, "task_id": env.task.task_id},
            )

        try:
            score = float(env.desktop_env.evaluate())
        except Exception as exc:
            logger.exception("DesktopEnv.evaluate() failed")
            return EvaluationResult(
                success=False,
                score=0.0,
                reason=f"Evaluation failed: {exc}",
                error_type="evaluator.eval_error",
            )

        return EvaluationResult(
            success=score > 0.0,
            score=score,
            reason=None if score > 0.0 else "task not successful",
            native_result={
                "score": score,
                "task_id": env.task.task_id,
                "domain": (env._task_config or {}).get("_domain", ""),
            },
        )

class OSWorldSession(BackendSession):

    def __init__(
        self,
        task: Task,
        desktop_env: Any,
        *,
        provider: OSWorldBackendProvider | None = None,
        owns_env: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(task, **kwargs)
        self.desktop_env = desktop_env
        self._provider = provider
        self._owns_env = owns_env

    def close(self) -> None:
        if self._owns_env and self.desktop_env is not None:
            try:
                self.desktop_env.close()
            except Exception as exc:
                logger.warning("DesktopEnv.close() failed: %s", exc)
            self.desktop_env = None
        elif self._provider is not None:
            self._provider._release_env(self.desktop_env)
        super().close()

class OSWorldBackendProvider(BackendProvider):

    def __init__(
        self,
        osworld_path: str,
        provider_name: str = "docker",
        path_to_vm: str | None = None,
        region: str | None = None,
        snapshot_name: str = "init_state",
        action_space: str = "pyautogui",
        screen_size: tuple[int, int] = (1920, 1080),
        headless: bool = False,
        require_a11y_tree: bool = True,
        require_terminal: bool = False,
        os_type: str = "Ubuntu",
        reuse: bool = True,
        num_instances: int = 1,
        cache_dir: str = "cache",
        enable_proxy: bool = False,
        proxy_config: ProxyConfig | None = None,
        env_hooks: dict[str, Any] | None = None,
    ) -> None:
        self.osworld_path = os.path.abspath(osworld_path)
        self.provider_name = provider_name
        self.path_to_vm = path_to_vm
        self.region = region
        self.snapshot_name = snapshot_name
        self.action_space = action_space
        self.screen_size = screen_size
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal
        self.os_type = os_type
        self.reuse = reuse
        self.num_instances = num_instances
        self.cache_dir = cache_dir
        self.proxy_config = proxy_config
        self.env_hooks = env_hooks

        if proxy_config and proxy_config.vm_proxy:
            self.enable_proxy = True
        else:
            self.enable_proxy = enable_proxy

        self._env_queue: queue.Queue[Any] | None = None
        self._all_envs: list[Any] = []
        self._setup_done = False
        self._saved_env_vars: dict[str, str | None] = {}

    def _create_desktop_env(self, idx: int = 0) -> Any:

        DesktopEnv = _import_desktop_env(self.osworld_path)

        kwargs: dict[str, Any] = {
            "provider_name": self.provider_name,
            "snapshot_name": self.snapshot_name,
            "action_space": self.action_space,
            "screen_size": self.screen_size,
            "headless": self.headless,
            "require_a11y_tree": self.require_a11y_tree,
            "require_terminal": self.require_terminal,
            "os_type": self.os_type,
            "cache_dir": self.cache_dir,
            "enable_proxy": self.enable_proxy,
        }
        if self.path_to_vm:
            kwargs["path_to_vm"] = self.path_to_vm
        if self.region:
            kwargs["region"] = self.region

        return DesktopEnv(**kwargs)

    @staticmethod
    def _patch_vm_server(env: Any, idx: int = 0) -> None:

        patch_script = (
            "import subprocess, os\n"
            "target = '/home/user/server/main.py'\n"
            "if not os.path.isfile(target):\n"
            "    print('PATCH_SKIP: not found')\n"
            "else:\n"
            "    with open(target) as f:\n"
            "        content = f.read()\n"
            "    if 'lambda *a, **kw: None' in content:\n"
            "        print('PATCH_SKIP: already patched')\n"
            "    elif '_append_event' not in content:\n"
            "        print('PATCH_SKIP: not needed')\n"
            "    else:\n"
            "        stub = '_append_event = lambda *a, **kw: None\\n'\n"
            "        patched = stub + content\n"
            "        with open(target, 'w') as f:\n"
            "            f.write(patched)\n"
            "        # Restart in background so this request can return cleanly\n"
            "        subprocess.Popen(\n"
            "            'sleep 1 && sudo systemctl restart osworld_server',\n"
            "            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL\n"
            "        )\n"
            "        print('PATCH_DONE')\n"
        )
        if os.environ.get("AGENT_RUNTIME_SKIP_VM_SERVER_PATCH", "").strip() in {"1", "true", "yes"}:

            logger.info("Skipping VM server patch on env %d (AGENT_RUNTIME_SKIP_VM_SERVER_PATCH set)", idx)
            return
        logger.info("Patching VM server on env %d to fix _append_event...", idx)
        result = env.controller.run_python_script(patch_script)
        output = ""
        if result:
            output = result.get("output", "") or ""
            logger.info("VM server patch on env %d: %s", idx, output.strip())
        else:
            logger.warning("VM server patch returned None on env %d", idx)
        if "PATCH_DONE" in output:
            time.sleep(5)
            logger.info("VM server restarted on env %d", idx)

    def _setup_vm_proxy_env(self, env: Any, idx: int = 0) -> None:

        if not self.proxy_config or not self.proxy_config.vm_proxy:
            return
        proxy_url = self.proxy_config.vm_proxy
        no_proxy = self.proxy_config.no_proxy

        env_content = (
            f"export http_proxy={proxy_url}\n"
            f"export https_proxy={proxy_url}\n"
            f"export HTTP_PROXY={proxy_url}\n"
            f"export HTTPS_PROXY={proxy_url}\n"
            f"export no_proxy={no_proxy}\n"
            f"export NO_PROXY={no_proxy}\n"
        )
        write_script = (
            "import os, subprocess\n"
            f"content = {env_content!r}\n"
            "with open('/tmp/agent_runtime_proxy.env', 'w') as f:\n"
            "    f.write(content)\n"
            "os.chmod('/tmp/agent_runtime_proxy.env', 0o644)\n"
            "print('PROXY_ENV_WRITTEN')\n"
            "\n"
            "# Fix DNS: reduce resolve timeout from 15s to 2s.\n"
            "# The VM's link DNS is unreachable; this makes DNS fail fast\n"
            "# instead of blocking for 15 seconds per query.\n"
            "# HTTP operations through proxy are unaffected (proxy resolves DNS).\n"
            "dns_conf = (\n"
            "    '[Resolve]\\n'\n"
            "    'DNSStubListener=yes\\n'\n"
            "    'ResolveUnicastSingleLabel=yes\\n'\n"
            ")\n"
            "resolv_options = 'options timeout:2 attempts:1\\nnameserver 127.0.0.53\\n'\n"
            "try:\n"
            "    # Reduce DNS timeout by overwriting /etc/resolv.conf options\n"
            "    subprocess.run(\n"
            "        'echo password | sudo -S bash -c \"echo \\'options timeout:2 attempts:1\\' >> /etc/resolv.conf\"',\n"
            "        shell=True, capture_output=True, timeout=5\n"
            "    )\n"
            "    # Flush stale cache entries\n"
            "    subprocess.run(\n"
            "        'echo password | sudo -S resolvectl flush-caches',\n"
            "        shell=True, capture_output=True, timeout=5\n"
            "    )\n"
            "    print('DNS_TIMEOUT_REDUCED')\n"
            "except Exception as e:\n"
            "    print(f'DNS_FIX_SKIPPED: {e}')\n"
        )
        result = env.controller.run_python_script(write_script)
        output = (result.get("output", "") or "") if result else ""
        if "PROXY_ENV_WRITTEN" in output:
            logger.info("Proxy env file written on env %d: %s", idx, proxy_url)
        else:
            logger.warning("Failed to write proxy env file on env %d: %s", idx, result)
        if "DNS_TIMEOUT_REDUCED" in output:
            logger.info("DNS timeout reduced on env %d (2s instead of 15s)", idx)
        elif "DNS_FIX_SKIPPED" in output:
            logger.debug("DNS fix skipped on env %d: %s", idx, output)

    def _setup_vm_apt_proxy(self, env: Any, idx: int = 0) -> None:

        if not self.proxy_config or not self.proxy_config.vm_proxy:
            return
        proxy_url = self.proxy_config.vm_proxy
        script = (
            "import subprocess, os\n"
            f"proxy = {proxy_url!r}\n"
            "lines = [\n"
            "    f'Acquire::http::Proxy \"{proxy}\";',\n"
            "    f'Acquire::https::Proxy \"{proxy}\";',\n"
            "]\n"
            "content = '\\n'.join(lines) + '\\n'\n"
            "tmp = '/tmp/99proxy'\n"
            "with open(tmp, 'w') as f:\n"
            "    f.write(content)\n"
            "subprocess.run(\n"
            "    f'echo password | sudo -S cp {tmp} /etc/apt/apt.conf.d/99proxy',\n"
            "    shell=True, capture_output=True, timeout=10,\n"
            ")\n"
            "os.remove(tmp)\n"
            "print('APT_PROXY_SET')\n"
        )
        result = env.controller.run_python_script(script)
        output = (result.get("output", "") or "") if result else ""
        if "APT_PROXY_SET" in output:
            logger.info("APT proxy configured on env %d: %s", idx, proxy_url)
        else:
            logger.warning("Failed to configure APT proxy on env %d: %s", idx, result)

    def provision_vm(self, env: Any, idx: int = 0) -> None:

        self._patch_vm_server(env, idx)
        self._setup_vm_proxy_env(env, idx)
        self._setup_vm_apt_proxy(env, idx)
        self._upload_env_files(env, idx)
        self._run_env_hooks(env, idx)

    def _upload_env_files(self, env: Any, idx: int = 0) -> None:

        if not self.env_hooks:
            return
        files = self.env_hooks.get("files")
        if not files:
            return
        for i, entry in enumerate(files):
            local_path = entry.get("local_path", "")
            vm_path = entry.get("vm_path", "")
            if not local_path or not vm_path:
                logger.warning("env_hooks.files[%d]: missing local_path or vm_path", i)
                continue
            t0 = time.time()
            try:
                self._upload_file_to_vm(env, local_path, vm_path)
                elapsed = time.time() - t0
                logger.info("Env file upload [%d] on env %d done in %.1fs: %s → %s",
                            i, idx, elapsed, local_path, vm_path)
            except Exception as exc:
                logger.error("Env file upload [%d] on env %d failed: %s → %s: %s",
                             i, idx, local_path, vm_path, exc)
                raise

    def _run_env_hooks(self, env: Any, idx: int = 0) -> None:

        if not self.env_hooks:
            return
        post_start = self.env_hooks.get("post_start")
        if not post_start:
            return
        script_path = Path(post_start)
        if not script_path.exists():
            logger.warning("env_hooks post_start script not found: %s", post_start)
            return
        script_content = script_path.read_text()
        label = f"env_hook:post_start:{script_path.name}[{idx}]"

        logger.info("Running post_start hook on env %d: %s", idx, post_start)
        try:
            t0 = time.time()
            result = self._run_bash_in_vm(env, script_content, timeout=600)
            elapsed = time.time() - t0
            _log_script_result(result, label=label, elapsed=elapsed, debug=True)
        except Exception as exc:
            logger.error("post_start hook failed on env %d: %s", idx, exc)

    _PROXY_ENV_PATH = "/tmp/agent_runtime_proxy.env"

    _MIN_UPLOAD_SPEED_BPS = 1 * 1024 * 1024

    _VERIFY_MIN_TIMEOUT_S = 10.0
    _VERIFY_POLL_INTERVAL_S = 0.5

    @staticmethod
    def _stat_size_in_vm(env: Any, vm_path: str) -> int | None:

        script = (
            "import os, json\n"
            f"p = {vm_path!r}\n"
            "try:\n"
            "    size = os.path.getsize(p)\n"
            "    print('AP_STAT_OK:' + str(size))\n"
            "except FileNotFoundError:\n"
            "    print('AP_STAT_MISSING')\n"
            "except Exception as e:\n"
            "    print('AP_STAT_ERROR:' + repr(e))\n"
        )
        result = env.controller.run_python_script(script)
        if not isinstance(result, dict):
            return None
        output = (result.get("output") or "").strip()
        for line in output.splitlines():
            if line.startswith("AP_STAT_OK:"):
                try:
                    return int(line[len("AP_STAT_OK:"):])
                except ValueError:
                    return None
            if line == "AP_STAT_MISSING":
                return None
        return None

    @staticmethod
    def _wait_for_upload_visible(
        env: Any,
        vm_path: str,
        expected_size: int,
        *,
        timeout: float | None = None,
        poll: float = _VERIFY_POLL_INTERVAL_S,
    ) -> None:

        if timeout is None:
            timeout = max(
                OSWorldBackendProvider._VERIFY_MIN_TIMEOUT_S,
                expected_size / OSWorldBackendProvider._MIN_UPLOAD_SPEED_BPS,
            )
        start = time.time()
        deadline = start + timeout
        last_seen: int | None = None
        while True:
            size = OSWorldBackendProvider._stat_size_in_vm(env, vm_path)
            if size == expected_size:
                if last_seen is not None and last_seen != expected_size:
                    logger.info(
                        "Upload became visible on VM: %s (%d bytes, settled after %.1fs)",
                        vm_path, expected_size, time.time() - start,
                    )
                return
            last_seen = size
            if time.time() >= deadline:
                raise UploadIncompleteError(
                    vm_path,
                    expected_size,
                    last_seen,
                    elapsed=time.time() - start,
                )
            time.sleep(poll)

    @staticmethod
    def _upload_file_to_vm(
        env: Any,
        local_path: str,
        vm_path: str,
        *,
        max_retries: int = 3,
        connect_timeout: int = 30,
        verify_timeout: float | None = None,
    ) -> None:

        import requests
        from requests_toolbelt import MultipartEncoder

        local_path = os.path.abspath(local_path)
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        http_server = env.controller.http_server
        file_size = os.path.getsize(local_path)
        read_timeout = max(60, int(file_size / OSWorldBackendProvider._MIN_UPLOAD_SPEED_BPS))
        timeout = (connect_timeout, read_timeout)
        url = f"{http_server}/setup/upload"
        last_error: Exception | None = None

        logger.info(
            "Uploading %s (%d bytes) → VM:%s  timeout=%s retries=%d",
            local_path, file_size, vm_path, timeout, max_retries,
        )

        for attempt in range(max_retries):
            try:
                with open(local_path, "rb") as fp:
                    form = MultipartEncoder({
                        "file_path": vm_path,
                        "file_data": (os.path.basename(vm_path), fp),
                    })
                    resp = requests.post(
                        url,
                        headers={"Content-Type": form.content_type},
                        data=form,
                        timeout=timeout,
                    )

                if resp.status_code == 200:
                    try:
                        OSWorldBackendProvider._wait_for_upload_visible(
                            env, vm_path, file_size, timeout=verify_timeout,
                        )
                    except UploadIncompleteError as verify_exc:

                        logger.warning(
                            "Attempt %d/%d uploaded but size mismatch: %s",
                            attempt + 1, max_retries, verify_exc,
                        )
                        last_error = verify_exc
                    else:
                        logger.info(
                            "Uploaded %s (%d bytes) → VM:%s",
                            local_path, file_size, vm_path,
                        )
                        return
                else:
                    msg = (
                        f"Upload returned {resp.status_code}: "
                        f"{local_path} → {vm_path}: {resp.text[:200]}"
                    )
                    logger.warning("Attempt %d/%d failed — %s", attempt + 1, max_retries, msg)
                    last_error = RuntimeError(msg)

            except requests.RequestException as exc:
                logger.warning(
                    "Attempt %d/%d failed — %s: %s",
                    attempt + 1, max_retries, type(exc).__name__, exc,
                )
                last_error = exc

            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.info("Retrying in %ds...", backoff)
                time.sleep(backoff)

        if isinstance(last_error, UploadIncompleteError):
            raise last_error
        raise RuntimeError(
            f"File upload failed after {max_retries} attempts: "
            f"{local_path} → {vm_path}"
        ) from last_error

    @staticmethod
    def _wrap_with_proxy_source(script_content: str) -> str:

        proxy_source = (
            f"[ -f {OSWorldBackendProvider._PROXY_ENV_PATH} ] "
            f"&& source {OSWorldBackendProvider._PROXY_ENV_PATH}\n"
        )
        if script_content.startswith("#!"):
            shebang_end = script_content.index("\n") + 1
            return script_content[:shebang_end] + proxy_source + script_content[shebang_end:]
        return proxy_source + script_content

    @staticmethod
    def _run_bash_in_vm(env: Any, script_content: str, timeout: int = 300) -> dict[str, Any]:

        wrapped = OSWorldBackendProvider._wrap_with_proxy_source(script_content)

        result = env.controller.run_bash_script(wrapped, timeout=timeout)
        if result and result.get("status") != "error":
            return result

        error_info = ""
        if result:
            error_info = result.get("output", "") or result.get("error", "")
        logger.info("run_bash_script failed (%s), falling back to run_python_script",
                    error_info[:120])

        b64_script = base64.b64encode(wrapped.encode()).decode()
        py_wrapper = (
            "import os, tempfile, base64\n"
            f"script = base64.b64decode('{b64_script}').decode()\n"
            "fp = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False)\n"
            "fp.write(script); fp.close()\n"
            "os.chmod(fp.name, 0o755)\n"
            "rc = os.system('/bin/bash ' + fp.name)\n"
            "os.unlink(fp.name)\n"
            "print('RETURNCODE:', rc >> 8)\n"
        )
        fb_result = env.controller.run_python_script(py_wrapper)
        if fb_result and fb_result.get("status") != "error":
            output = fb_result.get("output", "")
            rc = 0 if "RETURNCODE: 0" in output else 1
            return {"status": "success" if rc == 0 else "error", "output": output, "returncode": rc}
        return fb_result or {"status": "error", "output": "both methods failed", "returncode": -1}

    def _release_env(self, env: Any) -> None:

        if self._env_queue is not None:
            self._env_queue.put(env)

    def _apply_proxy_env(self) -> None:

        proxy = (self.proxy_config.host_proxy if self.proxy_config else None) or os.environ.get("AGENT_RUNTIME_PROXY")
        if not proxy:
            return
        no_proxy = self.proxy_config.no_proxy if self.proxy_config else "127.0.0.1,localhost,::1"
        env_keys = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
        no_proxy_keys = ["no_proxy", "NO_PROXY"]
        for key in env_keys:
            self._saved_env_vars[key] = os.environ.get(key)
            os.environ[key] = proxy
        for key in no_proxy_keys:
            self._saved_env_vars[key] = os.environ.get(key)
            os.environ[key] = no_proxy

    def _restore_proxy_env(self) -> None:

        for key, original in self._saved_env_vars.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
        self._saved_env_vars.clear()

    def _write_vm_proxy_config(self) -> None:

        if not self.proxy_config or not self.proxy_config.vm_proxy:
            return
        from urllib.parse import urlparse
        parsed = urlparse(self.proxy_config.vm_proxy)
        host = parsed.hostname or "172.30.0.1"
        port = parsed.port or 19890
        config = [{
            "host": host,
            "port": port,
            "username": parsed.username or "",
            "password": parsed.password or "",
            "protocol": parsed.scheme or "http",
            "provider": "local",
            "type": "http",
        }]
        import tempfile
        self._proxy_config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="osworld_proxy_", delete=False,
        )
        _json.dump(config, self._proxy_config_file)
        self._proxy_config_file.close()
        os.environ["PROXY_CONFIG_FILE"] = self._proxy_config_file.name
        logger.info("VM proxy config written to %s → %s:%d",
                     self._proxy_config_file.name, host, port)

    def setup(self) -> None:
        if self._setup_done:
            return

        self._apply_proxy_env()
        self._write_vm_proxy_config()

        n = self.num_instances if self.reuse else 0
        if n > 0:
            self._env_queue = queue.Queue()
            for i in range(n):
                logger.info(
                    "Creating DesktopEnv %d/%d (provider=%s)...",
                    i + 1, n, self.provider_name,
                )
                env = self._create_desktop_env(idx=i)
                self._all_envs.append(env)
                logger.info(
                    "DesktopEnv %d/%d ready, VM IP: %s",
                    i + 1, n, getattr(env, "vm_ip", "unknown"),
                )
            for env in self._all_envs:
                self._env_queue.put(env)

        self._setup_done = True

    def teardown(self) -> None:
        for i, env in enumerate(self._all_envs):
            logger.info("Closing DesktopEnv %d/%d...", i + 1, len(self._all_envs))
            try:
                env.close()
            except Exception as exc:
                logger.warning("DesktopEnv %d close failed: %s", i + 1, exc)
        self._all_envs.clear()
        if self._env_queue is not None:
            while not self._env_queue.empty():
                try:
                    self._env_queue.get_nowait()
                except queue.Empty:
                    break
            self._env_queue = None
        self._setup_done = False

        self._restore_proxy_env()
        if hasattr(self, "_proxy_config_file") and self._proxy_config_file:
            try:
                os.unlink(self._proxy_config_file.name)
            except OSError:
                pass

        self._cleanup_orphan_containers()

    def _cleanup_orphan_containers(self) -> None:

        try:
            import docker
            client = docker.from_env()
            orphans = client.containers.list(
                all=True,
                filters={"label": "agent_runtime.osworld=true"},
            )
            if orphans:
                logger.info("Cleaning up %d orphan container(s)...", len(orphans))
            for c in orphans:
                try:
                    c.remove(force=True)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("Orphan container cleanup skipped: %s", exc)

    def create_session(self, task: Task, **kwargs: Any) -> OSWorldSession:
        if not self._setup_done:
            self.setup()

        if self.reuse:
            assert self._env_queue is not None
            desktop_env = self._env_queue.get()
            return OSWorldSession(
                task=task,
                desktop_env=desktop_env,
                provider=self,
                owns_env=False,
            )
        else:
            desktop_env = self._create_desktop_env()
            return OSWorldSession(
                task=task,
                desktop_env=desktop_env,
                provider=self,
                owns_env=True,
            )

    def capabilities(self) -> dict[str, Any]:
        return {
            "type": "osworld",
            "protocol": "desktop_env",
            "provider": self.provider_name,
            "session_type": "qemu_vm",
            "action_space": self.action_space,
            "reuse": self.reuse,
            "num_instances": self.num_instances,
            "a11y_tree": self.require_a11y_tree,
            "proxy": self.enable_proxy,
        }

    def task_resources(self) -> dict[str, float]:
        if self.provider_name == "docker":
            return {"osworld_vm": 1}
        return {}

@register_benchmark(
    "osworld",
    description="OSWorld desktop GUI benchmark (QEMU/VMware/Docker VM adapter).",
    overwrite=True,
)
class OSWorldBenchmark(Benchmark):

    name = "osworld"

    def __init__(
        self,
        osworld_path: str,
        split: str = "osworld_all",
        action_space: str = "pyautogui",
        **provider_kwargs: Any,
    ) -> None:
        self.osworld_path = os.path.abspath(osworld_path)
        self.default_split = split
        self.action_space = action_space
        self._provider_kwargs = provider_kwargs
        self._task_cache: dict[str, Task] | None = None

    def _load_all_tasks(self, split: str | None = None) -> dict[str, Task]:
        effective_split = split or self.default_split
        configs = _load_task_configs(self.osworld_path, split=effective_split)
        return {cfg["id"]: _task_config_to_task(cfg) for cfg in configs}

    def list_tasks(
        self,
        split: str | None = None,
        limit: int | None = None,
        **filters: Any,
    ) -> list[Task]:
        effective_split = split or self.default_split
        domains: list[str] | None = filters.pop("domains", None)
        if isinstance(domains, str):
            domains = [domains]

        configs = _load_task_configs(
            self.osworld_path,
            split=effective_split,
            domains=domains,
        )
        tasks = [_task_config_to_task(cfg) for cfg in configs]

        domain_filter = filters.pop("domain", None)
        if domain_filter:
            tasks = [t for t in tasks if t.metadata.get("domain") == domain_filter]

        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def get_task(self, task_id: str) -> Task:
        if self._task_cache is None:
            self._task_cache = self._load_all_tasks()
        if task_id in self._task_cache:
            return self._task_cache[task_id]

        for domain in _DOMAINS:
            task_file = (
                Path(self.osworld_path) / "evaluation_examples" / "examples"
                / domain / f"{task_id}.json"
            )
            if task_file.exists():
                with open(task_file) as f:
                    config = _json.load(f)
                config["_domain"] = domain
                task = _task_config_to_task(config)
                self._task_cache[task_id] = task
                return task

        raise KeyError(f"Task '{task_id}' not found in OSWorld benchmark")

    def make_env(self, task: Task, session: Any = None, **env_options: Any) -> OSWorldEnv:
        env_options.setdefault("action_space", self.action_space)
        return OSWorldEnv(task=task, session=session, **env_options)

    def make_evaluator(
        self, task: Task | None = None, mode: str = "native", **kwargs: Any
    ) -> OSWorldEvaluator:
        return OSWorldEvaluator(task=task)

    def requirements(self) -> dict[str, Any]:
        return {
            "qemu_vm": 1,
            "kvm": True,
            "ram_gb": 4,
            "cpu_cores": 2,
        }

__all__ = [
    "OSWorldBackendProvider",
    "OSWorldBenchmark",
    "OSWorldEnv",
    "OSWorldEvaluator",
    "OSWorldSession",
    "action_to_computer13",
    "action_to_pyautogui",
]
