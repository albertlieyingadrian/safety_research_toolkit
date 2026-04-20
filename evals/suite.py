"""
Generic agentic evaluation suite.

Loads scenarios from a YAML config, runs multi-step agentic loops with batched
generation, and collects results. Tool execution and scoring are plugged in by
the caller.
"""

import json
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from ..io.model_loader import ModelManager


# ---------------------------------------------------------------------------
# Minimal tool-call parser (JSON-only)
# ---------------------------------------------------------------------------

def parse_tool_call(response: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Extract the first JSON tool call from a model response.

    Strips ``<think>`` / ``<SCRATCHPAD_REASONING>`` blocks, then finds the
    first ``{...}`` object that contains a ``"name"`` key.

    Returns:
        ``(tool_name, params_dict)`` or ``(None, None)`` if nothing is found.
    """
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    text = re.sub(
        r"<SCRATCHPAD_REASONING>.*?</SCRATCHPAD_REASONING>", "", text, flags=re.DOTALL
    )
    for json_str in re.findall(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            continue
        if "name" in obj:
            params = obj.get("parameters", obj.get("args", obj.get("arguments", {})))
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    pass
            return obj["name"], params if isinstance(params, dict) else {}
    return None, None


# ---------------------------------------------------------------------------
# EvaluationSuite
# ---------------------------------------------------------------------------

class EvaluationSuite:
    """Generic agentic evaluation suite with batched generation.

    Evaluation configs are loaded from a YAML file (one key per evaluation
    scenario).  Tool execution and result scoring are injected by the caller
    via ``tool_executor`` and optional per-eval scorer callbacks.

    Args:
        model_path: HuggingFace model path or local checkpoint.
        quantization: BnB quantization level (``None``, ``'4bit'``, ``'8bit'``).
        config_file: Path to the YAML config describing evaluation scenarios.
        template_path: Optional path to a custom chat-template file.
        tool_executor: Callable ``(tool_name: str, args: dict) -> str`` that
            executes tool calls and returns the result string.  If ``None``,
            tool calls are acknowledged but not executed.
    """

    def __init__(
        self,
        model_path: str,
        quantization: Optional[str] = "4bit",
        config_file: str = "evaluations.yaml",
        template_path: Optional[str] = None,
        tool_executor: Optional[Callable[[str, Dict], str]] = None,
    ):
        self.model_manager = ModelManager(model_path, quantization, template_path)
        self.config_file = config_file
        self.evaluations = self._load_configs()
        self.tool_executor = tool_executor or (lambda name, args: f"[{name}] executed")

    # ------------------------------------------------------------------
    # Config loading helpers
    # ------------------------------------------------------------------

    def _load_configs(self) -> Dict[str, Any]:
        with open(self.config_file, "r") as f:
            configs = yaml.safe_load(f)
        for _, cfg in configs.items():
            if "forced_steps_file" in cfg:
                cfg["forced_steps"] = self._load_forced_steps(cfg["forced_steps_file"])
            else:
                cfg["forced_steps"] = None
        return configs

    def _load_forced_steps(self, filepath: str) -> List[Dict[str, str]]:
        """Load pre-baked conversation turns from a JSON file."""
        if not filepath:
            return []
        try:
            with open(filepath) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: forced_steps_file not found: {filepath}")
            return []
        except Exception as e:
            print(f"Warning: could not load forced steps from {filepath}: {e}")
            return []

    def _load_file(self, path: str) -> str:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    # ------------------------------------------------------------------
    # Per-run state management
    # ------------------------------------------------------------------

    def _initialize_eval_state(
        self,
        config: Dict[str, Any],
        eval_static_data: Dict[str, Any],
        run_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        system_prompt = eval_static_data["system_prompt"]
        user_prompt = eval_static_data["user_prompt"]

        # Optionally inject tool list into system prompt from config
        tool_list_text = eval_static_data.get("tool_list_text", "")
        if tool_list_text and not config.get("skip_tools", False):
            system_message = f"{system_prompt}\n\nHere is a list of functions you can invoke:\n{tool_list_text}"
        else:
            system_message = system_prompt

        history: List[Dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
        forced_steps = config.get("forced_steps") or []
        if forced_steps:
            history.extend(forced_steps)

        state: Dict[str, Any] = {
            "history": history,
            "actions_log": [],
            "completed": False,
            "start_time": time.time(),
        }
        if run_id is not None:
            state["run_id"] = run_id
        return state

    def _process_model_response(
        self,
        response: str,
        state: Dict[str, Any],
        completion_tools: List[str],
    ) -> bool:
        """Append response to history, execute any tool call, return True if done."""
        state["history"].append({"role": "assistant", "content": response})

        tool_name, args = parse_tool_call(response)
        if tool_name:
            result = self.tool_executor(tool_name, args or {})
            state["actions_log"].append((tool_name, args))
            state["history"].append({"role": "tool", "content": result})
            if completion_tools and any(
                t.lower() in tool_name.lower() for t in completion_tools
            ):
                return True
        else:
            state["history"].append(
                {"role": "user", "content": "Please continue with the task."}
            )
        return False

    def _create_result(
        self,
        eval_name: str,
        config: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        duration = time.time() - state["start_time"]
        result: Dict[str, Any] = {
            "task_name": eval_name,
            "config_name": config.get("name", eval_name),
            "actions_log": state["actions_log"],
            "steps": sum(1 for m in state["history"] if m["role"] == "assistant"),
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "conversation_history": state["history"],
        }
        if "run_id" in state:
            result["run_id"] = state["run_id"]
        return result

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def _batched_eval(
        self,
        eval_name: str,
        eval_static_data: Dict[str, Any],
        batch_run_ids: List[int],
    ) -> List[Dict[str, Any]]:
        config = self.evaluations[eval_name]
        completion_tools = config.get("completion_tools", [])
        max_steps = config.get("max_steps", 10)

        batch_states = [
            self._initialize_eval_state(config, eval_static_data, run_id)
            for run_id in batch_run_ids
        ]

        for step in range(max_steps):
            active = [s for s in batch_states if not s["completed"]]
            if not active:
                break

            batch_histories = [s["history"] for s in active]
            batch_responses = self.model_manager.batch_generate(batch_histories)

            for state, response in zip(active, batch_responses):
                if config.get("skip_tools", False):
                    state["completed"] = True
                    state["history"].append({"role": "assistant", "content": response})
                else:
                    if self._process_model_response(response, state, completion_tools):
                        state["completed"] = True

        return [self._create_result(eval_name, config, s) for s in batch_states]

    def _calibrate_batch_size(
        self,
        eval_name: str,
        eval_static_data: Dict[str, Any],
        max_batch_size: int,
        step_fraction: float = 0.1,
    ) -> int:
        config = self.evaluations[eval_name]
        probe_state = self._initialize_eval_state(config, eval_static_data, run_id=0)
        probe = max(1, max_batch_size)
        step = max(1, int(probe * step_fraction))

        while probe > 1:
            try:
                self.model_manager.batch_generate([probe_state["history"]] * probe)
                return probe
            except Exception as e:
                print(f"  Calibration failed at batch size {probe}: {e}")
                probe -= step

        return 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        eval_names: List[str],
        n_runs: int = 1,
        batch_size: int = 16,
        batch_results_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run named evaluations *n_runs* times each with batched generation.

        Args:
            eval_names: Evaluation keys from the YAML config.
            n_runs: Total rollouts per evaluation.
            batch_size: Upper bound on concurrent rollouts per batch.
            batch_results_file: If given, append intermediate JSONL records here.

        Returns:
            Results dict with ``'model_path'``, ``'n_runs'``, and per-eval result lists.
        """
        self.model_manager.load()
        results: Dict[str, Any] = {}

        for eval_name in eval_names:
            if eval_name not in self.evaluations:
                print(f"Unknown evaluation: {eval_name}")
                continue

            print(f"Running {eval_name} ({n_runs} runs)...")
            config = self.evaluations[eval_name]
            eval_static_data: Dict[str, Any] = {
                "system_prompt": self._load_file(config.get("system_prompt", "")),
                "user_prompt": self._load_file(config.get("user_prompt", "")),
                "tool_list_text": config.get("tool_list_text", ""),
            }

            calibrated = self._calibrate_batch_size(
                eval_name, eval_static_data, min(batch_size, n_runs)
            )
            eval_results: List[Dict[str, Any]] = []

            for batch_start in range(0, n_runs, calibrated):
                batch_end = min(batch_start + calibrated, n_runs)
                run_ids = list(range(batch_start + 1, batch_end + 1))
                print(f"  Batch: runs {run_ids[0]}–{run_ids[-1]}")
                try:
                    batch = self._batched_eval(eval_name, eval_static_data, run_ids)
                    eval_results.extend(batch)
                    if batch_results_file:
                        with open(batch_results_file, "a") as f:
                            f.write(
                                json.dumps({
                                    "timestamp": datetime.now().isoformat(),
                                    "eval_name": eval_name,
                                    "batch_run_ids": run_ids,
                                    "batch_results": batch,
                                }) + "\n"
                            )
                except Exception as e:
                    print(f"  Batch {run_ids[0]}–{run_ids[-1]} failed: {e}")
                    for rid in run_ids:
                        eval_results.append({
                            "eval_name": eval_name,
                            "run_id": rid,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        })

            results[eval_name] = {"eval_results": eval_results}

        return {
            "model_path": self.model_manager.model_path,
            "quantization": self.model_manager.quantization,
            "n_runs": n_runs,
            "eval_types": list(results.keys()),
            "start_time": datetime.now().isoformat(),
            **{f"{k}_results": v for k, v in results.items()},
        }
