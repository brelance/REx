"""Core agent execution loop: Think -> Act -> Observe -> Repeat."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

if TYPE_CHECKING:
    from .user_agent import UserAgent

from ..config import MediaConfig, ModelConfig, PromptConfig
from ..models.content import ContentBlock, TextBlock, ToolResultBlock
from ..models.message import Message
from ..models.task import TaskDefinition
from ..models.trace import (
    AuditSnapshot,
    CompactEvent,
    DimensionScores,
    MediaLoad,
    RecapEvent,
    TokenUsage,
    TraceEnd,
    TraceMessage,
    TraceStart,
)
from ..trace.writer import TraceWriter
from .agent_tools import build_agent_tools
from .compact import (
    _estimate_tokens,
    do_auto_compact,
    micro_compact,
    should_auto_compact,
)
from .dispatcher import ToolDispatcher
from .plan_execute import PlanExecuteRunner
from .media_loader import collect_media_references, load_media_from_ref, model_supports_modality, to_content_block
from .providers.openai_compat import OpenAICompatProvider
from .protocol import is_reasoning_only_response, protocol_response_text
from .recap import RecapController, RecapState, parse_recap_json
from .rex_runner import RExRunner
from .system_prompt import build_system_prompt
from .todo import TodoManager
from .rex_core.runtime import is_context_overflow_error


def _log(msg: str) -> None:
    """Print a log line and flush immediately (important for container logs)."""
    print(msg, flush=True)


PROVIDER_CHAT_TIMEOUT_SECONDS = 300
RUNNER_TIMEOUT_MULTIPLIER = 2
REX_INTERNAL_MESSAGE_EXTRA = {
    "internal": True,
    "source": "rex",
}


def _reflection_prompt() -> str:
    return """Reflect briefly on the latest tool result, then continue the task.
Call tools if more work is needed. If the task is complete, answer the user."""


def _brief(d: dict, max_len: int = 80) -> str:
    """Compact one-line summary of a dict for logging."""
    s = json.dumps(d, ensure_ascii=False)
    return s if len(s) <= max_len else s[:max_len] + "..."


def _make_local_tool_result(tool_use, text: str, is_error: bool = False) -> ToolResultBlock:
    """Create a ToolResultBlock for a locally dispatched agent tool."""
    return ToolResultBlock(
        tool_use_id=tool_use.id,
        content=[TextBlock(text=text)],
        is_error=is_error,
    )


def _write_recap_event(trace_id: str, writer: TraceWriter, recap: RecapController, *, done_task: str | None = None, remaining_subtasks: list[str] | None = None, note: str = "") -> None:
    """Write current ReCAP state to the trace."""
    writer.write_event(RecapEvent(
        trace_id=trace_id,
        state=recap.state.value,
        depth=recap.depth,
        current_task=recap.current_task,
        done_task=done_task,
        remaining_subtasks=remaining_subtasks or [],
        tree_snapshot=recap.tree_snapshot(),
        note=note,
    ))


def _provider_chat_with_timeout(
    provider: OpenAICompatProvider,
    messages: list[Message],
    tools: Any,
    timeout_s: int = PROVIDER_CHAT_TIMEOUT_SECONDS,
) -> tuple[Message, TokenUsage] | None:
    """Call provider.chat, returning None if one call exceeds timeout_s."""
    result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result_queue.put(("ok", provider.chat(messages, tools=tools)), block=False)
        except BaseException as exc:
            result_queue.put(("err", exc), block=False)

    thread = threading.Thread(target=_run, name="provider-chat-timeout", daemon=True)
    thread.start()

    try:
        status, payload = result_queue.get(timeout=timeout_s)
    except queue.Empty:
        return None

    if status == "err":
        raise cast(BaseException, payload)
    return cast(tuple[Message, TokenUsage], payload)


def _cap_conversation_images(messages: list[Message], max_images: int) -> int:
    """Drop earliest image blocks in-place when total exceeds *max_images*.

    Protects messages[0] (system) and messages[1] (initial user prompt).
    Keeps the *last* max_images images; replaces earlier ones with text
    placeholders.  Returns the number of images dropped.
    """
    if max_images <= 0:
        return 0

    # Count ALL images (including protected prompt) to respect global budget
    total_images = sum(
        1 for msg in messages for b in msg.content if b.type == "image"
    )
    if total_images <= max_images:
        return 0

    # Protected images in messages[0:2] (system + initial prompt)
    protected = sum(
        1 for msg in messages[:2] for b in msg.content if b.type == "image"
    )
    # Non-protected budget: total budget minus the protected images
    allowed = max(0, max_images - protected)

    # Collect positions of droppable images (messages[2:])
    positions: list[tuple[int, int]] = []
    for mi in range(2, len(messages)):
        for bi, block in enumerate(messages[mi].content):
            if block.type == "image":
                positions.append((mi, bi))

    if len(positions) <= allowed:
        return 0

    # Drop earliest, keep the last `allowed`
    n_drop = len(positions) - allowed
    for mi, bi in positions[:n_drop]:
        messages[mi].content[bi] = TextBlock(
            text="[Image dropped: conversation image limit reached]"
        )
    return n_drop


def _strip_old_turn_images(messages: list[Message], keep_recent_turns: int = 3) -> int:
    """Strip ImageBlocks from messages older than *keep_recent_turns* turns.

    A turn is defined by an assistant message.  All images in messages before
    the *keep_recent_turns*-th most recent assistant message are removed
    in-place.  Text content (including media captions) is preserved.
    """
    if keep_recent_turns <= 0:
        return 0

    assistant_indices = [
        i for i, msg in enumerate(messages)
        if msg.role == "assistant"
    ]

    if len(assistant_indices) <= keep_recent_turns:
        return 0

    cutoff_idx = assistant_indices[-keep_recent_turns]

    n_stripped = 0
    for i in range(cutoff_idx):
        msg = messages[i]
        new_content = [b for b in msg.content if b.type != "image"]
        removed = len(msg.content) - len(new_content)
        if removed:
            msg.content = new_content
            n_stripped += removed

    return n_stripped


def _build_initial_user_content(
    task: TaskDefinition,
    *,
    trace_id: str,
    writer: TraceWriter,
    model_cfg: ModelConfig | None,
    media_cfg: MediaConfig | None,
) -> list[ContentBlock]:
    content: list[ContentBlock] = [TextBlock(text=task.prompt.text)]
    if media_cfg is not None and not media_cfg.enabled:
        return content

    cfg = media_cfg or MediaConfig()
    model = model_cfg or ModelConfig()
    refs = collect_media_references(task.prompt.text, task.prompt.attachments)
    if not refs:
        return content

    workspace_root = Path.cwd()
    task_dir = Path(task.task_file).parent if task.task_file else None
    for idx, ref in enumerate(refs):
        ref_modality = "image"
        if ref.mime_type:
            if ref.mime_type.startswith("audio/"):
                ref_modality = "audio"
            elif ref.mime_type.startswith("video/"):
                ref_modality = "video"
            elif ref.mime_type.startswith("text/") or ref.mime_type in {"application/json", "application/xml"}:
                ref_modality = "document"
        if idx >= cfg.max_files:
            writer.write_event(MediaLoad(
                trace_id=trace_id,
                modality=ref_modality,  # type: ignore[arg-type]
                source_path=ref.raw_path,
                mime_type=ref.mime_type or "",
                size_bytes=0,
                sha256="",
                status="skipped",
                note=f"exceeds max_files={cfg.max_files}",
            ))
            continue
        try:
            loaded = load_media_from_ref(
                ref,
                workspace_root=workspace_root,
                task_dir=task_dir,
                max_bytes=cfg.max_bytes_per_file,
                image_max_dimension=cfg.image_max_dimension,
            )
            if not model_supports_modality(model.input_modalities, loaded.modality):
                writer.write_event(MediaLoad(
                    trace_id=trace_id,
                    modality=loaded.modality,  # type: ignore[arg-type]
                    source_path=loaded.source_path,
                    mime_type=loaded.mime_type,
                    size_bytes=loaded.size_bytes,
                    sha256=loaded.sha256,
                    status="skipped",
                    note=f"model does not support modality: {loaded.modality}",
                ))
                if cfg.strict_mode:
                    raise ValueError(f"Model {model.model_id} does not support {loaded.modality} input")
                continue
            content.append(to_content_block(loaded))
            writer.write_event(MediaLoad(
                trace_id=trace_id,
                modality=loaded.modality,  # type: ignore[arg-type]
                source_path=loaded.source_path,
                mime_type=loaded.mime_type,
                size_bytes=loaded.size_bytes,
                sha256=loaded.sha256,
                status="loaded",
                note=ref.source,
            ))
        except Exception as exc:
            writer.write_event(MediaLoad(
                trace_id=trace_id,
                modality=ref_modality,  # type: ignore[arg-type]
                source_path=ref.raw_path,
                mime_type=ref.mime_type or "",
                size_bytes=0,
                sha256="",
                status="error",
                note=str(exc),
            ))
            if cfg.strict_mode:
                raise
    return content


def run_task(
    task: TaskDefinition,
    provider: OpenAICompatProvider,
    trace_dir: str | Path = "traces",
    *,
    sandbox_tools: bool = False,
    sandbox_url: str | None = None,
    prompt_cfg: PromptConfig | None = None,
    model_cfg: ModelConfig | None = None,
    media_cfg: MediaConfig | None = None,
    user_agent: "UserAgent | None" = None,
) -> Path:
    """Execute one trial of a task and write JSONL trace.

    Args:
        sandbox_tools: When True, sandbox tools (shell/file/browser) are
            appended to task tools and dispatched via
            :class:`SandboxToolDispatcher`.
        sandbox_url: When provided, sandbox tool calls are routed over
            HTTP to a container sandbox server at this URL (e.g.
            ``http://localhost:18080``).  When *None*, sandbox tools
            execute locally via subprocess (backward compatibility).

    Returns the path to the trace file.
    """
    trace_id = str(uuid4())
    trace_path = Path(trace_dir) / f"{task.task_id}_{trace_id[:8]}.jsonl"

    endpoint_map = task.get_endpoint_map()
    http_dispatcher = ToolDispatcher(endpoint_map)
    _mcfg = media_cfg or MediaConfig()

    sandbox_tool_list = None
    if sandbox_tools:
        from .sandbox_dispatcher import SandboxToolDispatcher
        from .sandbox_tools import SANDBOX_TOOLS

        # Deduplicate: skip sandbox tools already defined in task.yaml
        existing_names = {t.name for t in task.tools}
        sandbox_tool_list = [t for t in SANDBOX_TOOLS if t.name not in existing_names]
        task_tools = list(task.tools) + sandbox_tool_list
        dispatcher = SandboxToolDispatcher(
            http_dispatcher,
            sandbox_url=sandbox_url,
            max_images_per_turn=_mcfg.max_images_per_turn,
            tool_image_max_dimension=_mcfg.tool_image_max_dimension,
            tool_image_quality=_mcfg.tool_image_quality,
        )
    else:
        task_tools = task.tools
        dispatcher = http_dispatcher

    # Build agent-level tools (todo, compact)
    agent_tool_list = build_agent_tools(
        enable_todo=task.environment.enable_todo,
        enable_compact=task.environment.enable_compact,
    )
    task_tools = task_tools + agent_tool_list

    # Initialise TodoManager and compact state
    todo_mgr = TodoManager() if task.environment.enable_todo else None
    auto_compact_count = 0
    context_window = model_cfg.context_window if model_cfg else 200_000

    total_usage = TokenUsage()
    turn_count = 0
    wall_start = time.monotonic()
    model_time_s = 0.0
    tool_time_s = 0.0
    timed_out = False
    context_overflow = False
    termination_reason = "completed"
    timeout_type: str | None = None
    timeout_seconds: int | None = None

    # User agent state
    user_agent_rounds = 0
    ua_done = False
    ua_cfg = task.user_agent
    ua_enabled = ua_cfg.enabled and user_agent is not None
    ua_max_rounds = ua_cfg.max_rounds if ua_enabled else 0
    # Apply the same budget multiplier to every execution mode, exactly once.
    provider_timeout_s = (
        PROVIDER_CHAT_TIMEOUT_SECONDS * RUNNER_TIMEOUT_MULTIPLIER
    )
    task_timeout_s = (
        task.environment.timeout_seconds * RUNNER_TIMEOUT_MULTIPLIER
    )

    _log(f"[start] task={task.task_id} model={provider.model_id} trace={trace_path.name}")
    _log(
        f"[config] max_turns={task.environment.max_turns} "
        f"timeout_multiplier={RUNNER_TIMEOUT_MULTIPLIER} "
        f"task_timeout={task_timeout_s}s "
        f"provider_chat_timeout={provider_timeout_s}s sandbox_tools={sandbox_tools}"
    )
    if agent_tool_list:
        _log(f"[agent tools] {', '.join(t.name for t in agent_tool_list)}")

    with TraceWriter(trace_path) as writer:
        # Write trace start
        writer.write_event(TraceStart(
            trace_id=trace_id,
            task_id=task.task_id,
            model=provider.model_id,
        ))

        # Build initial messages
        system_prompt = build_system_prompt(task, prompt_cfg, extra_tools=sandbox_tool_list)
        if model_cfg and model_cfg.system_prompt_prefix:
            system_prompt = model_cfg.system_prompt_prefix + "\n\n" + system_prompt
        if ua_enabled and ua_cfg.system_prompt_suffix:
            system_prompt = system_prompt + "\n\n" + ua_cfg.system_prompt_suffix
        user_content = _build_initial_user_content(
            task,
            trace_id=trace_id,
            writer=writer,
            model_cfg=model_cfg,
            media_cfg=media_cfg,
        )
        messages: list[Message] = [
            Message(role="system", content=[TextBlock(text=system_prompt)]),
            Message(role="user", content=user_content),
        ]
        recap = None
        recap_mode = task.environment.recap_mode
        if recap_mode not in {
            "recursive",
            "rex_high_confidence_batch",
            "plan_execute",
            "reflection",
        }:
            raise ValueError(f"Unsupported recap_mode: {recap_mode!r}")
        reflection_mode = (
            task.environment.enable_recap and recap_mode == "reflection"
        )
        if task.environment.enable_recap:
            if recap_mode == "recursive":
                recap = RecapController(
                    task.prompt.text,
                    max_depth=task.environment.recap_max_depth,
                    max_subtasks=task.environment.recap_max_subtasks,
                    max_obs_chars=task.environment.recap_max_obs_chars,
                    max_tree_chars=task.environment.recap_max_tree_chars,
                    force_final_answer=task.environment.recap_force_final_answer,
                    action_taken_prompt_variant=task.environment.recap_action_taken_prompt_variant,
                )

        # Log user message
        writer.write_event(TraceMessage(
            trace_id=trace_id,
            message=messages[-1],
        ))
        if recap is not None:
            recap_msg = recap.initial_prompt()
            messages.append(recap_msg)
            writer.write_event(TraceMessage(
                trace_id=trace_id,
                message=recap_msg,
                extra=dict(REX_INTERNAL_MESSAGE_EXTRA),
            ))
            _write_recap_event(trace_id, writer, recap, note="init")

        # Agent loop — wrapped in try/finally so trace_end is always written,
        # even if the model API throws an unrecoverable error mid-run.
        loop_error: str | None = None
        loop_exc: Exception | None = None
        dag_mode_completed = False
        reflection_pending = False
        reflection_visibility_correction_pending = False
        recap_finalize_visibility_retried = False

        try:
            if reflection_mode:
                _log(
                    f"[{recap_mode}] starting without max_turns limit; "
                    f"task_timeout={task_timeout_s}s"
                )

            if task.environment.enable_recap and recap_mode in {
                "rex_high_confidence_batch",
                "plan_execute",
            }:
                if recap_mode == "plan_execute":
                    _log(
                        "[plan-execute] starting "
                        f"max_steps={task.environment.plan_execute_max_steps} "
                        f"max_step_turns="
                        f"{task.environment.plan_execute_max_step_turns} "
                        f"timeout_multiplier={RUNNER_TIMEOUT_MULTIPLIER} "
                        f"task_timeout={task_timeout_s}s "
                        f"provider_chat_timeout={provider_timeout_s}s"
                    )
                else:
                    _log(
                        "[REx] starting "
                        f"mode={recap_mode} "
                        f"max_steps={task.environment.rex_max_steps} "
                        f"max_batch_steps={task.environment.rex_max_plan_batch_steps} "
                        f"max_depth={task.environment.rex_max_depth} "
                        f"max_tool_calls={task.environment.rex_max_tool_calls} "
                        "horizontal_rolling=True "
                        f"vertical_recursion={task.environment.rex_recursive_decomposition_enabled} "
                        f"timeout_multiplier={RUNNER_TIMEOUT_MULTIPLIER} "
                        f"task_timeout={task_timeout_s}s "
                        f"provider_chat_timeout={provider_timeout_s}s"
                    )
                rex_runner_cls = {
                    "rex_high_confidence_batch": RExRunner,
                    "plan_execute": PlanExecuteRunner,
                }[recap_mode]
                plan_execute_timeout_kwargs = (
                    {"task_timeout_s": task_timeout_s}
                    if recap_mode == "plan_execute"
                    else {}
                )
                rex_runner = rex_runner_cls(
                    task=task,
                    provider=provider,
                    trace_id=trace_id,
                    writer=writer,
                    base_messages=messages,
                    tools=task_tools,
                    endpoints=endpoint_map,
                    todo_mgr=todo_mgr,
                    chat_timeout_s=provider_timeout_s,
                    context_window=model_cfg.context_window if model_cfg else 128_000,
                    allow_needs_user=ua_enabled,
                    **plan_execute_timeout_kwargs,
                )
                original_rex_goal = task.prompt.text
                rex_goal = original_rex_goal
                rex_round = 0
                while True:
                    elapsed = time.monotonic() - wall_start
                    if elapsed > task_timeout_s:
                        _log(f"[timeout] {elapsed:.1f}s exceeded limit {task_timeout_s}s")
                        timed_out = True
                        timeout_type = "task"
                        timeout_seconds = task_timeout_s
                        break

                    frame_id = "root" if rex_round == 0 else f"user_round_{rex_round}"
                    log_prefix = {
                        "plan_execute": "[plan-execute]",
                    }.get(recap_mode, "[REx]")
                    _log(f"{log_prefix} round {rex_round} frame={frame_id}")
                    rex_result = rex_runner.run(
                        root_goal=rex_goal,
                        frame_id=frame_id,
                    )
                    total_usage.input_tokens += rex_result.usage.input_tokens
                    total_usage.output_tokens += rex_result.usage.output_tokens
                    turn_count += rex_result.turns
                    model_time_s += rex_result.model_time_s
                    tool_time_s += rex_result.tool_time_s
                    if rex_result.timed_out:
                        timed_out = True
                        timeout_type = rex_result.timeout_type
                        timeout_seconds = rex_result.timeout_seconds
                        break
                    _log(
                        f"{log_prefix} round complete "
                        f"turns={rex_result.turns}"
                    )

                    if rex_result.needs_user and ua_enabled:
                        if not rex_result.user_prompt:
                            break
                        assistant_msg = Message(
                            role="assistant",
                            content=[TextBlock(text=rex_result.user_prompt)],
                        )
                        messages.append(assistant_msg)
                        writer.write_event(TraceMessage(
                            trace_id=trace_id,
                            message=assistant_msg,
                        ))
                        if user_agent_rounds >= ua_max_rounds:
                            break
                        ua_text = user_agent.generate_response(
                            persona=ua_cfg.persona,
                            conversation_messages=messages,
                        )
                        if ua_text is None:
                            ua_done = True
                            _log(
                                "[user-agent] user satisfied after REx "
                                f"clarification {recursive_round}"
                            )
                            break
                        user_agent_rounds += 1
                        ua_msg = Message(
                            role="user",
                            content=[TextBlock(text=f"[user_agent]\n{ua_text}")],
                        )
                        messages.append(ua_msg)
                        writer.write_event(TraceMessage(trace_id=trace_id, message=ua_msg))
                        _log(
                            f"[user-agent] round {user_agent_rounds}/{ua_max_rounds}: "
                            f"{ua_text[:100]}"
                        )
                        recursive_goal = original_recursive_goal
                        recursive_round += 1
                        continue

                    if not ua_enabled or user_agent_rounds >= ua_max_rounds:
                        break
                    ua_text = user_agent.generate_response(
                        persona=ua_cfg.persona,
                        conversation_messages=messages,
                    )
                    if ua_text is None:
                        ua_done = True
                        _log(
                            "[user-agent] user satisfied after REx "
                            f"round {recursive_round}"
                        )
                        break
                    user_agent_rounds += 1
                    ua_msg = Message(
                        role="user",
                        content=[TextBlock(text=f"[user_agent]\n{ua_text}")],
                    )
                    messages.append(ua_msg)
                    writer.write_event(TraceMessage(trace_id=trace_id, message=ua_msg))
                    _log(
                        f"[user-agent] round {user_agent_rounds}/{ua_max_rounds}: "
                        f"{ua_text[:100]}"
                    )
                    recursive_goal = original_recursive_goal
                    recursive_round += 1
                dag_mode_completed = True
            # Recursive ReCAP and reflection use the task timeout,
            # rather than the benchmark's ReAct turn budget, as their outer bound.
            while not dag_mode_completed and (
                recap is not None
                or reflection_mode
                or turn_count < task.environment.max_turns
            ):
                # Check timeout
                elapsed = time.monotonic() - wall_start
                if elapsed > task_timeout_s:
                    _log(f"[timeout] {elapsed:.1f}s exceeded limit {task_timeout_s}s")
                    timed_out = True
                    timeout_type = "task"
                    timeout_seconds = task_timeout_s
                    break

                # --- Layer 1: Micro-compact (truncate old tool results & strip old images) ---
                if task.environment.enable_compact:
                    micro_compact(
                        messages,
                        keep_recent=task.environment.compact_keep_recent,
                        min_chars=task.environment.compact_min_chars,
                    )

                # --- Layer 2: Auto-compact (summarise when context is large) ---
                if (
                    task.environment.enable_compact
                    and auto_compact_count < task.environment.compact_max_auto_compacts
                    and should_auto_compact(messages, context_window, task.environment.compact_threshold_pct)
                ):
                    tokens_before = _estimate_tokens(messages)
                    msgs_before = len(messages)
                    _log(f"[auto-compact] triggering (est. {tokens_before} tokens, {msgs_before} msgs)")
                    messages = do_auto_compact(
                        messages,
                        provider,
                        keep_recent_on_summary=task.environment.compact_keep_recent_on_summary,
                        protect_tokens=task.environment.compact_protect_tokens,
                        todo_mgr=todo_mgr,
                    )
                    auto_compact_count += 1
                    tokens_after = _estimate_tokens(messages)
                    writer.write_event(CompactEvent(
                        trace_id=trace_id,
                        layer="auto",
                        estimated_tokens_before=tokens_before,
                        estimated_tokens_after=tokens_after,
                        messages_before=msgs_before,
                        messages_after=len(messages),
                    ))
                    _log(f"[auto-compact] done: {tokens_before} → {tokens_after} tokens, {msgs_before} → {len(messages)} msgs")

                # Strip images from turns older than keep_recent_turns
                n_old = _strip_old_turn_images(messages, _mcfg.image_keep_recent_turns)
                if n_old > 0:
                    _log(f"  [image-strip] stripped {n_old} image(s) from old turns, keeping last {_mcfg.image_keep_recent_turns} turns")

                # Cap total images in conversation before API call
                n_dropped = _cap_conversation_images(messages, _mcfg.max_conversation_images)
                if n_dropped > 0:
                    _log(f"  [image-cap] dropped {n_dropped} oldest image(s), keeping last {_mcfg.max_conversation_images}")

                reflection_prompt_active = False
                reflection_correction_active = (
                    reflection_visibility_correction_pending
                )
                reflection_visibility_correction_pending = False
                if reflection_pending:
                    reflection_pending = False
                    reflection_prompt_active = True
                    reflection_message = Message(
                        role="user",
                        content=[TextBlock(text=_reflection_prompt())],
                    )
                    messages.append(reflection_message)
                    writer.write_event(TraceMessage(
                        trace_id=trace_id,
                        message=reflection_message,
                        extra={
                            "internal": True,
                            "source": "reflection",
                            "phase": "reflect",
                        },
                    ))

                # Call model
                turn_limit = (
                    "unlimited"
                    if reflection_mode or recap is not None
                    else str(task.environment.max_turns)
                )
                _log(f"[turn {turn_count + 1}/{turn_limit}] calling model ...")
                model_t0 = time.monotonic()
                chat_result = _provider_chat_with_timeout(
                    provider, messages, tools=task_tools, timeout_s=provider_timeout_s
                )
                model_time_s += time.monotonic() - model_t0
                if chat_result is None:
                    _log(f"[timeout] provider.chat exceeded {provider_timeout_s}s; ending agent loop")
                    timed_out = True
                    timeout_type = "provider_chat"
                    timeout_seconds = provider_timeout_s
                    break
                response, usage = chat_result
                total_usage.input_tokens += usage.input_tokens
                total_usage.output_tokens += usage.output_tokens
                turn_count += 1

                text_blocks = [b for b in response.content if b.type == "text"]
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                recap_response_internal = recap is not None and (
                    recap.state != RecapState.FINALIZE
                    or bool(tool_uses)
                    or is_reasoning_only_response(response)
                )
                reflection_response_internal = (
                    (reflection_prompt_active or reflection_correction_active)
                    and (bool(tool_uses) or is_reasoning_only_response(response))
                )

                # ReCAP planning/tool responses are runner control-plane messages.
                # Only a no-tool natural-language response in FINALIZE is public.
                writer.write_event(TraceMessage(
                    trace_id=trace_id,
                    message=response,
                    usage=usage,
                    extra=(
                        dict(REX_INTERNAL_MESSAGE_EXTRA)
                        if recap_response_internal
                        else (
                            {
                                "internal": True,
                                "source": "reflection",
                                "phase": "reflect",
                            }
                            if reflection_response_internal
                            else {}
                        )
                    ),
                ))

                messages.append(response)

                if (
                    reflection_prompt_active
                    and is_reasoning_only_response(response)
                ):
                    reflection_visibility_correction_pending = True
                    correction = Message(
                        role="user",
                        content=[
                            TextBlock(
                                text=(
                                    "Your reflection produced no visible response. "
                                    "Call the required tool, or provide the final "
                                    "user-facing answer in assistant content."
                                )
                            )
                        ],
                    )
                    messages.append(correction)
                    writer.write_event(
                        TraceMessage(
                            trace_id=trace_id,
                            message=correction,
                            extra={
                                "internal": True,
                                "source": "reflection",
                                "phase": "visible_output_correction",
                            },
                        )
                    )
                    continue

                if recap is not None and recap.state != RecapState.FINALIZE:
                    recap_finalize_visibility_retried = False
                if (
                    recap is not None
                    and recap.state == RecapState.FINALIZE
                    and is_reasoning_only_response(response)
                    and not recap_finalize_visibility_retried
                ):
                    recap_finalize_visibility_retried = True
                    correction = Message(
                        role="user",
                        content=[
                            TextBlock(
                                text=(
                                    "No visible final answer was returned. Provide only "
                                    "the final user-facing answer in assistant content, "
                                    "or call a tool if verification is still required."
                                )
                            )
                        ],
                    )
                    messages.append(correction)
                    writer.write_event(
                        TraceMessage(
                            trace_id=trace_id,
                            message=correction,
                            extra=dict(REX_INTERNAL_MESSAGE_EXTRA),
                        )
                    )
                    continue

                # Summarize what the model returned
                text_preview = text_blocks[0].text[:120].replace("\n", " ") if text_blocks else ""
                _log(f"[turn {turn_count}] assistant: {len(text_blocks)} text, {len(tool_uses)} tool_use | tokens: +{usage.input_tokens}in +{usage.output_tokens}out")
                if text_preview:
                    _log(f"  text: {text_preview}{'...' if len(text_blocks[0].text) > 120 else ''}")

                if not tool_uses:
                    if recap is not None:
                        recap_text = "\n".join(b.text for b in text_blocks)
                        if recap.state != RecapState.FINALIZE:
                            recap_text = protocol_response_text(
                                response, parse_recap_json
                            )
                        recap_step = recap.process_assistant_text(recap_text)
                        _write_recap_event(
                            trace_id,
                            writer,
                            recap,
                            done_task=recap_step.done_task_name,
                            remaining_subtasks=recap_step.remaining_subtasks,
                            note=recap_step.note,
                        )
                        if recap_step.continue_loop and recap_step.prompt:
                            recap_msg = Message(role="user", content=[TextBlock(text=recap_step.prompt)])
                            messages.append(recap_msg)
                            writer.write_event(TraceMessage(
                                trace_id=trace_id,
                                message=recap_msg,
                                extra=dict(REX_INTERNAL_MESSAGE_EXTRA),
                            ))
                            _log(f"[recap] {recap_step.state.value}: current='{recap.current_task[:80]}'")
                            continue
                        if recap_step.done:
                            _log(f"[recap] complete: done_task='{recap_step.done_task_name or ''}'")
                            break

                    if ua_enabled and user_agent_rounds < ua_max_rounds:
                        ua_text = user_agent.generate_response(
                            persona=ua_cfg.persona,
                            conversation_messages=messages,
                        )
                        if ua_text is None:
                            ua_done = True
                            _log(f"[user-agent] user satisfied — ending at turn {turn_count}")
                            break
                        user_agent_rounds += 1
                        ua_msg = Message(role="user", content=[TextBlock(text=f"[user_agent]\n{ua_text}")])
                        messages.append(ua_msg)
                        writer.write_event(TraceMessage(trace_id=trace_id, message=ua_msg))
                        _log(f"[user-agent] round {user_agent_rounds}/{ua_max_rounds}: {ua_text[:100]}")
                        continue
                    _log(f"[done] no tool calls — agent finished at turn {turn_count}")
                    break

                # Dispatch each tool call
                result_blocks = []
                media_blocks: list[ContentBlock] = []
                has_non_agent_tool = False
                for tu in tool_uses:
                    _log(f"  -> tool: {tu.name}({_brief(tu.input)})")

                    # --- Local agent tool dispatch ---
                    if tu.name == "todo" and todo_mgr:
                        result_text = todo_mgr.update(tu.input.get("items", []))
                        result = _make_local_tool_result(tu, result_text)
                        result_blocks.append(result)
                        _log(f"  <- todo: OK (local)")
                        continue

                    if tu.name == "compact" and task.environment.enable_compact:
                        tokens_before = _estimate_tokens(messages)
                        msgs_before = len(messages)
                        messages = do_auto_compact(
                            messages,
                            provider,
                            keep_recent_on_summary=task.environment.compact_keep_recent_on_summary,
                            protect_tokens=task.environment.compact_protect_tokens,
                            todo_mgr=todo_mgr,
                            focus=tu.input.get("focus"),
                        )
                        auto_compact_count += 1
                        tokens_after = _estimate_tokens(messages)
                        writer.write_event(CompactEvent(
                            trace_id=trace_id,
                            layer="manual",
                            estimated_tokens_before=tokens_before,
                            estimated_tokens_after=tokens_after,
                            messages_before=msgs_before,
                            messages_after=len(messages),
                        ))
                        result = _make_local_tool_result(
                            tu, f"Context compacted. {tokens_before} → {tokens_after} est. tokens."
                        )
                        result_blocks.append(result)
                        _log(f"  <- compact: OK (local, {tokens_before} → {tokens_after} tokens)")
                        continue

                    # --- Standard dispatcher (sandbox / HTTP) ---
                    has_non_agent_tool = True
                    dispatch_result = dispatcher.dispatch(tu, trace_id)
                    # Support both 2-tuple (legacy) and 3-tuple (media-aware) dispatch
                    if len(dispatch_result) == 3:
                        result, dispatch_event, extra_media = dispatch_result
                    else:
                        result, dispatch_event = dispatch_result
                        extra_media = None
                    writer.write_event(dispatch_event)
                    result_blocks.append(result)
                    if extra_media:
                        media_blocks.extend(extra_media)
                    tool_time_s += dispatch_event.latency_ms / 1000.0
                    status_tag = "OK" if not result.is_error else "ERR"
                    _log(f"  <- {tu.name}: {status_tag} ({dispatch_event.latency_ms:.0f}ms)")

                # Message 1: tool results (becomes role:tool in OpenAI format)
                tool_msg = Message(role="user", content=result_blocks)
                messages.append(tool_msg)

                writer.write_event(TraceMessage(
                    trace_id=trace_id,
                    message=tool_msg,
                ))

                # Message 2: visual content (role:user with images, only if there are images)
                if media_blocks:
                    from ..models.content import ImageBlock as _IB
                    caption = TextBlock(text=f"[Visual content from tool results: {len(media_blocks)} image(s)]")
                    media_msg = Message(role="user", content=[caption] + media_blocks)
                    messages.append(media_msg)
                    writer.write_event(TraceMessage(
                        trace_id=trace_id,
                        message=media_msg,
                    ))
                    _log(f"  [media] injected {len(media_blocks)} image(s) into conversation")

                if recap is not None:
                    observation = recap.observe_tool_result(tool_msg)
                    recap_step = recap.after_tool_action(observation)
                    _write_recap_event(
                        trace_id,
                        writer,
                        recap,
                        remaining_subtasks=recap_step.remaining_subtasks,
                        note=recap_step.note,
                    )
                    if recap_step.prompt:
                        recap_msg = Message(role="user", content=[TextBlock(text=recap_step.prompt)])
                        messages.append(recap_msg)
                        writer.write_event(TraceMessage(
                            trace_id=trace_id,
                            message=recap_msg,
                            extra=dict(REX_INTERNAL_MESSAGE_EXTRA),
                        ))
                        _log(f"[recap] {recap_step.state.value}: current='{recap.current_task[:80]}'")
                elif reflection_mode:
                    reflection_pending = True
        except Exception as exc:
            loop_error = f"{type(exc).__name__}: {exc}"
            loop_exc = exc  # preserve original exception for re-raise
            if is_context_overflow_error(exc):
                context_overflow = True
                termination_reason = "context_overflow"
                loop_error = None
                loop_exc = None
                _log("[context-overflow] provider rejected an oversized request")
            else:
                _log(f"[error] agent loop failed: {loop_error}")

        # Fetch audit snapshots from mock services (best-effort)
        import httpx as _httpx

        for svc in task.services:
            if svc.reset_endpoint:
                audit_url = svc.reset_endpoint.rsplit("/reset", 1)[0] + "/audit"
                try:
                    resp = _httpx.get(audit_url, timeout=5)
                    writer.write_event(AuditSnapshot(
                        trace_id=trace_id,
                        service_name=svc.name,
                        audit_url=audit_url,
                        audit_data=resp.json(),
                    ))
                except Exception:
                    pass  # audit fetch is best-effort

        # Write trace end (always, even on error)
        wall_time = time.monotonic() - wall_start
        input_tok = total_usage.input_tokens
        output_tok = total_usage.output_tokens
        total_tok = total_usage.input_tokens + total_usage.output_tokens
        other_time_s = max(0.0, wall_time - model_time_s - tool_time_s)
        failure_modes = [loop_error] if loop_error else []
        writer.write_event(TraceEnd(
            trace_id=trace_id,
            total_turns=turn_count,
            model_input_tokens=input_tok,
            model_output_tokens=output_tok,
            input_tokens=input_tok,
            output_tokens=output_tok,
            total_tokens=total_tok,
            model_time_s=round(model_time_s, 2),
            tool_time_s=round(tool_time_s, 2),
            other_time_s=round(other_time_s, 2),
            wall_time_s=round(wall_time, 2),
            failure_modes=failure_modes,
            timed_out=timed_out,
            timeout_type=timeout_type,
            timeout_seconds=timeout_seconds,
            termination_reason=termination_reason,
            context_overflow=context_overflow,
            user_agent_rounds=user_agent_rounds,
            user_agent_max_rounds=ua_max_rounds,
            user_agent_done=ua_done,
        ))

        # Re-raise original exception so the caller (_run_single_task) can
        # match on exception type (e.g. APIConnectionError) for retry logic.
        if loop_error and loop_exc is not None:
            raise loop_exc

    _log(
        f"[end] turns={turn_count} tokens={total_tok} "
        f"({input_tok}in/{output_tok}out) "
        f"time=model {model_time_s:.1f}s tool {tool_time_s:.1f}s wall {wall_time:.1f}s"
    )

    dispatcher.close()
    return trace_path
