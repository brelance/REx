# OpenCode model provider

The `opencode` provider connects to an OpenAI-compatible endpoint and adds the `x-opencode-session` header required by OpenCode-backed account pools. Configure the New API endpoint and token through `OPENCODE_BASE_URL` and `OPENCODE_API_KEY`:

```bash
export OPENCODE_BASE_URL=http://127.0.0.1:3000/v1
export OPENCODE_API_KEY=your-new-api-token

inspect eval inspect_evals/<eval-name> \
  --model-role 'grader={model: opencode/deepseek-v4-flash, temperature: 0}'
```

Within an Inspect evaluation, the provider derives a deterministic UUID from the eval, run, task, sample, epoch, model, and `session_namespace`. Calls from the same sample and judge therefore retain their session across API and sample retries, while independent samples receive different sessions. Outside an active sample, the provider uses one process-local session per model instance.

Use `session_namespace` to separate independent uses of the same model within one sample:

```bash
inspect eval inspect_evals/<eval-name> \
  --model-role 'grader={model: opencode/deepseek-v4-flash, model_args: {session_namespace: primary-grader}}'
```

For targeted debugging, `session_id` supplies a fixed session. An explicit `x-opencode-session` entry in `GenerateConfig.extra_headers` takes precedence over automatic generation.
