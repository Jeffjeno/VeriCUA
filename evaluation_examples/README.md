# OSWorld Task Definitions

Each manifest maps domain names to task IDs. Task definitions live at
`examples/<domain>/<id>.json` and contain the instruction, environment
setup and native evaluator configuration.

| Manifest | Tasks | Consumer |
|---|---:|---|
| `rl_tasks.json` | 280 | Formal judge RL and reward ablation launchers |
| `sft_tasks.json` | 287 | SFT collection configurations |
| `osworld_all.json` | 369 | OSWorld benchmark and actor rollout defaults |
| `osworld_no_drive.json` | 361 | Qwen, EvoCUA and standalone replay defaults |
| `service_tasks.json` | 35 | Environment service fallback default |

The benchmark evaluation suite uses the separate downloaded benchmark manifest at
`data/osworld/manifest.json`, as specified in its dataset
configuration. The manifests here do not replace that evaluation set.

`settings/` contains placeholder service configuration. Supply credentials
locally through the relevant environment or settings before running tasks
that require those services. VM images and downloaded task assets are not
bundled.
