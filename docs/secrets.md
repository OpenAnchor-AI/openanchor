# GitHub Actions Secrets — OpenAnchor-AI/openanchor

本文件只记录 secret 名称与用途，不记录真实值。

## 当前结论

- 仓库现有 CI 工作流：`.github/workflows/pytest.yml`
- 该工作流**不引用任何 `secrets.*`**，因此当前 CI 不需要额外 secrets 即可启动。
- 经 `gh secret list` 核对，仓库当前**无已配置 secret**。

## 若后续工作流引入 secrets，建议按此登记

| Secret | 用途 | 备注 |
|---|---|---|
| `ANCHOR_API_KEYS` | 网关客户端鉴权 | 逗号分隔多 token |
| `OPENCODE_ZEN_API_KEY` | DeepSeek V4 Flash 主 key | Zen 免费池 |
| `OPENCODE_ZEN_API_KEY_2` | DeepSeek V4 Flash 备用 key | Zen 免费池 |
| `DEEPSEEK_API_KEY` | DeepSeek 官方 API | dpsk-pro / dpsk-flash-official |
| `MINIMAX_API_KEY_1` | MiniMax M3 主 key | 多 key round-robin |
| `MINIMAX_API_KEY_2` | MiniMax M3 备用 key | 多 key round-robin |
| `BAOSIAPI_API_KEY` | Baosi 旧/通用 key | legacy |
| `BAOSIAPI_GPT_API_KEY` | Baosi GPT 组独立 key | Sol / Grok |
| `ANCHOR_BAOSIAPI_CLAUDE_API_KEY` | Baosi Claude 组 key | Fable/Sonnet/Opus/Haiku |
| `AGNES_API_KEY` | 图像生成 key | image-gen endpoint |

## 设置方式

```bash
gh secret set SECRET_NAME --repo OpenAnchor-AI/openanchor
```

## 验证方式

- 空 PR 触发 Actions：`gh pr create --repo OpenAnchor-AI/openanchor ...`
- 查看运行：`gh run list --repo OpenAnchor-AI/openanchor`
