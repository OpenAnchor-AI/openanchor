# Anchor Project — Agent Dispatch Rules (formerly Fusio)

## Communication
- 中文优先（除非用户用英文）
- 简洁直接，no fluff
- 代码注释保持英文

## 3 Lane Dispatch (Code-First Architecture, 2026-07-04)

```
用户给一个 goal
    ↓
1. 我自己规划 (Codex M3, 1M context)
    ↓
2. 按 type 分派:
    │
    ├─ Routine code (≥100 LOC) → OpenCode free swarm (3 models parallel)
    │   - opencode/deepseek-v4-flash-free (主力)
    │   - opencode/mimo-v2.5-free (review)
    │
    ├─ 复杂架构 / 多文件 refactor / 1M context → 我自己做 (Codex M3)
    │
    ├─ 需要 advisor 视角 → spawn `claude --print --dangerously-skip-permissions "<prompt>"`
    │
    └─ Multimodal input → 我自己做 (Codex M3, view_image)
    ↓
3. 收集结果, 集成, 验证
    ↓
4. 端到端报告
```

## Anchor-Specific Rules

### 命名
- **包名**: `anchor` (小写, 不连字符)
- **CLI**: `anchor`
- **API 路径**: `POST /v1/chat/completions {"model": "anchor"}` (单产品，无 tier SKU)
- **Python 模块**: `anchor.config`, `anchor.server`, `anchor.clients`, `anchor.fusion_modes`, `anchor.head`
- **不允许**: `fugo-x`, `fugox`, `openfugu` 在代码中出现

### 文件结构
- 自写代码放 `src/anchor/`
- v0.7+ 不再维护 OpenFugu fork（已彻底分离）；research 仓库不再 submodule 到 src
- Research 仓库用 git submodule，**不直接 clone 到 src**

### 训练目标
- 永远不只是 quality, **永远 quality / cost** = Pareto
- head 在 `anchor/head.py`，calibration 在 `anchor/calibration.py`
- M3 当 oracle（**永远是 oracle**）
- head 训完用 git lfs 存

### Worker 配置 (v0.9.62 lean single-product pool)
- 加 worker = 改 `src/anchor/config.py:WORKERS` list (并更新 `src/anchor/workers.py` 常量)
- `enabled=False` 表示 blocked / env-gated / 冗余 opt-in
- fallback: `WORKER_NAME_FALLBACK` (runtime) + slot `FALLBACK_CHAIN` (role expansion)
- **所有 OpenAI-compat worker 用同一 `OpenAICompatClient`** (含 dpsk-pro / gpt-5.6-sol)
- M3 单独用 `KeyPool` (多 key round-robin: `MINIMAX_API_KEY_1`/`_2`)
- **公开产品**: 仅 `model: "anchor"` (+ `anchor-image`)；无 basic/premium/ultra SKU
- **Worker 池 (15 configured, 9 enabled default)**:
  - enabled: minimax-m3 / deepseek-v4-flash / deepseek-v4-flash-official / claude-sonnet-5 / claude-opus-5 / grok-4-6-reasoning / gpt-5.6-luna / gpt-5.6-terra / kimi-k3
  - opt-in (env-gated): claude-haiku-4-5 (`ANCHOR_ENABLE_HAIKU=1`) / gpt-5.6-sol (Phase 3 FIX-E HOLD, 11% err) / grok-4-5-reasoning (`ANCHOR_ENABLE_GROK4=1`, 4-6 supersedes) / ollama-ornith-35b (`ANCHOR_ENABLE_OLLAMA_ORNITH=1`)
  - gradual: claude-fable-5 (`FABLE5_TRAFFIC_PCT=25`, sacred auto via Phase 3a)
  - opt-in (B6 audit 2026-08-15): deepseek-v4-pro (`ANCHOR_ENABLE_DEEPSEEK_PRO=1`, legacy env, 仅 calibration 用; Opus-5 ¥1.58/M 在 T3 hard Pareto 占优, dpsk-pro ¥9/M 仅作 second-source fallback)
- **prior**: 9 enabled × 12 qt = **108**
- **资源边界**:
  - Baosi Claude group (fable + opt-in sonnet/opus/haiku): shared ¥688/mo + $15000, `api.baosiapi.com`
  - Baosi GPT group (sol + luna/terra): independent, `baosiapi.com/v1`
  - DeepSeek Zen free ≠ DeepSeek official pro
  - **v0.9.76+: Zen dual-key shares quota** — two OpenCode Zen API keys added
    on 2026-08-19 both 429'd at the same instant (`Retry-After: 45209s`
    identical down to the second). Round-robin would burn key_2 every time
    key_1 hit the cap, so `deepseek-v4-flash` uses `key_strategy="failover"`
    (pin key_0, advance to key_1 only when key_0 is in per-key cooldown).
    Re-probe before assuming independent quotas.
- **Image-gen**: `agnes-image-2.1-flash`
- Direct worker alias 保留但 **disabled → 400**；不进 `/v1/models`

### 验证
- 改完任何 Python 代码: `PYTHONPATH=src python -c "import anchor"`
- 改完 config: `PYTHONPATH=src python -m anchor.config`
- 改完 client: `PYTHONPATH=src python -m anchor.clients.<module>`
- 改完 routing: 跑 `tests/test_routing.py`
- 改完 gateway: `curl localhost:8088/v1/chat/completions -d '{"model":"anchor","messages":[...]}'`

### Anti-patterns (禁止)
- ❌ 不要在远程沙箱内假设 Claude Code / OpenCode 能跑 (无网) — macOS 本地开发不受限
- ❌ 不要 back-to-back 派同 OpenCode 模型 (20min 限速)
- ❌ 不要 `cd <dir> && cmd`, 用 `workdir` 参数
- ❌ 不要 inline ≥100 LOC 代码, 必须写文件
- ❌ 不要把 `fusio` / `fugo-x` / `openfugu` / `fugox` 留在 anchor 代码里
- ❌ 不要把 LiteLLM 当 anchor 的核心 (M3 可以, 其他不行)
- ❌ 不要 `import openfugu` 在 anchor 自写代码里 (v0.7+ 已不再维护 fork)

### 沙箱状态 (远程服务器, 2026-07-05 — 本地 macOS 不适用)
- ✅ mihomo 代理 `127.0.0.1:7890` (openai.com 后缀)
- ✅ apihub 直连 (`apihub.agnes-ai.com` 在 no_proxy)
- ✅ baosiapi 直连 (`api.baosiapi.com` 在 no_proxy)
- ✅ OpenRouter 公开 API
- ✅ comfyenv torch 2.12.1+rocm7.2 (`/home/dongshenglu/comfyenv/`)
- ✅ Ollama `qwen3-embedding:8b` 本地
- ✅ Docker pgvector + redis 在跑
- ❌ ANTHROPIC_API_KEY 走 baosiapi 无效
- ❌ AGNES_API_KEY 部分时段 timeout (网络问题)
- ❌ OpenRouter key 未注册

## 关键命令

```bash
# 项目根
export ANCHOR_ROOT=$PWD
source $ANCHOR_ROOT/.venv/bin/activate

# 验证
PYTHONPATH=src python -m anchor.config
python -m pytest tests/

# 跑 gateway
PYTHONPATH=src python -m anchor.server

# 跑 RouterArena 评测
PYTHONPATH=src python -m anchor.eval

# Weekly retrain
bash scripts/weekly_retrain.sh
```

## 决策记录

### 2026-08-24 (anchor 运维: per-worker timeout env override + Zen dpsk-flash quarantine)
**触发**: hermes 96K-token 大上下文请求连续 HTTP 424。第一波 `all_workers_failed` (dpsk-flash 上游挂), 第二波 m3/luna/sonnet/terra 接连 `TimeoutError`。
**根因 1 (timeout)**: `routing_core.py:_worker_call_timeout` 默认值太小 (minimax-m3 25s, 其他 15s)。96K prompt 的 prefill 就要 30s+, 所有 worker 在 timeout 内完不成 → 全部超时 → 424。
**修复 (vps `.env`, 无代码改动)**:
```
ANCHOR_WORKER_TIMEOUT_DEFAULT=120
ANCHOR_WORKER_TIMEOUT_MINIMAX_M3=180
ANCHOR_WORKER_TIMEOUT_DEEPSEEK_V4_FLASH=90
```
`.env` 是 gitignore 的本地配置; override 逻辑本身已存在于 `_worker_call_timeout` (env key 格式: `ANCHOR_WORKER_TIMEOUT_<NAME大写下划线>`)。
**根因 2 (dpsk-flash 上游挂)**: OpenCode Zen 平台侧 `deepseek-v4-flash-free` "Model is unavailable" (双 key 都复现, 非 anchor 问题)。
**处置**: 手动 quarantine — 写 `/root/anchor/data/circuit_breaker.json` 的 `err_rate_quarantine.deepseek-v4-flash.quarantined=true` (reason 标注 manual + 日期)。路由热加载 (`is_quarantined()` 每次读 json), 无需 restart。
**验证**: quarantine 后 primary 直接选 minimax-m3 无 fallback 延迟; timeout 调整后 12K-token 测试 3.8s 正常返回。
**Zen 恢复后解除 quarantine** (vps):
```bash
python3 -c "
import json
p = '/root/anchor/data/circuit_breaker.json'
s = json.load(open(p))
s['err_rate_quarantine']['deepseek-v4-flash']['quarantined'] = False
json.dump(s, open(p, 'w'), indent=2)
"
```
**要点**: 大上下文客户端 (hermes/codex 长会话) 必须配大 timeout; 免费上游 (Zen) 故障时用 circuit_breaker.json 手动 quarantine 是标准处置, 热加载即时生效。

### 2026-08-21 (Claude Code 4 节点网络修复: baosiapi DNS 被污染 → no_proxy 豁免导致直连超时)
**触发**: 检查/对齐 4 节点 claude code (本机/vps/aimax/macmini) 时, claude 全部 `Execution error` / 卡住超时 (除 vps)。curl 诊断: 本机/aimax/macmini 直连 `baosiapi.com` 全部 `000` (立即超时), 走本地代理 (本机 7897 / aimax 7890 / macmini 7898) 则 `200`。
**根因 (DNS 层劫持, 不是 baosiapi 需要代理)**:
- 所有公共 DNS (8.8.8.8 / 1.1.1.1 / 223.5.5.5 / 114.114.114.114) 把 `baosiapi.com` 解析到 **Facebook/Meta IP 段** (31.13.x / 108.160.x / 199.59.x / 173.252.x)。
- 只有 vps 上解析到真实 IP `43.160.214.116` (`api.baosiapi.com` → `43.161.251.248`), 直连 200。
- 三台本机 claude 直连 → 解析到污染 IP → 连不上; **clash/mihomo 代理有独立 DNS, 能正确解析 baosiapi 真实 IP** → 走代理才通。
- 修复前 `.zshrc`/`.bashrc`/`.profile` 的 `no_proxy` 豁免了 `*.baosiapi.com,baosiapi.com` → claude 绕过代理直连 → 超时。
**修复 (每节点)**:
- **本机** (`~/.zshrc`): no_proxy 移除 baosiapi 豁免 (只留 `localhost,127.0.0.1,100.91.82.25,100.76.189.39,45.197.146.62,huggingface.co`); 代理 env 7897 (Clash Verge, 系统代理已启用)。
- **aimax** (`~/.bashrc` + `~/.profile` + `~/.zshrc` 三处): 全部移除 baosiapi 豁免; 代理 env 7890 (mihomo)。
- **macmini** (`~/.zshrc`): 移除 baosiapi 豁免; 代理 env 7898 (注意 macmini 7897 是 dead 端口, 用 7898)。
- **vps**: 无 no_proxy 问题 (直连 baosiapi 200), 只需统一 base_url + 补 key。
**对齐规范 (4 节点统一)**:
- `ANTHROPIC_BASE_URL="https://baosiapi.com"` (vps 原本用 `api.baosiapi.com`, 已统一)
- `ANTHROPIC_API_KEY="sk-gW6e7Ri..."` (vps 原本无 key, 已补入 `.bashrc`)
- `ANTHROPIC_MODEL="claude-sonnet-5"`; claude code 版本 2.1.220 (vps 2.1.197 待升)
- macmini 原本无 claude 二进制 (只有 settings), 已从本机复制 `claude.exe` (arm64 单文件, 245MB, npm 包 `@anthropic-ai/claude-code`) → `~/.local/bin/claude`。
- 验证: 4 节点 `claude -p "Reply with exactly: CLAUDE-OK"` 全返回 `CLAUDE-OK`。
- 备份: 各节点 `.zshrc`/`.bashrc`/`.profile` 均有 `.bak-20260821`。
**要点**: baosiapi 走代理是**环境特有** (本机/aimax/macmini 的 DNS 被劫持, 只有 clash/mihomo 代理能正确解析), 非 baosiapi 本身需要代理; vps 始终可直连。改 no_proxy 时**不要**再加回 baosiapi 豁免。

**补充 (2026-08-21 再测)**: 本机直连 baosiapi **不只是 DNS 问题** — 强制 vps 解析到的正确 IP (`43.160.214.116` / `43.161.251.248`) 直连时: TCP 443 建连成功、证书 Verify OK, 但 **TLS ClientHello 后立即 `Connection reset by peer`** (0.02-0.05s)。即本机出口网络到 baosiapi 的 **TLS 层被中间设备阻断** (疑似 SNI 过滤/出口 ACL), 只有走 clash 代理换出口才通。对照: 本机直连 `163.com` 正常, vps 直连同一 IP 200。结论: baosiapi 必须走代理是**出口网络层阻断**, 不是 baosiapi 需要代理。

### 2026-08-24 (WireGuard 全链路验证: 4 节点 mesh + Anchor API 真实调用)
**触发**: 在 4 节点 baosiapi claude 对齐基础上, 验证 WireGuard UDP mesh 跨节点延迟 + Anchor API 真实调用耗时, 确认 m3 worker 路径全打通。
**WireGuard ping 矩阵 (ms)**:

| 源 \ 目标 | vps (10.0.0.1) | 本机 (10.0.0.2) |
|---|---|---|
| 本机 | 201 | — |
| macmini | 39 | 509 |
| aimax | 52 | 234 |

**节点物理位置 + ISP**:
| 节点 | 物理位置 | 网络 |
|---|---|---|
| 本机 | 新加坡 | 中国移动 (漫游/海外流量) |
| vps | 香港 | Ansheng |
| macmini | 中国 (北京/上海) | 中国移动 |
| aimax | 中国 (上海) | 中国电信 |

**最佳链路 (含物理归因)**:
- macmini ↔ vps: **39ms** (国内 → 香港跨境, 直连)
- macmini ↔ aimax: 91ms (国内 → 经 vps 中转)
- aimax → vps: **52ms** (udp2raw 链路稳定, 跨 ISP 抗 UDP 封锁)
- 本机 → vps: 201ms (新加坡 → 香港, **物理跨境延迟**, 不可软件优化)
- macmini → 本机: 509ms (国内 → 新加坡, **跨境 + 路径绕**, 物理瓶颈)
- aimax → 本机: 234ms (国内 → 新加坡, 合理跨境延迟)

**Anchor API 真实调用 (M3 worker, 走 baosiapi claude 后端)**:
- 本机: **3478ms**
- macmini: **1830ms** (最优 — 地理近 + 路径短)
- aimax: **2569ms**

**观察 / 软件层 vs 物理层边界**:
- **软件层 ✅ 已最优**: WireGuard UDP 直连 + aimax udp2raw 抗 ISP 封锁 + base_url/no_proxy 全部对齐 + 4 节点 API 真实调用通 + 1264 tests 全绿 + Tailscale DERP 425ms 已下线 (比 WG 慢 1.6-2x 平均, 4x+ best case)
- **物理层 ⚠️ 不可软件优化**: 新加坡-香港 201ms / 国内-新加坡 509ms 都是光速 + 跨境 ISP 路由决定
- **进一步降延迟路径**: 加新加坡 WG 中转 VPS, 本机 → vps 可降到 <50ms, 但需要新 VPS 月费。
- macmini 端到端 1830ms < 本机 3478ms, **本机 API 路径有额外开销** (公网绕路 / 出口 NAT / clash 代理多一跳), 不是 anchor 服务侧慢。
- vps 充当 mesh hub (10.0.0.1), 跨节点流量默认经 vps 中转。
- aimax 100.91.82.25 走 udp2raw 抗中国电信 UDP 封锁, 链路 52ms 稳定。
**要点**: 4 节点 WireGuard + baosiapi 修复叠加, 全链路达成; macmini 是当前最快的 M3 调用节点 (1830ms), 本机次之 (3478ms, 受新加坡地理位置限制), aimax 用于 GPU 类大任务 (ollama ornith-35b-q8-agent 262K ctx / ROCm 7.2.2)。软件层已无优化空间, 剩余延迟 = 物理距离 + 跨境 ISP 路由。
**下一步候选**:
1. **加 ollama worker (aimax 10.0.0.4:11434)** — 大任务 GPU 加速, 零边际成本
2. **延迟监控告警 cron** — 4 节点延迟超过阈值自动告警, 防回归
3. **新加坡 WG 中转 VPS** — 降本机→vps 201ms → <50ms, 但需月费

### 2026-08-19 v0.9.76 (stream fallback 链修复: 三层 cascade bug)
**触发**: 本机 `opencode run` (默认 `anchor/anchor`) 间歇 `UnknownError`; curl stream 只回角色帧 + `error` event 无内容。诊断根因链: m3 key_1 (XOSUTM) Token Plan 429 → stream 无 key failover → server 无 worker 级 fallback chain → 且 fallback 块有 `UnboundLocalError`。**4 个 commit 串联修复**:

1. **Stream worker 级 fallback chain** (`b3c115d`) — `server.py::_gen` stream 路径原本 `select_only` 只选主 worker, 失败直接 emit SSE error 无 cascade。
   - **修复**: `Response.fallback_chain` 字段 (`api_models.py`); `routing_core.py::_route_tier` 在 select_only 模式返回完整 chain; `_gen` 构建 `_stream_chain` 失败时 cascade (m3→luna→sonnet→terra→grok), 仅全部失败才 emit error event。`STREAM_FALLBACK primary=X -> next=Y err=Z` warning 保留 audit trail。

2. **KeyPool.stream() key failover parity with chat()** (`de12559`) — `chat()` 已有 failover loop 但 `stream()` 死守 avail[0], m3 key_1 429 时 key_2 (健康) 永远用不到。
   - **修复**: `stream()` 补全 key failover loop (`_tried_keys` set, RateLimit/Timeout/Connection/Auth/5xx → cooldown → next key; 耗尽抛 `AllKeysRateLimited`)。**后续整改: stream 与 chat 共用同一 failover 逻辑避免双路径漂移。**

3. **stream() key_2 重试 UnboundLocalError** (`86f5f54`) — except 块 `_clamped` 只在 `_ra is not None` 分支赋值, 无 Retry-After 的 429 时 `_clamped` 未绑定。
   - **修复**: 改 `_sf_secs` 用无条件表达式 `(_clamp_retry_after(_ra) if _ra is not None else self._PER_KEY_COOLDOWN_SECS)`。

4. **server _gen fallback 块 UnboundLocalError** (`8fa9289`) — `import re as _re` 在 while 循环体内, 主 worker 首帧前失败时 except 块 (STREAM_FALLBACK) 引用 `_re` 未绑定。
   - **修复**: `import re as _re` 提到 `_gen` 顶部 (try 块外)。

**验证**: 本地 1265 tests 全绿; vps 重启后 m3 429 期间 stream 返回完整内容 (2615 bytes) 无 error event; 新 UnboundLocalError 计数 0。全部 push github + vps (HEAD `8fa9289`)。

### 2026-07-25 v0.9.52-p3 + p4 (M3 / dpsk-flash / dpsk-pro 400 BadRequestError 根因修复)
**触发**: 用户报告 2 个 400 — `[error:minimax-m3:BadRequestError]` + `[error:deepseek-v4-flash:BadRequestError]`. `WORKER_ERR` 完整 log 暴露根因。
**5 个独立 root cause + 1 个 pre-existing typo**:

1. **M3 stream leak `<think>...</think>`** — `extra_body={"thinking":{"type":"disabled"}}` 在 stream 模式被 M3 忽略, 推理块泄到客户端。
   - **修复**: `clients/base.py` + `clients/key_pool.py` 各加 module-level `_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)`, 在 stream chunk 出口 strip。非流式 `_call()` 路径已在 0.9.24 加过同样 strip, 本次对齐流式行为。
   - **Perf**: 之前每次 `stream()` 调用都 `import re as _re; _re.sub(...)`, 提到 module top-level 后 compile 一次。

2. **`role="function"` (OpenAI 1.x deprecated) 被 M3 / dpsk 拒** — `BadRequestError`。
   - **修复**: `routing_text.py::normalize_openai_messages` 把 `function` 转为 `tool`, 保留 `name` + `tool_call_id` + `content`。

3. **`n > 1` 被 M3 拒** — `BadRequestError: does not support n > 1`。
   - **修复**: `server.py::v1_chat_completions` 检测 `req.n > 1` → `req.n = None` (从 forward payload 中移除); 同时从 `extra_kw` 字典推导里删除 `"n"` key (避免 `n=1` 也被转发触发某些 vendor edge case)。`N_GT_1_TRUNCATED requested=N delivered=1` warning 保留 audit trail。

4. **Orphan tool message 引发 M3 / dpsk-pro 400** — `tool result's tool id(call_x1) not found (2013)` / `Messages with role 'tool' must be a response to a preceding message with 'tool_calls'`。
   - **修复**: `routing_text.py::normalize_openai_messages` 追踪已发出的 `assistant.tool_calls[].id` 集合, 遇到 orphan tool message (`tool_call_id` 不在集合) 自动注入 synthetic assistant tool_calls anchor: `{role:assistant, content:null, tool_calls:[{id, type:function, function:{name, arguments:"{}"}}]}`. `name` 优先用 `tool.name`, fallback `"unknown"`; 重复 `tool_call_id` 只注入一次。

5. **OpenAI Responses `function_call_output` 缺 `call_id`** — M3 同上 400。
   - **修复**: `server.py::_responses_input_to_messages` 生成 `call_fallback_{len(messages)}` 兜底 id。

6. **Pre-existing `predict` NameError** — `routing_core.py:366` 用裸名 `predict` 但 module top-level 只 import 了 `predict as _head_predict`。无人触发因为 production 路径不走到该行; 单元测试一旦命中就 NameError。Stash 验证是 main 分支已存在 13+ 天的 bug。
   - **修复**: 改用 `_head_predict(...)` (line 46 alias)。解锁 2 个 stale 测试 (`test_pareto_quarantine.py::test_server_filters_quarantined_workers` + `test_server.py::test_anchor_chat_cn`)。

**额外修复** (`routing_core.py::_call_worker`):
- 之前 worker error 只回 `[error:worker:Type] msg[:60]` 给用户, 没 log。→ 加 `WORKER_ERR worker=X err_type=Y full_msg=Z` warning, 完整上游错误进 `/tmp/anchor-gateway.log`。Operators 可以诊断 root cause。

**根因诊断 (`deepseek-v4-flash` "Console Upstream request failed")**:
- 用 OpenAI SDK 直接 hit OpenCode Zen dpsk-flash-free, 测了 7 种 tool message 形态, 发现 OpenCode Zen 的 "Console" 中间层只接受 strict pattern: `assistant(tool_calls) + tool(id_matches)`. 任何 orphan / 不匹配形态 → 400 "Error from provider (Console)".
- **`thinking=disabled` 是必需的** (已在 key_pool.py:259 注入; `_THINK_FIRST_NAMES = ("minimax","m3","deepseek")` 覆盖 dpsk).
- **确认**: 走 anchor 全链路 (`normalize_openai_messages` → factory → KeyPool.chat`) 后, orphan case 自动被 synthetic anchor 修复 → 200 OK. **orphan fix 同时解决 M3 + dpsk-pro + dpsk-flash 三家 strict provider.**

**测试**: `tests/test_v0952_p3_compat_fixes.py` 新增 **22 unit tests** (15 原始 + 7 orphan):
- <think> strip: closed / multiple / unclosed partial / clean / cross-module invariant
- function → tool: basic / preserve tool_call_id / developer→system / empty
- n>1: end-to-end strip + static `extra_kw` excludes n
- call_fallback: 3-way uniqueness / explicit `call_id` / `tool_call_id` field
- WORKER_ERR: full message via patched `factory.build_client`
- **orphan anchor (p4)**: inject / no synthetic for anchored / mixed real+orphan / legacy function role / unknown name / no tool_call_id no anchor / no double-inject for repeated id

**Smoke test (PID 55423, 新 bytecode)**:
- M3 stream "READY" → 干净返回, 无 `<think>` leak
- Orphan tool message → 之前 400, 现在 `"42"` 干净返回 via dpsk-flash fallback
- 真实 tool_calls 对话 (user → assistant tool_calls → tool) → `"42"` 干净
- `n=3` → `N_GT_1_TRUNCATED` logged, n 被 strip

### 2026-07-22 v0.9.50-p1 (Gate probe results + probe hardening)
- **Fable 5 Gate-2 probe PASSED** (placeholder_rate=0.0% on 50 stratified samples vs opus-4-8; `data/probe_fable5_results.json`). Per spec <5% → enable: 默认 FABLE5_TRAFFIC_PCT 10% → **100%**. baosiapi 已修复 vendor placeholder echo.
- **DeepSeek V4 Pro Gate-1 probe HOLD** (placeholder_rate=13.3% on 30 stratified samples vs sonnet-5; `data/probe_deepseek_pro_results.json`). Per spec 5%–20% → hold, `ANCHOR_ENABLE_DEEPSEEK_PRO=0` 默认计划。**但 v0.9.53 实际 override 为 1** — 用作 M3 fallback (用 `DEEPSEEK_API_KEY` 独立 2-key pool)，不再等 Gate-1 retry。placeholder 风险接受。
- **Probe 脚本加固** (`probe_fable5.py` + `probe_deepseek_pro.py`): 加 `asyncio.wait_for(..., timeout=45.0)` per-call,捕获 `APIError` + `TimeoutError` 防止单点 vendor hang 拖垮 probe。

### 2026-07-22 v0.9.50-p0 (resource-boundary cleanup)
- **DeepSeek 双轨**: Zen free (`deepseek-v4-flash`) vs 官方 (`deepseek-v4-pro`)
- **GPT-5.6 Sol** 加入 Baosi GPT 组 (独立 quota, baosiapi.com host)
- **Fable 5 gradual rollout**: 默认 10%→100% (Gate-2 PASS; 见 v0.9.50-p1)
- **M3 cost**: ¥119/mo / 18B tokens (was ¥238/mo)
- **Nemotron 移除**: probe inconclusive; script + test 删除

### 2026-07-05 (initial)
- **名字**: Anchor (formerly Fusio; 跟 fugo-x 完全分离, 不再回头)
- **定位**: 公开版 Sakana Fugu + 商业化 API
- **worker 池 (2026-07-05)**: 9 worker (8 enabled) — Agnes / DeepSeek V4 / M3 / Sonnet 5 / Gemini 3.5 / Kilo / Opus 4.8 / Haiku 4.5
- **M3 客户端**: 自建 `KeyPool` (多 key round-robin), 不用 LiteLLM
- **其他 worker**: `OpenAICompatClient` 统一抽象
- **训练 reward**: `J(θ) = E[quality] - α·cost`, 1 head + 3 α
- **路径**: 渐进 (MVP 自建 → 商业化扩展)
