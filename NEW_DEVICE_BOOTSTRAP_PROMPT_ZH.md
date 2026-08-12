# 新设备一轮对话完整恢复提示词

将下方代码块完整复制到新设备的 Codex。执行前应先在新设备登录 GitHub，并确保
账号能够读取三个私有/正式仓库。

```text
我要在这台新设备上完整恢复并继续开发 rPPG 项目。请在本轮对话中持续执行到
安全完成或遇到必须由我处理的真实阻塞；不要只给命令而不执行。

仓库：

- 项目迁移备份：https://github.com/Lance9-svg/rppg-pipeline-migration.git
- 后续正式开发：https://github.com/Lance9-svg/rppg-pipeline.git
- 项目自定义 Skills：https://github.com/Lance9-svg/codex-research-skills.git
- 目标分支：codex/research/ubfc-reliability-experiment
- 迁移基线提交至少应包含：0e9f422 和 46576ee

全局安全规则：

1. 先检查再执行，不假设任何步骤成功。
2. 不覆盖、删除或 reset 新设备已有的未提交内容。
3. 禁止 `git reset --hard`、`git clean -fd`、重写历史和 force push。
4. 不提交或上传数据集、参与者数据、模型、运行输出、secrets、凭据、虚拟环境、
   缓存、浏览器状态或插件缓存。
5. 不回显 token、密码、API key、cookie 或其他 secret 的实际值。
6. 只有实际运行成功后才能声称测试、Ruff、安装或同步完成。
7. 后续正式 pull、push、branch 和 PR 使用原 `rppg-pipeline`；migration 仅作备份。

请按以下顺序执行。

一、发现并保护现有状态

1. 检查当前目录、可用磁盘空间、Git、GitHub 登录、Python 3.13 和 Codex 状态。
2. 如果已有同名项目，运行：
   `git status --short --branch`、`git remote -v`、`git branch -vv`、
   `git log -5 --oneline --decorate`。
3. 如有未提交修改或未知 remote，停止会覆盖它的操作，先报告并保护现状；其余
   不冲突的检查可以继续。

二、恢复项目与远端关系

如果项目尚未存在：

```powershell
git clone https://github.com/Lance9-svg/rppg-pipeline-migration.git
Set-Location rppg-pipeline-migration
git switch codex/research/ubfc-reliability-experiment
```

核对 `0e9f422`、`46576ee` 以及 `MIGRATION.md`、`AGENTS.md`、
`NEW_DEVICE_BOOTSTRAP_PROMPT_ZH.md`。然后把 remote 配置为：

- `origin` = `https://github.com/Lance9-svg/rppg-pipeline.git`
- `migration` = `https://github.com/Lance9-svg/rppg-pipeline-migration.git`

克隆迁移仓库后的通常命令为：

```powershell
git remote rename origin migration
git remote add origin https://github.com/Lance9-svg/rppg-pipeline.git
git fetch origin
git fetch migration
git remote -v
```

若 remote 已存在，按实际 URL 安全调整，不覆盖未知配置。完整阅读并遵守：

- `AGENTS.md`
- `MIGRATION.md`
- `README.md`
- `CODING_AND_GIT_POLICY.md`（如果仓库内存在）

三、恢复 23 个项目自定义 Skills

不要复制旧设备插件缓存。将 Skills 私有仓库克隆到项目之外的临时/工具目录：

```powershell
git clone https://github.com/Lance9-svg/codex-research-skills.git
```

先检查其 `README.md`、Git 状态和提交历史。验证：

- 共有 23 个 `SKILL.md`；
- 不包含嵌套 `.git`、`.env`、凭据、虚拟环境和缓存；
- 每个 Skill 引用的本地 scripts、references、assets、templates 都存在。

将 `codex-research-skills/skills` 下每个完整目录复制到当前 rPPG 项目的
`.codex/skills`。不要只复制 `SKILL.md`，不要覆盖已有同名目录。若已有同名 Skill，
先逐文件比较；一致则保留，存在差异则报告并等待，不要猜测合并。

目标应包括：academic-research-suite、humanizer、literature-review、
nature-academic-search、nature-citation、nature-data、nature-downloader、
nature-experiment-log、nature-figure、nature-literature-pipeline、nature-paper2ppt、
nature-paper-card、nature-paper-to-patent、nature-polishing、nature-reader、
nature-ref-verifier、nature-response、nature-reviewer、nature-shared、
nature-statistics、nature-writing、paper-spine、researchwrite。

注意两个显示名与目录名映射：

- `literature-review` 位于 `scientific-thinking-literature-review/`
- `researchwrite` 位于 `nature-proposal-writer/`

安装后重新扫描项目 `.codex/skills/**/SKILL.md`，列出实际恢复的 23 项。说明这些
Skills 是否需要新一轮对话或重启 Codex 才会加载。

四、恢复系统与插件 Skills

系统 Skills（imagegen、openai-docs、plugin-creator、review-agent、skill-creator、
skill-installer）应由当前 Codex 提供；缺失时检查 Codex 版本，不从不明来源下载。

检查并通过 Codex 官方插件系统重新安装当前任务所需的插件，不复制
`.codex/plugins/cache`。优先恢复：Browser/Chrome/Computer Use、documents、pdf、
Presentations、Spreadsheets、template-creator、sites、visualize、Canva、Adobe 和
Superpowers。某插件不可用时记录名称与原因，不安装不明替代品。

五、重建 Python 环境

项目记录使用 Python 3.13.4。检查 `py -0p` 和 `py -3.13 --version`，不要复制旧
`.venv`。执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m rppg_pipeline --help
git diff --check
git status --short --branch
```

六、检查手动资产但不上传

检查我是否已提供：

- UBFC-rPPG Dataset 2，结构为 `<dataset-root>/subjectN/vid.avi` 与
  `<dataset-root>/subjectN/ground_truth.txt`；
- `face_landmarker.task`，预期 SHA-256：
  `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`；
- 可选历史 `local_outputs/`。

缺失时只报告，不从来源不明的位置下载。不要运行完整 42-subject 实验。资产就绪后
只准备 subjects 1 和 3 的 smoke-test 命令，输出目录必须是新的；等我确认路径后再
运行。

七、把迁移分支接回正式仓库

```powershell
git fetch origin
git log --oneline --left-right `
  origin/codex/research/ubfc-reliability-experiment...codex/research/ubfc-reliability-experiment
```

仅当推送是正常 fast-forward 且不会覆盖远端新提交时：

```powershell
git push origin codex/research/ubfc-reliability-experiment
git branch --set-upstream-to=origin/codex/research/ubfc-reliability-experiment `
  codex/research/ubfc-reliability-experiment
```

被拒绝时禁止 force push；fetch 后报告分叉。migration 不设为正式 upstream。以后：

```powershell
git pull --ff-only origin codex/research/ubfc-reliability-experiment
git push origin codex/research/ubfc-reliability-experiment
```

八、最终核验与报告

实际核验三个仓库 URL、当前 commit、remote refs、工作树、pytest、Ruff、CLI、
23 个项目 Skills、系统/插件 Skills 和手动资产。最后报告：

1. 项目与 Skills checkout 的绝对路径；
2. 当前 branch/commit 和 origin/migration URL；
3. 是否与正式仓库同步；
4. Python、依赖、pytest 实际通过数、Ruff、CLI；
5. 已恢复/缺失/需重启的 Skills；
6. 数据集、模型哈希和历史输出状态；
7. secrets/大文件/忽略规则审计结果；
8. 当前工作树是否干净、是否可以继续开发；
9. 剩余风险和下一步唯一建议。

不要声称未实际完成的安装、测试、实验、人工审查或同步已经完成。
```

