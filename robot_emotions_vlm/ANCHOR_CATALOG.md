# Anchor Catalog

Documentacao do modulo `robot_emotions_vlm.anchor_catalog` e do fluxo window-level Qwen → Kimodo.

## O que faz

O comando `build-anchor-catalog` cria um catalogo por janela a partir de:

- `pose3d_manifest.jsonl` exportado do ramo real
- `kimodo_window_prompt_catalog.jsonl` exportado pelo Qwen por janela

Para cada janela, ele:

- seleciona a janela temporal exata em `pose3d.npz`
- interpola a trajetoria 2D da raiz (`root_translation_m`) no grid de frames do Kimodo
- rebasa para que a primeira raiz comece em `x=z=0` e converte para o sistema de coordenadas do Kimodo (`-X`, `-Z`)
- escreve `constraints.json` com `root2d` denso e, opcionalmente, end-effectors esparsos
- salva `traceability.json`
- escreve uma entrada em `kimodo_anchor_catalog.jsonl`

## Contrato atual

Por padrao (`--effector-keyframes 0`), a unica constraint e `root2d`-only: ancora a trajetoria de chao da captura real e deixa o modelo gerar a pose livremente a partir do prompt textual.

- `root2d` denso (um ponto por frame do Kimodo), interpolado da trajetoria real
- janelas quase estaticas (`root2d_net_displacement_m < 0.05`) usam `root2d_motion_mode = stabilized_linear`
- `global_root_heading` e adicionado somente quando o deslocamento liquido justifica um heading confiavel (`>= 0.10 m`)
- nenhum retarget de pose e realizado; nenhuma dependencia do `kimodo` conda env no `build-anchor-catalog`

Com `--effector-keyframes N` (N > 0), adicionam-se quatro constraints esparsas de end-effector:

- `left-hand`, `right-hand`, `left-foot`, `right-foot` com N keyframes uniformemente espacados
- esses sao os tipos de constraint para os quais o modelo de difusao do Kimodo foi treinado,
  garantindo interpolacao temporalmente coerente entre os keyframes ancoras
- requer retarget IMUGPT22 → SMPLX22: corrige inversao de direcao dos ossos de quadril (L_Hip, R_Hip)
  e redimensiona comprimentos de osso para os valores canonicos do SMPLX22
- cada constraint inclui `global_joints_positions` (K, 22, 3) — o Kimodo seleciona o joint relevante
  internamente via `joint_names`
- requer o ambiente conda `kimodo`

Em uma frase: o catalogo ancora a trajetoria de chao da captura real enquanto o Kimodo gera
variacao plausivel de pose a partir do prompt textual, com opcao de ancorar tambem as
extremidades (maos e pes) para preservar o carater do movimento original.

## Defaults recomendados

Para manter boa cobertura temporal:

- `describe-windows --window-sec 5.0`
- `describe-windows --window-hop-sec 2.5`
- `describe-windows --num-video-frames 48`

Configuracao fixa desta versao:

- `root2d_min_displacement_m = 0.05`
- `heading_min_displacement_m = 0.10`

`build-anchor-catalog` usa `root2d`-only por padrao. Use `--effector-keyframes N` para adicionar
end-effectors nas extremidades (requer env `kimodo`). Um valor de 5–10 keyframes e suficiente
para janelas de 5 segundos a 20 fps.

## Como rodar

### 1. Exportar pose3d real

Na `.venv` do projeto:

```bash
./.venv/bin/python -m pose_module.robot_emotions export-pose3d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose3d \
  --env-name openmmlab \
  --motionbert-device cuda:0 \
  --no-debug
```

### 2. Gerar o catalogo textual do Qwen por janela

No ambiente `kimodo`:

```bash
python -m robot_emotions_vlm describe-windows \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --output-dir output/robot_emotions_qwen_windows \
  --window-sec 5.0 \
  --window-hop-sec 2.5 \
  --num-video-frames 48
```

### 3. Construir o catalogo ancorado

Modo padrao (`root2d` apenas):

```bash
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors \
  --model Kimodo-SMPLX-RP-v1
```

Com end-effectors nas extremidades (requer env `kimodo`):

```bash
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors_effectors_5 \
  --model Kimodo-SMPLX-RP-v1 \
  --effector-keyframes 5
```

O `constraints.json` resultante salva:

- `root2d.frame_indices`, `root2d.smooth_root_2d`, `root2d.global_root_heading` quando aplicavel
- `left-hand`, `right-hand`, `left-foot`, `right-foot` com `global_joints_positions` (K, 22, 3)
  quando `--effector-keyframes > 0`

### 4. Gerar motions com as ancoras

```bash
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors_effectors_5/kimodo_anchor_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_generated_5
```

Para iterar em uma janela especifica:

```bash
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors_effectors_5/kimodo_anchor_catalog.jsonl \
  --prompt-id robot_emotions_30ms_u03_tag07__w000 \
  --output-dir output/robot_emotions_kimodo_generated_single_5
```

### 5. Converter Kimodo SMPL-X para pose3d do pipeline

Após `generate-kimodo`, as motions estão no formato SMPL-X (motion.npz). Para compará-las com a pose3d real,
converta-as para o sistema de coordenadas normalizado do pipeline:

```bash
python scripts/batch_kimodo_pose3d.py \
  --manifest output/robot_emotions_kimodo_generated_5/kimodo_generation_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_pose3d_5
```

Isso:
- Lê cada `motion.npz` gerado pelo Kimodo
- Aplica `metric_normalizer` (normaliza escala óssea) e `root_estimator` (estima root global)
- Produz `pose3d.npz` em cada subdiretório `<prompt_id>/`, com mesma estrutura que a pose3d real
- Gera `kimodo_pose3d_manifest.jsonl` com referência a todos os outputs

Opcional: use `--skip-existing` (padrão) para pular janelas já processadas.

### 6. Exportar IMU virtual

Sem alinhamento (comportamento anterior):

```bash
python -m robot_emotions_vlm export-kimodo-virtual-imu \
  --kimodo-manifest output/robot_emotions_kimodo_generated/kimodo_generation_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_imu
```

Com alinhamento geométrico + calibração por percentil (recomendado):

```bash
python -m robot_emotions_vlm export-kimodo-virtual-imu \
  --kimodo-manifest output/robot_emotions_kimodo_generated_single_5/kimodo_generation_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_imu \
  --real-imu-root output/exp_real_pose \
  --real-imu-signal-mode acc
```

O flag `--real-imu-root` resolve automaticamente o `imu.npz` de cada janela a partir do
`reference_clip_id` do manifesto Kimodo. Para cada janela aplica, em ordem:

1. Alinhamento geométrico (`run_geometric_alignment`) — corrige o mismatch de frame entre o
   sensor simulado (gravity no eixo Y do SMPL-X) e o sensor físico (montagem fixa no braço).
   Controlado por `pose_module/configs/imu_alignment_config.yaml`.
2. Calibração por percentil (`calibrate_virtual_imu_sequence`) — mapeia a distribuição de
   amplitude do IMU sintético para a do IMU real do clipe de referência.

Flags adicionais:

```
--real-imu-signal-mode acc|gyro|both    Sinal usado na calibração (padrão: acc)
--real-imu-percentile-resolution N      Bins de percentil (padrão: 100)
--no-real-imu-per-class-calibration     Desabilita calibração por classe de emoção
--real-imu-label-key FIELD              Campo do manifesto para calibração por classe (ex: emotion)
```

## Principais saidas

Em `build-anchor-catalog`:

- `kimodo_anchor_catalog.jsonl`
- `kimodo_anchor_catalog.summary.json`
- `<prompt_id>/constraints.json`
- `<prompt_id>/traceability.json`

Em `describe-windows`:

- `window_description_manifest.jsonl`
- `window_description_summary.json`
- `kimodo_window_prompt_catalog.jsonl`
- `<prompt_id>/window.mp4`

Em `generate-kimodo`:

- `kimodo_generation_manifest.jsonl`
- `kimodo_generation_summary.json`
- `<prompt_id>/motion.npz` ou pasta `motion/` quando `num_samples > 1`
- `motion_amass.npz` para `Kimodo-SMPLX-RP-v1`

Em `batch_kimodo_pose3d.py`:

- `kimodo_pose3d_manifest.jsonl`
- `kimodo_pose3d_summary.json`
- `<prompt_id>/pose3d.npz` — pose3d normalizada, comparável com a pose3d real
- `<prompt_id>/error_trace.txt` — se houver falha no processamento
