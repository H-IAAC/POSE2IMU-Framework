# Instalação do Kimodo

## 1. Obter o código

Se o repositório ainda não tiver o submódulo:

```bash
git submodule add git@github.com:nv-tlabs/kimodo.git
git submodule update --init --recursive
```

Se o submódulo já existir:

```bash
git submodule update --init --recursive
```

## 2. Criar o ambiente

```bash
conda create -n kimodo python=3.10
conda activate kimodo
pip install --upgrade pip
```

## 3. Instalar PyTorch

Opcao recomendada via conda:

```bash
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

## 4. Corrigir incompatibilidade conhecida do MKL

Em alguns ambientes, o `torch` falha ao importar com o erro abaixo:

```text
ImportError: .../libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

Se isso acontecer, faça downgrade do `mkl` para uma versao compativel:

```bash
conda install -n kimodo -y mkl=2023.1.0
```

## 5. Instalar o Kimodo

```bash
cd kimodo
pip install -e .
```

Se voce quiser usar a demo interativa, instale os extras:

```bash
pip install -e ".[demo]"
```

## 6. Testar se o ambiente ficou correto

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
kimodo_gen --help
kimodo_textencoder --help
```

## 7. Execucao local

Os comandos abaixo rodam os modelos localmente. O que pode acontecer na primeira execucao e apenas o download dos pesos do Hugging Face.

Servidor local do text encoder:

```bash
kimodo_textencoder
```

Geracao local via CLI:

```bash
kimodo_gen "A person sitting, telling a story" \
  --model Kimodo-SMPLX-RP-v1 \
  --duration 10.0 \
  --output output/kimodo/

kimodo_gen "A person is sitting and expressing happiness while recounting something positive, expressive arm gestures, open posture, lively upper-body motion, small weight shifts, as if recalling a personal episode, arm_raise_ratio 0,01 elbow_opening_deg 90 movement_energy 0,99 num_frames 1999 root_speed_mean 0,018 side_symmetry_score 0,89 step_cadence_hz 0,34 trunk_inclination_deg 2,24 vertical_variation_m 0 wrist_amplitude_mean: 0,49" \
  --model Kimodo-SMPLX-RP-v1 \
  --duration 10.0 \
  --output output/kimodo/
```

{"prompt_id": "robot_emotions_happiness_sitting_happy_seated_storytelling_10", "prompt_text": "A person is sitting and expressing happiness while recounting something positive, expressive arm gestures, open posture, lively upper-body motion, small weight shifts, as if recalling a personal episode.", "labels": {"emotion": "happiness", "modality": "sitting", "stimulus": "autobiographical_recall"}, "seed": 123, "num_samples": 1, "fps": 20.0, "duration_hint_sec": 8.0, "action_detail": "happy_seated_storytelling", "stimulus_type": "autobiographical_recall", "reference_clip_id": "robot_emotions_10ms_u02_tag10", "source_metadata": {"dataset": "RobotEmotions", "condition_index": 9, "stimulus_details": "Ask the participant to recall a happy episode/funny anecdote", "reference_clip_ids": ["robot_emotions_10ms_u02_tag10", "robot_emotions_10ms_u03_tag10", "robot_emotions_10ms_u05_tag10", "robot_emotions_30ms_u02_tag03", "robot_emotions_30ms_u03_tag03", "robot_emotions_30ms_u04_tag03", "robot_emotions_30ms_u05_tag03_2", "robot_emotions_30ms_u06_tag03", "robot_emotions_30ms_u08_tag03"], "motion_lexicon": {"arm_raise_ratio": 0.013800700862542887, "duration_sec": 99.94999999999999, "elbow_opening_deg": 89.73967615763347, "movement_energy": 0.09012295140160455, "num_frames": 1999.0, "root_speed_mean": 0.017807453146411314, "side_symmetry_score": 0.8874801728460524, "step_cadence_hz": 0.3376583696533339, "trunk_inclination_deg": 2.2415460679266186, "vertical_variation_m": 0.0, "wrist_amplitude_mean": 0.48591100838449264}}}


Para forcar uso apenas de cache/arquivos locais:

```bash
export LOCAL_CACHE=true
export TEXT_ENCODER_MODE=local
```
