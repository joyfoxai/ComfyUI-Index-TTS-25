# ComfyUI IndexTTS 2 / 2.5 节点

这是一个原生 ComfyUI `AUDIO` 节点，支持 IndexTTS 2 和 IndexTTS 2.5。Loader
会读取模型目录中 `config.yaml` 的 `version` 字段，自动选择 `infer_v2` 或
`infer_v2_5`，不需要手动切换代码。

## 功能

- IndexTTS 2 与 IndexTTS 2.5 共用一套节点。
- 标准 ComfyUI `AUDIO` 输入和输出。
- 说话人音色克隆。
- 五种情感控制模式。
- 模型预加载、缓存和显存释放。
- 支持采样、语速、分段、随机种子及可选加速参数。
- 已兼容 Transformers 4.52.1～4.57.x。
- 已处理 TorchAudio 2.9 整数 PCM 保存导致的严重削波问题。

## 节点目录

在 ComfyUI 的 `custom_nodes` 目录中克隆本仓库：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/joyfoxai/ComfyUI-Index-TTS-25.git
```

节点已经内置所需的官方 IndexTTS Python 源码：

```text
ComfyUI-Index-TTS-25/index-tts/indextts/
```

内置源码对应官方提交 `a371df7`。通常不需要另外克隆 IndexTTS 仓库。

## 安装依赖

使用启动 ComfyUI 的同一个 Python 安装：

```bash
cd ComfyUI
python -m pip install -r custom_nodes/ComfyUI-Index-TTS-25/requirements.txt
```

如果 ComfyUI 使用虚拟环境或便携版 Python，请将上面的 `python` 换成该环境对应的
Python 可执行文件。不要为了本节点单独覆盖 ComfyUI 已安装的 Torch、TorchAudio 或 CUDA。

### 额外依赖说明

以下依赖并非所有 ComfyUI 环境都会预装：

- `fugashi`、`unidic-lite`：日语分词和日语词典。缺少时模型会在初始化日语处理器时报错，
  即使当前只生成中文，也建议安装。
- `WeTextProcessing`：中文和英文文本规范化。
- `g2p-en`：英文文本转音素。
- `jieba`、`cn2an`：中文切词和数字规范化。
- `librosa`、`descript-audiotools`：参考音频处理。
- `omegaconf`、`sentencepiece`、`safetensors`：配置、分词和模型读取。

只补装日语依赖可执行：

```bash
python -m pip install fugashi unidic-lite
```

Loader 中的以下选项属于可选加速功能，默认关闭即可：

- `deepspeed`：需要额外安装与当前 Torch/CUDA 匹配的 `deepspeed`。
- `gpt_accel`：需要与当前 Torch/CUDA 匹配的 `flash-attn`。
- `torch_compile`：Linux 通常使用 Torch 自带 Triton；首次编译耗时较长。
- `cuda_kernel`：会尝试使用 BigVGAN CUDA 自定义内核，失败时应关闭。

## 下载模型

节点目录中提供了下载脚本：

```bash
cd ComfyUI/custom_nodes/ComfyUI-Index-TTS-25
```

下载两个版本：

```bash
./download_models.sh all
```

只下载 IndexTTS 2：

```bash
./download_models.sh 2
```

只下载 IndexTTS 2.5：

```bash
./download_models.sh 2.5
```

脚本使用 Hugging Face：

- `IndexTeam/IndexTTS-2`
- `IndexTeam/IndexTTS-2.5`

如果仓库需要鉴权或遇到限流，可先设置 Token：

```bash
export HF_TOKEN="你的 Hugging Face Token"
./download_models.sh all
```

也可以使用 Hugging Face 镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
./download_models.sh all
```

指定其他公共模型根目录：

```bash
INDEXTTS_MODELS_DIR=/data/comfyui_models ./download_models.sh all
```

脚本会自动追加 `indextts` 子目录。也可以指定 ComfyUI 的 Python：

```bash
PYTHON_BIN=/path/to/comfyui/python ./download_models.sh all
```

## 模型应该放在哪里

默认情况下，模型放在 ComfyUI 的标准模型目录：

```text
ComfyUI/models/indextts/
├── IndexTTS-2/
│   ├── config.yaml
│   ├── bpe.model
│   ├── gpt.pth
│   ├── s2mel.pth
│   ├── wav2vec2bert_stats.pt
│   └── ...
└── IndexTTS-2.5/
    ├── config.yaml
    ├── codec.pth
    ├── gpt.pth
    ├── s2mel.pth
    ├── wav2vec2bert_stats.pt
    ├── qwen0.6bemo4-merge/
    ├── hf_cache/
    └── ...
```

节点会依次检查：

1. 环境变量 `INDEXTTS_MODELS_DIR` 指定的位置。
2. `ComfyUI/models/indextts`。

修改模型目录或新下载模型后，需要重启 ComfyUI 才会刷新 Loader 下拉列表。

### 使用自定义模型目录

不需要创建软链接。通过环境变量指定模型根目录后再启动 ComfyUI：

```bash
export INDEXTTS_MODELS_DIR=/path/to/models
python main.py
```

如果路径本身不是以 `indextts` 命名，节点会自动在其下查找 `indextts` 子目录。

## 节点和连接方式

右键菜单位置：

```text
audio → IndexTTS
```

提供三个节点：

1. `IndexTTS 2 / 2.5 Model Loader`
2. `IndexTTS 2 / 2.5 Synthesize`
3. `IndexTTS 2 / 2.5 Unload Model`

基本工作流：

```text
IndexTTS Model Loader.model
             │
             ▼
IndexTTS Synthesize.model

Load Audio.AUDIO ──────────► IndexTTS Synthesize.speaker_audio

IndexTTS Synthesize.audio ─► Preview Audio / Save Audio
```

操作步骤：

1. 在 Loader 的 `model_name` 中选择 `IndexTTS-2` 或 `IndexTTS-2.5`。
2. 将 Loader 的 `model` 输出连接到 Synthesize 的 `model`。
3. 使用 ComfyUI `Load Audio` 加载说话人参考音频，连接到 `speaker_audio`。
4. 输入文本并选择语言、情感模式和生成参数。
5. 将 Synthesize 的 `audio` 连接到 `Preview Audio` 或 `Save Audio`。
6. 不再使用时运行 Unload 节点释放缓存模型和显存。

一个 Loader 可以连接多个 Synthesize 节点，模型只会缓存一份。切换版本、模型、精度、
设备或加速设置时会自动卸载旧缓存并加载新模型。

## 五种情感模式

- `same_as_speaker`：沿用说话人参考音频中的情感。
- `reference_audio`：连接可选输入 `emotion_audio`，使用另一段音频作为情感参考。
- `emotion_vector`：使用八个滑块控制快乐、愤怒、悲伤、恐惧、厌恶、忧郁、惊讶和平静。
- `emotion_text`：直接根据待合成正文识别情感。
- `extra_emotion_text`：根据单独填写的 `extra_emotion_text` 识别情感；该字段映射到
  IndexTTS 原生 `emo_text` 参数。

`reference_audio` 模式必须连接 `emotion_audio`；`extra_emotion_text` 模式必须填写非空文本。

## 版本差异

- IndexTTS 2：本节点支持 `ZH`、`EN`；选择 Loader 的 `bf16` 时实际使用 FP16。
- IndexTTS 2.5：支持 `ZH`、`EN`、`JA`、`AR`、`ES`；`bf16` 使用 BF16。
- `fp32`：两个版本都使用全精度，显存占用更高。
- `duration_factor` 与 `text_normalization` 是 2.5 参数；使用 2.0 时节点会安全忽略。

## 常见问题

### Loader 能看到模型但提示模型不完整

确认模型文件不是 Git LFS 指针或下载中断文件，并检查上面的必要文件是否存在。重新运行
下载脚本会复用已有缓存并继续下载。

### Hugging Face 返回 429

稍后重试、配置 `HF_TOKEN`，或者设置 `HF_ENDPOINT` 后重新执行下载脚本。

### 报缺少 `fugashi` 或 `MeCab`

安装：

```bash
python -m pip install fugashi unidic-lite
```

必须安装到启动 ComfyUI 的 Python 环境，而不是系统 Python。

### 输出刺耳或接近白噪声

节点会将 IndexTTS 返回的原生 PCM 显式归一化为 ComfyUI 浮点 `AUDIO`，以避免部分
TorchAudio 版本直接处理 `int16` 张量时产生削波。如果自行修改输出处理，请保留这一步
归一化。

### 修改后节点没有出现

完整重启 ComfyUI，然后刷新浏览器。仅刷新网页不会重新加载 Python 节点。

## 环境变量

- `INDEXTTS_MODELS_DIR`：模型根目录或已经命名为 `indextts` 的目录。
- `INDEXTTS_REPO`：覆盖节点内置的 IndexTTS 源码位置，仅开发调试时使用。
- `HF_TOKEN`：Hugging Face 下载令牌。
- `HF_ENDPOINT`：Hugging Face API/镜像地址。
- `PYTHON_BIN`：下载脚本使用的 Python 可执行文件。
