# GeoFlow

这是 GeoFlow 的可共享、可复现实验代码版本。目录已经扁平化，原始工作目录
`/home/sunl/work/geoflow` 保持不变；本目录不包含数据集、训练输出、checkpoint、日志、缓存或实验临时脚本。

## 目录结构

```text
GeoFlow-share/
├── config/          # ScanNet / Matterport / GeoFlow 配置
├── dataset/         # 数据读取与预处理代码
├── models/          # GeoFlow、GeoDiff、Affinity 模型
├── run/             # 训练、验证和评测入口
├── tests/           # 轻量逻辑测试
├── third_party/     # Sonata、X-Decoder、Detectron2 源码（不含大模型权重）
├── environment/     # mix 环境导出、pip 包和机器信息
└── docs/            # 安装与实验说明
```

## 环境

本机使用的 Conda 环境名为 `mix`。复现信息见：

- `environment/mix-conda-environment.yml`
- `environment/mix-pip-freeze.txt`
- `environment/system-info.txt`
- `environment/nvidia-gpu.txt`

创建环境：

```bash
conda env create -f environment/mix-conda-environment.yml
conda activate mix
```

完整依赖和 CUDA/MinkowskiEngine/X-Decoder 安装说明见 [docs/Install.md](docs/Install.md)。

## 数据与外部模型

数据集不随仓库上传。请准备 ScanNet 或 Matterport3D，并按配置文件修改：

- `config/geoflow_scannet.yaml`
- `config/geoflow_matterport.yaml`

Sonata、X-Decoder checkpoint 也不随仓库上传；安装后按 `docs/Install.md` 下载，并确认
`config/xdecoder_focall_lang.yaml` 中的 `RESUME_FROM` 指向本地 checkpoint。

## 训练

```bash
conda activate mix
cd /path/to/GeoFlow-share
export PYTHONPATH=.:third_party/sonata:third_party/X-Decoder:third_party/detectron2

# ScanNet
bash run/train_geoflow.sh --exp_dir=out/scannet --config=config/geoflow_scannet.yaml

# Matterport3D
bash run/train_geoflow_matterport.sh --exp_dir=out/matterport --config=config/geoflow_matterport.yaml
```

训练输出应写入独立的 `out/`，该目录被 `.gitignore` 忽略。

## 验证 / 推理

```bash
bash run/val_geoflow.sh \
  --exp_dir=out/scannet \
  --config=config/geoflow_scannet.yaml \
  --ckpt_name=geoflow_last.pth \
  --gpus=0
```

ODE 步数由配置或命令行控制；ODE sweep 属于推理设置，不会改变 checkpoint。

## 轻量检查

不需要数据集即可运行：

```bash
python -m compileall -q dataset models run tests util
python tests/test_core_logic_numpy.py
python tests/test_geodiff_numpy.py
python tests/test_geoflow_numpy.py
```

## 许可证

见 [LICENSE](LICENSE)。第三方代码遵循各自目录中的许可证。
