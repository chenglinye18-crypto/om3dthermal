# om3dthermal

`om3dthermal` 当前是一个面向 HBM-on-GPU 水平堆叠的几何前端。它读取带单位的 YAML，校验并展开层堆叠，最终生成统一的 `AxisAlignedBox` 区域集合，供后续三维热阻网络使用。当前版本**不包含**热阻/FEM/FVM、温度求解、功率与边界条件、网格、正交 MOSAIC、COMSOL 或 GUI。

## 安装与运行

需要 Python 3.10+：

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m om3dthermal.cli build configs/hbm_on_gpu_12hi.yaml --out runs/hbm12
```

构建命令生成：

- `regions.csv`：每个 box 的 SI 坐标、材料、tags、source path 和预留 rotation；
- `geometry_summary.json`：box/material 计数、stack 高度、组件范围及极值尺寸；
- `top_view.png`：footprint 名称与毫米坐标；
- `xz_section.png`、`yz_section.png`：以 µm 显示 z 方向材料层。

## 配置结构

所有输入长度可以写成 Pint 可识别的字符串（如 `65 mm`、`41 um`），加载后统一保存为米。裸数字被视为 SI 米。

- `materials`：材料名、可空的局部导热率 `k_local: [kx, ky, kz]`、metadata。当前 box 的 rotation 恒为单位矩阵，尚不旋转导热张量。
- `footprints`：二维矩形的中心和尺寸。每个 footprint 必须完全位于 `package_footprint` 内。
- `stack_templates`：从下到上的 `items`。普通项为 `kind: layer`；重复项为 `kind: repeat`，包含正整数 `count` 和若干层。展开层自动追加 `_01`、`_02` 等编号。
- `horizontal.foundation`：package foundation 的 footprint 和 stack。
- `horizontal.gpu`：foundation 顶部的 GPU footprint 和 stack。
- `horizontal.memory_zone`：GPU 顶部的参考 stack 高度、低优先级 background slab 与 columns。
- `horizontal.top`：memory zone 顶部的 TIM/Lid stack。

column 支持两种模式：指定 `stack`；或指定单一 `material` 并用 `match_height_of` 引用 stack 高度。比 `reference_stack` 矮的 stack 必须给出 `fill_above` 材料，且构建后严格补齐到参考高度。所有绝对 z 坐标均由 builder 自动推导，配置中不接受 z 坐标。

示例的 `hbm_12hi` 为 45 µm base die、11 次（41 µm DRAM Si + 10 µm bonding）及 149 µm top DRAM Si，总厚度 755 µm，其中 DRAM Si 层恰为 12 个。

## 当前边界

`HorizontalColumnsBuilder` 是专用 builder，不是通用 CAD/布尔引擎。memory zone 的未占用部分当前用带 `priority` 的 background slab 表示；可视化按 priority 让 column 覆盖 background。未来的 `OrthogonalBladesBuilder` 应输出相同的 `AxisAlignedBox` 集合，但本阶段未实现。

## 仍需依据原论文/原始配置确认

本工作区未包含用户提到的 YAML 原文或论文参数表。为使示例可运行，除明确要求的 `hbm_12hi` DRAM 层数量、41/149 µm 厚度与 755 µm 总高度外，下列值均在 YAML 中标为 `NEEDS_PAPER_CONFIRMATION`，不应视为论文数据：

- package、GPU、memory zone、4 个 HBM 及辅助 column 的 footprint 尺寸和位置；
- package substrate、interposer、GPU Si、microbump、base die、bonding、TIM、lid 的厚度；
- 11 个 bonding 层的数量解释及各 10 µm 厚度；
- 用于凑足 755 µm 的 45 µm base die 归属和厚度；
- 所有材料的 `kx/ky/kz`；
- HBM 数量/布局、background 与 fill 材料、priority；
- 截面默认取 `x=0`、`y=0` 是否与论文展示位置一致。

获得原始 YAML 或论文参数表后，应直接替换这些配置值；几何代码无需为参数变化而修改。
