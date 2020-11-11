# LiteLaneNet PyTorch

## Introduction
PyTorch Implementation of LiteLaneNet(llnet).

## Train
```bash
python lib/train.py exp_configs/llnet.yml
```

## Evaluation
* CULane Evaluation code is ported from [official implementation](<https://github.com/XingangPan/SCNN>) and an extra `CMakeLists.txt` is provided. 
* Modify `root` as absolute project path in `lib/eval/CULane/Run.sh`, then:
    ```bash
    cd lib/eval/CULane
    mkdir build && cd build
    cmake ..
    make
    ```
  Just run `lib/eval_on_culane.py`.
    ```bash
    python lib/eval_on_culane.py exp_configs/llnet.yml
    ```

## Simple Train&Evaluation Various LLNet 
```bash
sh run_exps.sh
```
