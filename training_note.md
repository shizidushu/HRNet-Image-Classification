
## Prepare environment

```bash
sudo apt-get install libgeos-de
conda activate torch
# comment opencv-python==3.4.1.15 in requirements.txt
pip install -r requirements.txt
```

## Training with only one image

```bash
python tools/train_custom.py --cfg experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100_OneImage.yaml --dataDir ./data/One-Image/1

python tools/train_custom.py --cfg experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100_OneImage.yaml --dataDir ./data/One-Image/2
```


```bash
python tools/valid_custom.py --cfg experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100_OneImage.yaml --testModel output/One-Image/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100_OneImage/1/model_best.pth.tar --dataDir ./data/One-Image/1

python tools/valid_custom.py --cfg experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100_OneImage.yaml --testModel output/One-Image/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100_OneImage/2/model_best.pth.tar --dataDir ./data/One-Image/2
```

Result: It overfits
