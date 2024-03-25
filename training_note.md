

```bash
sudo apt-get install libgeos-de
conda activate torch
# comment opencv-python==3.4.1.15 in requirements.txt
pip install -r requirements.txt


python tools/train.py --cfg experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
```
