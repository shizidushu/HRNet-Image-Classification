import torch
from config import config
from config import update_config
import torch.backends.cudnn as cudnn
import models
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--exportONNX',
                        help="exported onnx",
                        type=str,
                        default='output.onnx')

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)
    
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()
    
    weight_path = "./hrnet_w18_small_model_v1.pth"
    model.module.load_state_dict(torch.load(weight_path))
    
    if args.exportONNX:
        dump_input = torch.rand((1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]))
        torch.onnx.export(model.module.cpu(), dump_input, args.exportONNX, opset_version=12, do_constant_folding=True)

if __name__ == '__main__':
    main()


## Usage:
# python my_custom_code/lab.py --cfg experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
