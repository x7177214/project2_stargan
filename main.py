import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def main(config):

    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # Data loader
    data_loader = None

    if config.dataset in ['face'] and config.mode == 'train':
        train_data_loader = get_loader(config.face_image_path, config.face_crop_size,
                                   config.image_size, config.batch_size, 'face', config, config.mode)
        test_data_loader = get_loader("./", config.face_crop_size, config.image_size, 12, 'face', config, 'val')
    elif config.dataset in ['face'] and config.mode == 'test':
        test_data_loader = get_loader(config.face_image_path, config.face_crop_size,
                                      config.image_size, config.batch_size, 'face', config, config.mode)

    # Solver
    if config.mode == 'train':
        if config.dataset in ['face']:
            solver = Solver(train_data_loader, test_data_loader, config)
            solver.train()

    elif config.mode == 'test':
        if config.dataset in ['face']:
            solver = Solver(None, test_data_loader, config)
            
            if config.ranbow_output == 1:
                solver.test(config.test_model)
            else:
                if config.extract_feature == 1:
                    # solver.extract_feature()
                    pass
                else:
                    solver.test_save_single_img()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='visualize for training')

    #new loss choice
    parser.add_argument('--loss_id', action='store_true', help='id_loss for the generator')
    parser.add_argument('--loss_symmetry', action='store_true', help='loss_symmetry for the generator')
    parser.add_argument('--loss_identity', action='store_true', help='loss_identity for the generator')
    parser.add_argument('--loss_tv', action='store_true', help='tv for the generator')
    parser.add_argument('--loss_id_cls', action='store_true', help='loss_id_cls')
    parser.add_argument('--id_cls_loss', type=str, default='cross', choices=['angle', 'cross'], help='id cls loss func.')
    parser.add_argument('--use_sn', action='store_true', help='use sn')
    parser.add_argument('--use_si', action='store_true', help='use_siamese')
    parser.add_argument('--use_gpb', action='store_true', help='use global pooling branch at the generator')

    parser.add_argument('--lambda_idx', type=float, default=5.0)
    parser.add_argument('--lambda_symmetry', type=float, default=0.3)
    parser.add_argument('--lambda_tv', type=float, default=1.5e-6)
    parser.add_argument('--lambda_identity', type=float, default=3e-3)
    parser.add_argument('--lambda_id_cls', type=float, default=1.0)
    parser.add_argument('--lambda_si', type=float, default=1.5)

    # Model hyper-parameters
    parser.add_argument('--c_dim', type=int, default=20)
    parser.add_argument('--num_id', type=int, default=294, help='number of uni. id in the training set')
    parser.add_argument('--image_size', type=int, default=180)
    parser.add_argument('--face_crop_size', type=int, default=176)
    # parser.add_argument('--image_size', type=int, default=300)
    # parser.add_argument('--face_crop_size', type=int, default=289)

    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=9)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=6)
    parser.add_argument('--log_space', action='store_true', help='on log space')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--display_f', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='face', choices=['face'])
    parser.add_argument('--num_epochs', type=int, default=120)
    parser.add_argument('--num_epochs_decay', type=int, default=60)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default=None) # ex: '2_1000'

    # Test settings
    parser.add_argument('--test_model', type=str, default='120_800')
    parser.add_argument('--ranbow_output', type=int, default=0) # save ranbow illu. conditions in test mode?
    parser.add_argument('--extract_feature', type=int, default=0)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # face
    # parser.add_argument('--face_image_path', type=str, default='/Disk2/Multi-Pie/cropface_expression01_lm')
    parser.add_argument('--face_image_path', type=str, default='/Disk2/dataset/Multi-Pie/cropface_expression01_lm')
    # parser.add_argument('--face_image_path', type=str, default='/home/hank/Documents/Multi-Pie/cropface_expression01_lm')

    ## for liteon img testing, use this:
    # parser.add_argument('--face_image_path', type=str, default='/home/hank/Documents/mtcnn-pytorch-master/crop_liteon')

    ## for iir3d testing, use this:
    # parser.add_argument('--face_image_path', type=str, default='/file/dataset/face/IIR3D/IIR3D_new/RGB')

    parser.add_argument('--root_path', type=str, default='./face_gpb4_si_1.5_cross')

    parser.add_argument('--face_metadata_path', type=str, default='./data/pie_label.txt')
    parser.add_argument('--log_path', type=str, default='.//logs')
    parser.add_argument('--model_save_path', type=str, default='.//models')
    parser.add_argument('--sample_path', type=str, default='.//samples')
    parser.add_argument('--result_path', type=str, default='.//results')

    # Step size
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=800)

    config = parser.parse_args()

    config.log_path = config.root_path+'/logs'
    config.model_save_path = config.root_path+'/models'
    config.sample_path = config.root_path+'/samples'
    config.result_path = config.root_path+'/results'

    print(config)

    if not os.path.exists(config.root_path):
        os.mkdir(config.root_path)
    od = vars(config)
    for a in od:
        with open(config.root_path+'/meta.txt', 'a') as f:
            if isinstance(od[a], bool):
                f.writelines(a + ': %r \n  \n' % od[a])
            elif isinstance(od[a], int):
                f.writelines(a + ': %d \n  \n' % od[a])
            elif isinstance(od[a], float):
                f.writelines(a + ': %f \n  \n' % od[a])
            elif isinstance(od[a], str):
                f.writelines(a + ': %s \n  \n' % od[a])
    main(config)

#python3 main.py --mode='train'
