import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
# from model import Generator
# from model import Discriminator
# from model import Discriminator_idcls_angle_SN
from PIL import Image
from visualizer import Visualizer
# from light_cnn import LightCNN_9Layers, LightCNN_29Layers

class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input_, target):
        self.it += 1

        cos_theta, phi_theta = input_
        target = target.view(-1, 1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin, self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

class LaplacianIMG(nn.Module):
    def __init__(self, mode=1):
        super(LaplacianIMG, self).__init__()
        f = np.zeros([3, 3, 3, 3]).astype(np.float32)

        for i in range(3):
            if mode == 0:
                f[i, i, :, :] = -1.0
                f[i, i, 1, 1] = 8.0
            else:
                f[i, i, :, :] = -1.0
                f[i, i, 0, 0] = 0.0
                f[i, i, 0, 2] = 0.0
                f[i, i, 2, 0] = 0.0
                f[i, i, 2, 2] = 0.0
                f[i, i, 1, 1] = 4.0
        self.f = nn.Parameter(data=torch.FloatTensor(f), requires_grad=False)
    def forward(self, x):
        return torch.abs(F.conv2d(x, self.f, padding=1))

class Solver(object):

    def __init__(self, train_loader, test_loader, config):
        # Data loader
        print('cuda.device_count: ', torch.cuda.device_count())
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        if self.config.loss_symmetry:
            self.take_laplacian = LaplacianIMG()
            if torch.cuda.is_available():
                self.take_laplacian.cuda()

        # Build tensorboard if use
        self.build_model()

        if self.config.id_cls_loss == 'angle':
            self.id_cls_criterion = AngleLoss()
        elif self.config.id_cls_loss == 'cross':
            self.id_cls_criterion = nn.CrossEntropyLoss()

        self.down_function = nn.AvgPool2d((2, 2), stride=(2, 2))

        if self.config.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.config.pretrained_model:
            self.load_pretrained_model()

    def get_feature(self,img):
        img = img.clamp(-1,1)
        img = (img + 1.0) / 2.0 * 255.0
        #torch.FloatTensor(LRTrans(Image.fromarray(img.cpu().numpy().transpose(0, 2).transpose(0, 1).numpy())).numpy())
        # temp = img[:, 0] * 299 / 1000 + img[:,1] * 587 / 1000 + img[:,2] * 114 / 1000

        # temp = self.down_function(temp)
        #temp=(img[:,0]+img[:,1]+img[:,2])/3
        f1, f2 = self.l_model(img)

        return f1, f2  # .data.cpu().numpy()[0]

    def total_variation_loss(self, img):
        '''
        def lp_matrix():
            import numpy as np
            c = np.zeros([3, 3, 3, 3])
            l_k = np.zeros([3, 3])

            l_k[0, 1]=-1
            l_k[1, 0]=-1
            l_k[1, 2]=-1
            l_k[2, 1]=-1
            l_k[1 ,1]= 4
            return Variable(torch.FloatTensor(c).cuda(), volatile=True,requires_grad=False)
        '''
        #out=torch.mean(torch.abs(F.conv2d(img,lp_matrix)))
        out = (
            torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
            torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        )
        return out

    def build_model(self):
        if self.config.loss_identity:

            from light_cnn import LightCNN_29Layers
            self.l_model = LightCNN_29Layers(num_classes=294)

            self.l_model.eval()
            self.l_model = torch.nn.DataParallel(self.l_model).cuda()
            checkpoint = torch.load("data/lightCNN_160_checkpoint.pth.tar")
            self.l_model.load_state_dict(checkpoint['state_dict'])

        if self.config.mode == 'test':
            feature = True
        else:
            feature = False

        # Define a generator and a discriminator
        if self.config.use_gpb:
            from model import Generator_gpb
            self.G = Generator_gpb(self.config.g_conv_dim, self.config.c_dim, self.config.g_repeat_num)
            self.G = torch.nn.DataParallel(self.G, device_ids=[i for i in range(torch.cuda.device_count())]) # use DataParallel
        else:
            from model import Generator
            self.G = Generator(self.config.g_conv_dim, self.config.c_dim, self.config.g_repeat_num)
        # self.D = Discriminator(self.config.image_size, self.config.d_conv_dim, self.config.c_dim, self.config.d_repeat_num)
        
        if self.config.loss_id_cls:
            if self.config.id_cls_loss == 'angle':
                if self.config.use_sn:
                    from model import Discriminator_idcls_angle_SN
                    self.D = Discriminator_idcls_angle_SN(self.config.face_crop_size, self.config.d_conv_dim, self.config.c_dim, self.config.d_repeat_num, feature=feature, classnum=self.config.num_id)
                else:
                    from model import Discriminator_idcls_angle
                    self.D = Discriminator_idcls_angle(self.config.face_crop_size, self.config.d_conv_dim, self.config.c_dim, self.config.d_repeat_num, feature=feature, classnum=self.config.num_id)
            elif self.config.id_cls_loss == 'cross':
                if self.config.use_sn:
                    from model import Discriminator_idcls_cross_SN
                    self.D = Discriminator_idcls_cross_SN(self.config.face_crop_size, self.config.d_conv_dim, self.config.c_dim, self.config.d_repeat_num, feature=feature, classnum=self.config.num_id)
                else:
                    from model import Discriminator_idcls_cross
                    self.D = Discriminator_idcls_cross(self.config.face_crop_size, self.config.d_conv_dim, self.config.c_dim, self.config.d_repeat_num, feature=feature, classnum=self.config.num_id)
        else:
            if self.config.use_sn:
                from model import Discriminator_SN
                self.D = Discriminator_SN(self.config.face_crop_size, self.config.d_conv_dim, self.config.c_dim, self.config.d_repeat_num)
            else:
                from model import Discriminator
                self.D = Discriminator(self.config.face_crop_size, self.config.d_conv_dim, self.config.c_dim, self.config.d_repeat_num)
                    
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def flip(self, x, dim):
        x = x.data
        dim = x.dim() + dim if dim < 0 else dim
        inds = tuple(slice(None, None) if i != dim
                    else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
                    for i in range(x.dim()))
        return Variable(x[inds])

    def compute_sym_loss(self, img):
        b = self.flip(img, 3)
        loss = torch.mean(torch.abs((b-img))) # average on W/2 * H pixels
        return loss

    def find_sym_img_and_cal_loss(self, x, target_c=None, find_partial=True):

        if not find_partial: # cal. loss over all samples in a batch
            g_loss_sym = self.compute_sym_loss(x)
        else:
            _, idx = target_c.max(1)
            idx = idx.data.cpu().numpy()
            search_set = np.array([0, 7, 19])

            ix = np.isin(idx, search_set)
            ix = np.where(ix)[0]

            if len(ix) != 0:
                sym_idx = Variable(torch.LongTensor(ix).cuda())
                sym_x = x.index_select(0, sym_idx)
                g_loss_sym = self.compute_sym_loss(sym_x)
            else:
                g_loss_sym = Variable(torch.FloatTensor([0.0]).cuda())

        return g_loss_sym

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.config.model_save_path, '{}_G.pth'.format(self.config.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.config.model_save_path, '{}_D.pth'.format(self.config.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.config.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.config.log_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def log_restore(self,input):
        return (torch.exp(input * np.log(256)) - 1) / 255

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def denorm2(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def threshold(self, x):
        x = x.clone()
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x

    def compute_accuracy(self, x, y, dataset):

        x = F.sigmoid(x)
        predicted = self.threshold(x)
        correct = (predicted == y).float()
        accuracy = torch.mean(correct, dim=0) * 100.0

        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def make_celeb_labels(self,len):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        fixed_c_list = []

        for i in range(self.config.c_dim):
                fixed_c=np.zeros([len,self.config.c_dim])
                for j in range(len):
                    for k in range(self.config.c_dim):
                        fixed_c[j,k] = 0
                    fixed_c[j,i] = 1
                fixed_c_list.append(self.to_var(torch.FloatTensor(fixed_c), volatile=True))

        return fixed_c_list

    def train(self):

        # if self.config.visualize:
        visualizer = Visualizer()

        """Train StarGAN within a single dataset."""

        # Set dataloader
        self.data_loader = self.train_loader

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        for i, (imgs, labels, _) in enumerate(self.data_loader):
            fixed_x.append(imgs[0])
            real_c.append(labels)
            if i == 0:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        real_c = torch.cat(real_c, dim=0)

        fixed_c_list = self.make_celeb_labels(self.config.batch_size)

        # lr cache for decaying
        g_lr = self.config.g_lr
        d_lr = self.config.d_lr

        # Start with trained model if exists
        if self.config.pretrained_model:
            start = int(self.config.pretrained_model.split('_')[0])-1
        else:
            start = 0

        # Start training
        self.loss = {}
        start_time = time.time()

        for e in range(start, self.config.num_epochs):
            self.test(e)
            for i, (images, real_label, identity) in enumerate(self.data_loader):

                real_x = images[0]

                if self.config.use_si:
                    real_ox = self.to_var(images[1])
                    real_oo = self.to_var(images[2])
      
                if self.config.id_cls_loss == 'cross':
                    identity = identity.squeeze()

                # Generate fake labels randomly (target domain labels)
                rand_idx = torch.randperm(real_label.size(0))

                fake_label = real_label[rand_idx]

                real_c = real_label.clone()
                fake_c = fake_label.clone()

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_c = self.to_var(real_c)           # input for the generator
                fake_c = self.to_var(fake_c)
                real_label = self.to_var(real_label)   # this is same as real_c if dataset == 'CelebA'
                fake_label = self.to_var(fake_label)
                identity = self.to_var(identity)

                # ================== Train D ================== #

                # Compute loss with real images
                if self.config.loss_id_cls:
                    out_src, out_cls, out_id_real = self.D(real_x)
                else:
                    out_src, out_cls = self.D(real_x)

                d_loss_real = - torch.mean(out_src)

                d_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls, real_label, size_average=False) / real_x.size(0)

                if self.config.loss_id_cls:
                    d_loss_id_cls = self.id_cls_criterion(out_id_real, identity)
                    self.loss['D/loss_id_cls'] = self.config.lambda_id_cls * d_loss_id_cls.data[0]
                else:
                    d_loss_id_cls = 0.0

                # Compute classification accuracy of the discriminator
                if (i+1) % self.config.log_step == 0:
                    accuracies = self.compute_accuracy(out_cls.detach(), real_label, self.config.dataset)
                    log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    print('Classification Acc (20 classes): ')
                    print(log)
                    print('\n')

                # Compute loss with fake images
                if self.config.use_gpb:
                    fake_x, _ = self.G(real_x, fake_c)
                else:
                    fake_x = self.G(real_x, fake_c)
                fake_x = Variable(fake_x.data)

                if self.config.loss_id_cls:
                    out_src, out_cls, _ = self.D(fake_x.detach())
                else:
                    out_src, out_cls = self.D(fake_x.detach())

                d_loss_fake = torch.mean(out_src)

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake + self.config.lambda_cls * d_loss_cls + d_loss_id_cls * self.config.lambda_id_cls
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)

                if self.config.loss_id_cls:
                    out, out_cls, _ = self.D(interpolated)
                else:
                    out, out_cls = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                # Backward + Optimize
                d_loss = self.config.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging
                self.loss['D/loss_real'] = d_loss_real.data[0]
                self.loss['D/loss_fake'] = d_loss_fake.data[0]
                self.loss['D/loss_cls'] = self.config.lambda_cls * d_loss_cls.data[0]
                self.loss['D/loss_gp'] = self.config.lambda_gp * d_loss_gp.data[0]
                
                # ================== Train G ================== #
                if (i+1) % self.config.d_train_repeat == 0:

                    self.img = {}
                    # Original-to-target and target-to-original domain
                    if self.config.use_gpb:
                        fake_x, id_vector_real_in_x = self.G(real_x, fake_c)
                        rec_x, id_vector_fake_in_x = self.G(fake_x.detach(), real_c)
                    else:
                        fake_x = self.G(real_x, fake_c)
                        rec_x = self.G(fake_x.detach(), real_c)

                    # Compute losses
                    if self.config.loss_id_cls:
                        out_src, out_cls, out_id_fake = self.D(fake_x)
                    else:
                        out_src, out_cls = self.D(fake_x)

                    g_loss_fake = - torch.mean(out_src)
                    g_loss_rec = torch.mean(torch.abs(real_x - rec_x))
    
                    ### siamese loss
                    if self.config.use_si:
                        if self.config.use_gpb:
                            # feedforward
                            fake_ox, id_vector_ox = self.G(real_ox, fake_c)
                            fake_oo, id_vector_oo  = self.G(real_oo, fake_c)

                            id_vector_ox = id_vector_ox.detach()
                            id_vector_oo = id_vector_oo.detach()

                            mdist = 1.0 - torch.mean(torch.abs(id_vector_real_in_x - id_vector_oo))
                            mdist = torch.clamp(mdist, min=0.0)
                            g_loss_si = 0.5*(torch.pow(torch.mean(torch.abs(id_vector_real_in_x - id_vector_ox)), 2) + torch.pow(mdist, 2))

                            # backward
                            _, id_vector_ox = self.G(fake_ox.detach(), real_c)
                            _, id_vector_oo  = self.G(fake_oo.detach(), real_c)

                            id_vector_ox = id_vector_ox.detach()
                            id_vector_oo = id_vector_oo.detach()

                            mdist = 1.0 - torch.mean(torch.abs(id_vector_fake_in_x - id_vector_oo))
                            mdist = torch.clamp(mdist, min=0.0)
                            g_loss_si += 0.5*(torch.pow(torch.mean(torch.abs(id_vector_fake_in_x - id_vector_ox)), 2) + torch.pow(mdist, 2))

                            self.loss['G/g_loss_si'] = g_loss_si.data[0]
                        else:
                            fake_ox = self.G(real_ox, fake_c).detach()
                            
                            fake_ooc = fake_c.data.cpu().numpy().copy()
                            fake_ooc = np.roll(fake_ooc, np.random.randint(self.config.c_dim), axis=1)
                            fake_ooc = self.to_var(torch.FloatTensor(fake_ooc))

                            fake_oo = self.G(real_oo, fake_ooc).detach()
                            mdist = 1.0 - torch.mean(torch.abs(fake_x - fake_oo))
                            mdist = torch.clamp(mdist, min=0.0)
                    
                            g_loss_si = 0.5*(torch.pow(torch.mean(torch.abs(fake_x - fake_ox)), 2) + torch.pow(mdist, 2))
                            self.loss['G/g_loss_si'] = g_loss_si.data[0]
                    else:
                        g_loss_si = 0.0

                    ### id cls loss
                    if self.config.loss_id_cls:
                        g_loss_id_cls = self.id_cls_criterion(out_id_fake, identity)
                        self.loss['G/g_loss_id_cls'] = self.config.lambda_id_cls * g_loss_id_cls.data[0]
                    else:
                        g_loss_id_cls = 0.0

                    ### sym loss
                    if self.config.loss_symmetry:
                        g_loss_sym_fake = self.find_sym_img_and_cal_loss(fake_x, fake_c,
                                                                         True)  # cal. over samples w/ specific labels
                        g_loss_sym_rec = self.find_sym_img_and_cal_loss(rec_x, real_c, True)

                        lap_fake_x = self.take_laplacian(fake_x)
                        lap_rec_x = self.take_laplacian(rec_x)
                        g_loss_sym_lap_fake = self.find_sym_img_and_cal_loss(lap_fake_x, None,
                                                                             False)  # cal. over all samples
                        g_loss_sym_lap_rec = self.find_sym_img_and_cal_loss(lap_rec_x, None, False)
                        sym_loss=(g_loss_sym_fake+g_loss_sym_rec+g_loss_sym_lap_fake+g_loss_sym_lap_rec)
                        self.loss['G/g_loss_sym']=self.config.lambda_symmetry*sym_loss.data[0]
                    else:
                        sym_loss=0

                    ###id loss
                    if self.config.loss_id:
                        if self.config.use_gpb:
                            idx, _ = self.G(real_x, real_c)
                        else:
                            idx = self.G(real_x, real_c)
                        self.img['idx'] = idx

                        g_loss_id = torch.mean(torch.abs(real_x - idx))
                        self.loss['G/g_loss_id'] = self.config.lambda_idx * g_loss_id.data[0]
                    else:
                        g_loss_id= 0

                    ###identity loss
                    if self.config.loss_identity:
                        real_x_f, real_x_p = self.get_feature(real_x)
                        fake_x_f, fake_x_p = self.get_feature(fake_x)
                        g_loss_identity = torch.mean(torch.abs(real_x_f - fake_x_f))
                        g_loss_identity += torch.mean(torch.abs(real_x_p - fake_x_p))

                        self.loss['G/g_loss_identity'] = self.config.lambda_identity*g_loss_identity.data[0]
                    else:
                        g_loss_identity=0

                    ###total var loss
                    if self.config.loss_tv:
                        g_tv_loss = (self.total_variation_loss(fake_x) + self.total_variation_loss(rec_x)) / 2
                        self.loss['G/tv_loss'] = self.config.lambda_tv * g_tv_loss.data[0]
                    else:
                        g_tv_loss=0

                    ### D's cls loss
                    g_loss_cls = F.binary_cross_entropy_with_logits(
                            out_cls, fake_label, size_average=False) / fake_x.size(0)

                    # Backward + Optimize
                    g_loss = g_loss_fake +\
                             self.config.lambda_rec * g_loss_rec +\
                             self.config.lambda_cls * g_loss_cls+\
                             self.config.lambda_idx * g_loss_id+\
                             self.config.lambda_identity*g_loss_identity+\
                             self.config.lambda_tv*g_tv_loss+\
                             self.config.lambda_symmetry*sym_loss+\
                             self.config.lambda_id_cls * g_loss_id_cls+\
                             self.config.lambda_si * g_loss_si

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    self.img['real_x'] = real_x
                    self.img['fake_x'] = fake_x
                    self.img['rec_x'] = rec_x
                    self.loss['G/loss_fake'] = g_loss_fake.data[0]
                    self.loss['G/loss_rec'] = self.config.lambda_rec*g_loss_rec.data[0]
                    self.loss['G/loss_cls'] = self.config.lambda_cls*g_loss_cls.data[0]
                    #

                # Print out log info
                if (i+1) % self.config.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.config.num_epochs, i+1, iters_per_epoch)

                    for tag, value in self.loss.items():
                        log += ", {}: {}".format(tag, value)
                    print(log)

                    if self.config.use_tensorboard:
                        for tag, value in self.loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
                if (i) % self.config.sample_step == 0:
                    fake_image_list = [fixed_x]
                    for fixed_c in fixed_c_list:
                        if self.config.use_gpb:
                            fake_image_list.append(self.G(fixed_x, fixed_c)[0])
                        else:
                            fake_image_list.append(self.G(fixed_x, fixed_c))
                    fake_images = torch.cat(fake_image_list, dim=3)

                    if not self.config.log_space:
                        save_image(self.denorm(fake_images.data),
                            os.path.join(self.config.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    else:
                        fake_images = self.denorm(fake_images.data)*255.0
                        fake_images = torch.pow(2.71828182846, fake_images/255.0*np.log(256.0))-1.0
                        fake_images = fake_images/255.0
                        fake_images = fake_images.clamp(0.0, 1.0)
                        save_image(fake_images,
                            os.path.join(self.config.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)

                    print('Translated images and saved into {}..!'.format(self.config.sample_path))

                # Save model checkpoints
                if (i+1) % self.config.model_save_step == 0:
                    torch.save(self.G.state_dict(),
                        os.path.join(self.config.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.config.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))
                if self.config.visualize and (i + 1) % self.config.display_f == 0:
                    visualizer.display_current_results(self.img)
                    visualizer.plot_current_errors(e, float(i+1) / iters_per_epoch, self.loss)

            # Decay learning rate
            if (e+1) > (self.config.num_epochs - self.config.num_epochs_decay):
                g_lr -= (self.config.g_lr / float(self.config.num_epochs_decay))
                d_lr -= (self.config.d_lr / float(self.config.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self, e):
        # Load trained parameters
        if self.config.mode == 'test':
            G_path = os.path.join(self.config.model_save_path, '{}_G.pth'.format(self.config.test_model))
            self.G.load_state_dict(torch.load(G_path))

        self.G.eval()

        test_data_loader = self.test_loader

        for i, (imgs, _) in enumerate(test_data_loader):

            real_x = self.to_var(imgs[0], volatile=True)
            target_c_list = self.make_celeb_labels(len(real_x))

            # Start translations
            fake_image_list = [real_x]
            for target_c in target_c_list:
                if self.config.use_gpb:
                    fake_image_list.append(self.G(real_x, target_c)[0])
                else:
                    fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)
            save_path = os.path.join(self.config.result_path, '{}_{}_fake.png'.format(i+1, e))

            if not self.config.log_space:
                save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            else:
                fake_images = self.denorm(fake_images.data)*255.0
                fake_images = torch.pow(2.71828182846, fake_images/255.0*np.log(256.0))-1.0
                fake_images = fake_images/255.0
                fake_images = fake_images.clamp(0.0, 1.0)
                save_image(fake_images, save_path, nrow=1, padding=0)

            print('Translated test images and saved into "{}"..!'.format(save_path))
        self.G.train()

    def test_save_single_img(self, target_illu=7):

        # Load trained parameters
        G_path = os.path.join(self.config.model_save_path, '{}_G.pth'.format(self.config.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        data_loader = self.test_loader

        accumulate_board = [0] * 346

        k = 0
        for i, (imgs, img_names) in enumerate(data_loader):

            real_x = self.to_var(imgs[0], volatile=True)
            target_c_list = self.make_celeb_labels(len(real_x))
            target_c = target_c_list[target_illu]

            # Start translations
            fake_image_list = []
            if self.config.use_gpb:
                fake_image_list.append(self.G(real_x, target_c)[0])
            else:
                fake_image_list.append(self.G(real_x, target_c))
            # fake_image_list.append(real_x)
            fake_images = torch.cat(fake_image_list, dim=3)

            for j in range(len(real_x)):

                img_name = img_names[j]
                
                # img_foler = '/'.join(img_name.split('/')[-3:-1])
                img_foler = img_name.split('/')[0]
                img_name = img_name.split('/')[-1]

                save_path = self.config.result_path + '/multipie/%s/'%self.config.test_model + img_foler+'/'

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # save_path = os.path.join(save_path, idx+'_'+str(accumulate_board[int(idx)-1])+'.png')
                save_path = os.path.join(save_path, img_name)

                # save_image(self.denorm(fake_images[j].data), save_path, nrow=1, padding=0)

                if not self.config.log_space:
                    save_image(self.denorm(fake_images[j].data), save_path, nrow=1, padding=0)
                else:
                    fake_images_ = self.denorm(fake_images[j].data.cpu())*255.0
                    fake_images_ = torch.pow(2.71828182846, torch.FloatTensor(fake_images_/255.0*np.log(256.0)))-1.0
                    fake_images_ = fake_images_/255.0
                    fake_images_ = fake_images_.clamp(0.0, 1.0)
                    save_image(fake_images_, save_path, nrow=1, padding=0)

                print('Translated test images and saved into "{}"..!'.format(save_path))
                print(k+1)
                k+=1

    # def extract_feature(self):

    #     # Load trained parameters
    #     D_path = os.path.join(self.config.model_save_path, '{}_D.pth'.format(self.config.test_model))
    #     self.D.load_state_dict(torch.load(D_path))
    #     self.D.eval()

    #     data_loader = self.test_loader

    #     k = 0
    #     feature_list = np.zeros([0, 512])
    #     name_list = []

    #     for i, (real_x, img_names) in enumerate(data_loader):
            
    #         name_list.append(img_names)
            
    #         real_x = self.to_var(real_x, volatile=True)

    #         # Start extract_feature
    #         b = self.D(real_x)[-1].data.cpu().numpy()

    #         feature_list = np.concatenate((feature_list, b), axis=0)
    #         print(feature_list.shape)
    #         print(len(name_list))

    #     np.save('feature_'+self.config.test_model, feature_list)
    #     name_list = [a for l in name_list for a in l]
    #     np.save('name_'+self.config.test_model, name_list)
