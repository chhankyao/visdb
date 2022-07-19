import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from model import get_model
from dataset import MultipleDatasets

# dynamic dataset import
for i in range(len(cfg.trainset_3d)):
    exec('from ' + cfg.trainset_3d[i] + ' import ' + cfg.trainset_3d[i])
for i in range(len(cfg.trainset_2d)):
    exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        if cfg.stage == 'lixel':
            optimizer = torch.optim.Adam(list(model.module.pose_backbone.parameters()) + \
                                         list(model.module.pose_net.parameters()) + \
                                         list(model.module.pose2feat.parameters()) + \
                                         list(model.module.mesh_backbone.parameters()) + \
                                         list(model.module.mesh_net.parameters()), lr=cfg.lr)
            print('The parameters of pose_backbone, pose_net, pose2feat, mesh_backbone, and mesh_net are added to the optimizer.')
        else:
            optimizer = torch.optim.Adam(model.module.param_regressor.parameters(), lr=cfg.lr)
            print('The parameters of all modules are added to the optimizer.')
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)
        # optimizer.load_state_dict(ckpt['optimizer'])
        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))
        
        if len(trainset3d_loader) > 0 and len(trainset2d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset3d_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
            trainset2d_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
            trainset_loader = MultipleDatasets([trainset3d_loader, trainset2d_loader], make_same_len=True)
        elif len(trainset3d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
        elif len(trainset2d_loader) > 0:
            self.vertex_num = trainset2d_loader[0].vertex_num
            self.joint_num = trainset2d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
        else:
            assert 0, "Both 3D training set and 2D training set have zero length."
            
        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.train_batch_size, 
                                          shuffle=True, num_workers=cfg.num_thread, pin_memory=True, drop_last=True)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model(self.vertex_num, self.joint_num, 'train')
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer


class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.testset)(transforms.ToTensor(), "test")
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.test_batch_size, 
                                     shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        
        self.testset = testset_loader
        self.vertex_num = testset_loader.vertex_num
        self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(self.vertex_num, self.joint_num, 'test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)
