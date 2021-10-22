import torch.optim as optim
import dataset as datasets
from data import processing, sampler, DataLoader
import models.deeprep.deeprepnet as deeprep_nets
import actors.deeprep_actors as deeprep_actors
from trainers import SimpleTrainer
import data.transforms as tfm
from admin.multigpu import MultiGPU
from models.loss.image_quality_v2 import PSNR, PixelWiseError, MappedLoss


def run(settings):
    settings.description = ''
    settings.batch_size = 8
    settings.num_workers = 8
    settings.multi_gpu = False
    settings.print_interval = 1

    settings.crop_sz = 128
    settings.burst_sz = 8
    settings.pre_downsample_factor = 4
    settings.max_jitter_small = 2
    settings.max_jitter_large = 16
    openimages_train = datasets.OpenImagesDataset(split='train')
    zurich_val = datasets.ZurichRAW2RGB(split='test')

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True))

    data_processing_train = processing.DenoisingProcessing(crop_sz=settings.crop_sz, burst_size=settings.burst_sz,
                                                           pre_downsample_factor=settings.pre_downsample_factor,
                                                           max_jitter_small=settings.max_jitter_small,
                                                           max_jitter_large=settings.max_jitter_large,
                                                           transform=transform_train,
                                                           return_grayscale=True)

    data_processing_val = processing.DenoisingProcessing(crop_sz=settings.crop_sz, burst_size=settings.burst_sz,
                                                         pre_downsample_factor=settings.pre_downsample_factor,
                                                         max_jitter_small=settings.max_jitter_small,
                                                         max_jitter_large=settings.max_jitter_large,
                                                         transform=transform_val,
                                                         return_grayscale=True)

    # Train sampler and loader
    dataset_train = sampler.RandomImage([openimages_train], [1],
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)
    dataset_val = sampler.IndexedImage(zurich_val, processing=data_processing_val)

    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=5)

    net = deeprep_nets.deeprep_denoise_custom_flow_iccv21(num_iter=3, enc_dim=32,
                                                          enc_num_res_blocks=4,
                                                          enc_out_dim=64,
                                                          dec_dim_pre=64,
                                                          dec_num_res_blocks=9,
                                                          dec_in_dim=16,
                                                          use_feature_regularization=True,
                                                          use_noise_estimate=True,
                                                          wp_project_dim=16,
                                                          wp_offset_feat_dim=8,
                                                          wp_num_offset_feat_extractor_res=0,
                                                          wp_num_weight_predictor_res=1,
                                                          align_init_dim=32, align_num_res_blocks=6,
                                                          align_ds_factor=4,
                                                          offset_cdim=64,
                                                          offset_predictor_dims=(128, 64),
                                                          corr_max_disp=(2, 3),
                                                          color_input=False
                                                          )
    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {
        'rgb': MappedLoss(base_loss=PixelWiseError(metric='l1', boundary_ignore=4), mapping_fn=None),
        'psnr': MappedLoss(base_loss=PSNR(boundary_ignore=4), mapping_fn=None),
        'offsets': PixelWiseError(metric='l2_sqrt', boundary_ignore=4),
        'photometric': PixelWiseError(metric='l1', boundary_ignore=4)
    }

    loss_weight = {
        'rgb': 1.0,
        'offsets': 0.0,
        'photometric': 1.0
    }

    actor = deeprep_actors.DeepRepDenoisingActor(net=net, objective=objective, loss_weight=loss_weight)

    optimizer = optim.Adam([{'params': actor.net.parameters(), 'lr': 1e-4}],
                           lr=2e-4)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)

    trainer = SimpleTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(250, load_latest=True, fail_safe=True)
