from opt import config_parser
from torch.utils.data import DataLoader

# models
from models import *
from renderer import *
from utils import *
from data.ray_utils import ray_marcher, ray_marcher_fine

import imageio

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers
from render_video import render_video
from data.llff import LLFFDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, depth_pred, depth_gt, mask=None):
        if None == mask:
            mask = depth_gt > 0
        loss = self.loss(depth_pred[mask], depth_gt[mask]) * 2 ** (1 - 2)
        return loss


class MVSSystem(LightningModule):
    def __init__(self, args):
        super(MVSSystem, self).__init__()
        self.args = args
        self.args.feat_dim = 8 + 3 * 4
        self.args.dir_dim = 3
        self.idx = 0

        self.loss = SL1Loss()

        # Create nerf model
        self.render_kwargs_train, self.render_kwargs_test, start, self.grad_vars = create_nerf_mvs(args, use_mvs=True,
                                                                                                   dir_embedder=False,
                                                                                                   pts_embedder=True)
        filter_keys(self.render_kwargs_train)

        # Create mvs model
        self.MVSNet = self.render_kwargs_train['network_mvs']
        self.render_kwargs_train.pop('network_mvs')

        self.train_dataset = LLFFDataset(args, split='train')
        self.train_dataset_depth = LLFFDataset(args, split='train_depth')
        self.val_dataset = LLFFDataset(args, split='val')
        self.init_volume()
        self.grad_vars += list(self.volume.parameters())

    def init_volume(self):

        self.imgs, self.proj_mats, self.near_far_source, self.pose_source = self.train_dataset.read_source_views(
            device=device)
        ckpts = torch.load(args.ckpt)
        if 'volume' not in ckpts.keys():
            self.MVSNet.train()
            with torch.no_grad():
                volume_feature, _, _ = self.MVSNet(self.imgs, self.proj_mats, self.near_far_source, pad=args.pad,
                                                   lindisp=args.use_disp)
        else:
            volume_feature = ckpts['volume']['feat_volume']
            print('load ckpt volume.')
        self.imgs = self.unpreprocess(self.imgs)

        # project colors to a volume
        self.density_volume = None
        if args.use_color_volume or args.use_density_volume:
            D, H, W = volume_feature.shape[-3:]
            intrinsic, c2w = self.pose_source['intrinsics'][0].clone(), self.pose_source['c2ws'][0]
            intrinsic[:2] /= 4
            vox_pts = get_ptsvolume(H - 2 * args.pad, W - 2 * args.pad, D, args.pad, self.near_far_source, intrinsic,
                                    c2w)

            self.color_feature = build_color_volume(vox_pts, self.pose_source, self.imgs, with_mask=True).view(D, H, W,
                                                                                                               -1).unsqueeze(
                0).permute(0, 4, 1, 2, 3)  # [N,D,H,W,C]
            if args.use_color_volume:
                volume_feature = torch.cat((volume_feature, self.color_feature), dim=1)  # [N,C,D,H,W]

            if args.use_density_volume:
                self.vox_pts = vox_pts

            else:
                del vox_pts

        self.volume = RefVolume(volume_feature.detach()).to(device)
        del volume_feature

    def update_density_volume(self):
        with torch.no_grad():
            network_fn = self.render_kwargs_train['network_fn']
            network_query_fn = self.render_kwargs_train['network_query_fn']

            D, H, W = self.volume.feat_volume.shape[-3:]
            features = torch.cat((self.volume.feat_volume, self.color_feature), dim=1).permute(0, 2, 3, 4, 1).reshape(
                D * H, W, -1)
            self.density_volume = render_density(network_fn, self.vox_pts, features, network_query_fn).reshape(D, H, W)
        del features

    def decode_batch(self, batch):
        rays = batch[0]['rays'].squeeze()  # (B, 8)
        rgbs = batch[0]['rgbs'].squeeze()  # (B, 3)
        depths = batch[1]['depths'].squeeze()
        depRays = batch[1]['depRays'].squeeze()
        return rays, rgbs, depths, depRays

    def decode_batch_val(self, batch):
        rays = batch['rays'].squeeze()  # (B, 8)
        rgbs = batch['rgbs'].squeeze()  # (B, 3)
        return rays, rgbs

    def unpreprocess(self, data, shape=(1, 1, 3, 1, 1)):
        # to unnormalize image for visualization
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)
        return (data - mean) / std

    def forward(self):
        return

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.grad_vars, lr=self.args.lrate, betas=(0.9, 0.999))
        scheduler = get_scheduler(self.args, self.optimizer)
        return [self.optimizer], [scheduler]

    # FIXME: update learning rate
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train_dataloader(self):
        return [DataLoader(self.train_dataset,
                           shuffle=True,
                           num_workers=16,
                           batch_size=args.batch_size,
                           pin_memory=True),
                DataLoader(self.train_dataset_depth,
                           shuffle=True,
                           num_workers=16,
                           batch_size=args.batch_size,
                           pin_memory=True)
                ]

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=16,
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        # rays (batch_size, 8),
        # rgbs_target (batch_size, 3),
        # depths_target (batch_size),
        # depRays (batch_size, 2)
        rays, rgbs_target, depths_target, depRays = self.decode_batch(batch)

        if args.use_density_volume and 0 == self.global_step % 200:
            self.update_density_volume()
        rays = torch.cat([rays, depRays], 0)
        # xyz_coarse_sampled (B, N_samples, 3)
        # rays_o (B, 3)
        # rays_d (B, 3)
        # z_vals (B, N_samples)
        xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays, N_samples=args.N_samples,
                                                                 lindisp=args.use_disp, perturb=args.perturb)
        xyz_coarse_sampled = xyz_coarse_sampled.float()
        rays_o = rays_o.float()
        rays_d = rays_d.float()
        z_vals = z_vals.float()
        # Converting world coordinate to ndc coordinate
        H, W = self.imgs.shape[-2:]
        inv_scale = torch.tensor([W - 1, H - 1]).to(device)
        w2c_ref, intrinsic_ref = self.pose_source['w2cs'][0], self.pose_source['intrinsics'][0]
        xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                     near=self.near_far_source[0], far=self.near_far_source[1], pad=args.pad,
                                     lindisp=args.use_disp)

        # important sampleing
        if self.density_volume is not None and args.N_importance > 0:
            xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher_fine(rays, self.density_volume, z_vals, xyz_NDC,
                                                                          N_importance=args.N_importance)
            xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                         near=self.near_far_source[0], far=self.near_far_source[1], pad=args.pad,
                                         lindisp=args.use_disp)

        # rendering
        rgbs, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source, xyz_coarse_sampled, xyz_NDC,
                                                               z_vals, rays_o, rays_d,
                                                               self.volume, self.imgs, **self.render_kwargs_train)
        rgbs = rgbs[:args.batch_size, :]
        depth_pred = depth_pred[args.batch_size:]
        log, loss = {}, 0
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], rgbs_target)
            loss = loss + img_loss0
            psnr0 = mse2psnr2(img_loss0.item())
            self.log('train/PSNR0', psnr0.item(), prog_bar=True)

        ##################  rendering #####################
        if self.args.with_rgb_loss:
            img_loss = img2mse(rgbs, rgbs_target)
            loss += img_loss
            psnr = mse2psnr2(img_loss.item())

        if self.args.depth_loss:
            depth_loss = img2mse(depth_pred, depths_target)
            loss += 0.1 * depth_loss

            with torch.no_grad():
                self.log('train/loss', loss, prog_bar=True)
                self.log('train/img_mse_loss', img_loss.item(), prog_bar=False)
                self.log('train/PSNR', psnr.item(), prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):

        self.MVSNet.train()
        rays, img = self.decode_batch_val(batch)
        img = img.cpu()  # (H, W, 3)
        N_rays_all = rays.shape[0]

        ##################  rendering #####################
        keys = ['val_psnr_all']
        log = init_log({}, keys)
        with torch.no_grad():

            rgbs, depth_preds = [], []
            for chunk_idx in range(N_rays_all // args.chunk + int(N_rays_all % args.chunk > 0)):

                xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(
                    rays[chunk_idx * args.chunk:(chunk_idx + 1) * args.chunk],
                    N_samples=args.N_samples, lindisp=args.use_disp)

                # Converting world coordinate to ndc coordinate
                H, W = img.shape[:2]
                inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                w2c_ref, intrinsic_ref = self.pose_source['w2cs'][0], self.pose_source['intrinsics'][0].clone()
                intrinsic_ref[:2] *= args.imgScale_test / args.imgScale_train
                xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                             near=self.near_far_source[0], far=self.near_far_source[1],
                                             pad=args.pad * args.imgScale_test, lindisp=args.use_disp)

                # important sampleing
                if self.density_volume is not None and args.N_importance > 0:
                    xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher_fine(
                        rays[chunk_idx * args.chunk:(chunk_idx + 1) * args.chunk],
                        self.density_volume, z_vals, xyz_NDC, N_importance=args.N_importance)
                    xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                                 near=self.near_far_source[0], far=self.near_far_source[1],
                                                 pad=args.pad, lindisp=args.use_disp)

                # rendering
                rgb, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source, xyz_coarse_sampled,
                                                                      xyz_NDC, z_vals, rays_o, rays_d,
                                                                      self.volume, self.imgs,
                                                                      **self.render_kwargs_train)

                rgbs.append(rgb.cpu());
                depth_preds.append(depth_pred.cpu())

            rgbs, depth_r = torch.clamp(torch.cat(rgbs).reshape(H, W, 3), 0, 1), torch.cat(depth_preds).reshape(H, W)
            img_err_abs = (rgbs - img).abs()

            log['val_psnr_all'] = mse2psnr(torch.mean(img_err_abs ** 2))
            depth_r, _ = visualize_depth(depth_r, self.near_far_source)
            self.logger.experiment.add_images('val/depth_gt_pred', depth_r[None], self.global_step)

            img_vis = torch.stack((img, rgbs, img_err_abs.cpu() * 5)).permute(0, 3, 1, 2)
            self.logger.experiment.add_images('val/rgb_pred_err', img_vis, self.global_step)
            os.makedirs(f'runs_fine_tuning/{self.args.expname}/{self.args.expname}/', exist_ok=True)

            img_vis = torch.cat((img, rgbs, img_err_abs * 10, depth_r.permute(1, 2, 0)), dim=1).numpy()
            imageio.imwrite(
                f'runs_fine_tuning/{self.args.expname}/{self.args.expname}/{self.global_step:08d}_{self.idx:02d}.png',
                (img_vis * 255).astype('uint8'))
            self.idx += 1

        return log

    def validation_epoch_end(self, outputs):

        mean_psnr_all = torch.stack([x['val_psnr_all'] for x in outputs]).mean()
        self.log('val/PSNR_all', mean_psnr_all, prog_bar=True)
        return

    def save_ckpt(self, name='latest'):
        save_dir = f'runs_fine_tuning/{self.args.expname}/ckpts'
        os.makedirs(save_dir, exist_ok=True)
        path = f'{save_dir}/{name}.tar'
        ckpt = {
            'global_step': self.global_step,
            'network_fn_state_dict': self.render_kwargs_train['network_fn'].state_dict(),
            'volume': self.volume.state_dict(),
            'network_mvs_state_dict': self.MVSNet.state_dict()}
        if self.render_kwargs_train['network_fine'] is not None:
            ckpt['network_fine_state_dict'] = self.render_kwargs_train['network_fine'].state_dict()
        torch.save(ckpt, path)
        print('Saved checkpoints at', path)

def gen_pairs(args):
    pairs = torch.load('./configs/pairs.th')
    print('Generating pairs')
    scene = args.datadir.split('/')[-1]
    poses_bounds = np.load(os.path.join(args.datadir, 'poses_bounds.npy'))  # (N_images, 17)
    N_images = poses_bounds.shape[0]
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    poses = np.concatenate([poses[..., 1:2], - poses[..., :1], poses[..., 2:4]], -1)

    pair_idx = torch.randperm(len(poses))[:N_images].tolist()
    if f'{scene}_test' in pairs and f'{scene}_val' in pairs and f'{scene}_train':
        return
    if len(pair_idx) < 6:
        pairs[f'{scene}_test'] = []
        pairs[f'{scene}_val'] = []
        pairs[f'{scene}_train'] = pair_idx
    else:
        pairs[f'{scene}_test'] = pair_idx[::6]
        pairs[f'{scene}_val'] = pair_idx[::6]
        pairs[f'{scene}_train'] = np.delete(pair_idx, range(0, N_images, 6)).tolist()
    torch.save(pairs, './configs/pairs.th')

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(False)
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    gen_pairs(args)
    if args.render_only:
        render_video(args)
    else:
        system = MVSSystem(args)
        checkpoint_callback = ModelCheckpoint(os.path.join(f'runs_fine_tuning/{args.expname}/ckpts/', '{epoch:02d}'),
                                              monitor='val/PSNR',
                                              mode='max',
                                              save_top_k=0)

        logger = loggers.TestTubeLogger(
            save_dir="runs_fine_tuning",
            name=args.expname,
            debug=False,
            create_git_tag=False
        )

        args.num_gpus, args.use_amp = 1, False
        trainer = Trainer(max_epochs=args.num_epochs,
                          checkpoint_callback=checkpoint_callback,
                          logger=logger,
                          weights_summary=None,
                          progress_bar_refresh_rate=1,
                          gpus=args.num_gpus,
                          distributed_backend='ddp' if args.num_gpus > 1 else None,
                          num_sanity_val_steps=1,  # if args.num_gpus > 1 else 5,
                          val_check_interval=500,
                          benchmark=True,
                          precision=16 if args.use_amp else 32,
                          amp_level='O1')

        trainer.fit(system)
        system.save_ckpt()
        args.is_finetuned = True
        render_video(args)
