import os
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import imageio
import time
from tqdm import tqdm
import copy
import random

from dataset import MeshroomRadialK3Dataset
from evaluation_metrics import psnr, epoch_psnr
from utils import to_device, load_obj_mask_as_tensor, load_cameras
from neutex.neutex import NeuTexTrainWrapper


class Trainer:
    def __init__(self, 
                 model,
                 optim,
                 loss_fn, 
                 renderer,
                 data,
                 mesh,
                 config,
                 device):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.renderer = renderer
        self.mesh = mesh
        self.config = config

        self.use_lr_scheduler = config["training"].get("use_lr_scheduler", False)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode="min", factor=0.2, verbose=True)

        self.dataset_type = self.config["data"].get("type")

        self.H = config["data"]["img_height"]
        self.W = config["data"]["img_width"]

        self.train_data_loader = data["train"]
        self.val_data_loader = data["val"]
        if self.dataset_type is None:
            self.val_render_infos = list(zip(config["data"]["eval_render_input_paths"], config["data"]["eval_render_img_names"]))
        self.test_data_loader = data.get("test", None)

        self.out_dir = self.config["training"]["out_dir"]

        log_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        self.render_every = self.config["training"]["render_every"]
        self.print_every = self.config["training"]["print_every"]
        self.epochs = self.config["training"]["epochs"]

        self.checkpoint_every = self.config["training"].get("checkpoint_every")
        if self.checkpoint_every is not None:
            self.checkpoint_path = os.path.join(self.out_dir, "checkpoint.pt")

        self.device = device

        self.model_config = self.config["model"]
        self.best_model_weights_path = os.path.join(self.out_dir, "model.pt")
        self.best_model = copy.deepcopy(self.model)
        
        self.model_last_epoch_path = os.path.join(self.out_dir, "model_last_epoch.pt")

    def _train_step(self, batch):
        if isinstance(self.model, NeuTexTrainWrapper):
            loss, pred_rgbs = self.model(batch)
        else:
            pred_rgbs = self.model(batch)
            loss = self.loss_fn(pred_rgbs, batch["expected_rgbs"])

        # Setting the gradients to None is more efficient then zeroing them out.
        # https://pytorch.org/docs/master/generated/torch.optim.Optimizer.zero_grad.html?highlight=zero_grad
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        
        return loss.item(), pred_rgbs

    def write_vis_metrics_to_tensorboard(self, img_name, rendered_img, gt_img, obj_mask_1d, epoch):
        self.writer.add_image(img_name, rendered_img.transpose(2, 0, 1), global_step=epoch)

        # Calculate the PSNR, abs. dist. and 2D mean dist. between GT and rendered view

        self.writer.add_scalar(f"{img_name}_psnr", psnr(rendered_img, gt_img, obj_mask_1d.numpy()), epoch)

        # Compute the 2D mean distance
        mean_distance_2d = 1. - np.mean(np.abs(rendered_img - gt_img), -1)  # H x W
        mean_distance_2d = np.repeat(mean_distance_2d[None, ...], 3, axis=0)  # 3 x H x W

        self.writer.add_image(f"{img_name}_2d_mean_distance", mean_distance_2d, global_step=epoch)

        # Compute a scalar value for the absolute distance
        rendered_img = rendered_img.reshape(-1, 3)[obj_mask_1d]
        gt_img = gt_img.reshape(-1, 3)[obj_mask_1d]
        total_dist = np.abs(gt_img - rendered_img).sum()

        self.writer.add_scalar(f"{img_name}_dist", total_dist, epoch)

    @torch.no_grad()
    def _render_view_for_tensorboard(self, input_path, img_name, epoch):
        # Load the object mask and use it for speeding up rendering.
        obj_mask = load_obj_mask_as_tensor(input_path)
        obj_mask_1d = obj_mask.reshape(-1)

        # Load camera parameters
        camCv2world, K = load_cameras(input_path)
        # Render view
        rendered_img = self.renderer.render(camCv2world, 
                                            K, 
                                            obj_mask_1d=obj_mask_1d)

        # Load the original image
        gt_img = imageio.imread(os.path.join(input_path, "image", "000.png"))
        gt_img = gt_img.astype(np.float32) / 255.
        # Mask out background
        orig_shape = gt_img.shape
        gt_img = gt_img.reshape(-1, 3)
        gt_img[obj_mask_1d == False] = 1.
        gt_img = gt_img.reshape(orig_shape)

        self.write_vis_metrics_to_tensorboard(img_name, rendered_img, gt_img, obj_mask_1d, epoch)
    
    @torch.no_grad()
    def _render_views_for_tensorboard_meshroom_radial_k3(self, epoch):
        dataset_path = self.config["data"]["vis_dataset_path"]
        vis_dataset = MeshroomRadialK3Dataset(dataset_path, 
                                              self.config["data"]["vis_split"], 
                                              H=self.config["data"]["img_height"], 
                                              W=self.config["data"]["img_width"])
        vis_dataloader = torch.utils.data.DataLoader(vis_dataset,
                                                     batch_size=None,
                                                     shuffle=False,
                                                     drop_last=False)

        for idx, item in enumerate(tqdm(vis_dataloader)):
            camCv2world = item["camCv2world"]
            K = item["K"]
            gt_img = item["img"].numpy()
            obj_mask_1d = item["obj_mask_1d"]
            distortion_params = item["distortion_params"]
            distortion_type = item["distortion_type"]

            # Render view
            rendered_img = self.renderer.render(camCv2world, 
                                                K, 
                                                distortion_coeffs=distortion_params, 
                                                distortion_type=distortion_type)

            self.write_vis_metrics_to_tensorboard(f"meshroom_radial_k3_view_{idx}", rendered_img, gt_img, obj_mask_1d, epoch)

    @torch.no_grad()
    def _eval_step(self, model, batch):
        pred_rgbs = model(batch)
        loss = self.loss_fn(pred_rgbs, batch["expected_rgbs"])
        return loss, pred_rgbs

    def evaluate(self, epoch=None):
        self.model.eval()

        accumulated_loss = 0
        accumulated_l2_loss = 0
        total = 0

        for batch in self.val_data_loader:
            batch = to_device(batch, device=self.device)

            loss, pred_rgbs = self._eval_step(self.model, batch)

            batch_size = batch["expected_rgbs"].size()[0]
            accumulated_l2_loss += F.mse_loss(pred_rgbs, batch["expected_rgbs"], reduction="sum").item()
            accumulated_loss += loss.item() * batch_size
            total += batch_size

        val_loss = accumulated_loss / total
        self.writer.add_scalar("Val_Loss", val_loss, epoch)

        val_psnr = epoch_psnr(accumulated_l2_loss / total)
        self.writer.add_scalar("Val Epoch-PSNR", val_psnr, epoch)

        return val_loss, val_psnr
        
    def test(self):
        if self.test_data_loader is None:
            return

        self.best_model.eval()

        accumulated_loss = 0
        total = 0

        for batch in self.test_data_loader:
            batch = to_device(batch, device=self.device)

            loss, pred_rgbs = self._eval_step(self.best_model, batch)
            
            batch_size = batch["expected_rgbs"].size()[0]
            accumulated_loss += loss.item() * batch_size
            total += batch_size

        test_loss = accumulated_loss / total
        self.writer.add_scalar("Test Loss", test_loss)

        print(f"Test Loss: {test_loss}")

        return test_loss
    
    def _init_or_load_checkpoint(self):
        if self.checkpoint_every is None or not os.path.exists(self.checkpoint_path):
            return 0
        
        # Load from checkpoint
        print("Restoring from checkpoint...")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore random number generator states for reproducability
        torch.random.set_rng_state(checkpoint["pytorch_random_state"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])

        print("Done.")
        return checkpoint["epoch"] + 1  # +1 to advance to the next epoch

    def train(self):
        print("Starting training...")

        epoch_start = self._init_or_load_checkpoint()

        min_val_loss = 1.
        
        for epoch in range(epoch_start, self.epochs):
            accumulated_loss = 0
            accumulated_l2_loss = 0
            total = 0

            # Train step
            self.model.train()

            epoch_start = time.time()
            for batch in self.train_data_loader:
                batch = to_device(batch, device=self.device)

                loss, pred_rgbs = self._train_step(batch)

                batch_size = batch["expected_rgbs"].size()[0]
                accumulated_l2_loss += F.mse_loss(pred_rgbs, batch["expected_rgbs"], reduction="sum").item()
                accumulated_loss += loss * batch_size
                total += batch_size

            epoch_end = time.time()

            train_loss = accumulated_loss / total
            self.writer.add_scalar("Train_Loss", train_loss, epoch)

            train_psnr = epoch_psnr(accumulated_l2_loss / total)
            self.writer.add_scalar("Train Epoch-PSNR", train_psnr, epoch)

            # Evaluation step
            val_loss, val_psnr = self.evaluate(epoch)

            # Store the weights of the best model
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # https://pytorch.org/tutorials/beginner/saving_loading_models.html
                torch.save(self.model.state_dict(), self.best_model_weights_path)
                self.best_model = copy.deepcopy(self.model)

            # LR scheduling
            if self.use_lr_scheduler:
                self.lr_scheduler.step(val_loss)

            if epoch == 0 or (epoch + 1) % self.print_every == 0:
                print(f"Epoch: {epoch + 1} / {self.epochs}, Train Loss: {train_loss}, Train PSNR: {train_psnr}, "
                      f"Val Loss: {val_loss}, Val PSNR: {val_psnr}"
                      f"Epoch Time: {epoch_end - epoch_start}s")

            if epoch == 0 or (epoch + 1) % self.render_every == 0:
                # Visualize some data every now and then
                self.model.eval()
                print("Visualizing...")
                vis_start = time.time()

                if self.dataset_type is None:
                    for i, (input_path, img_name) in enumerate(tqdm(self.val_render_infos)):
                        self._render_view_for_tensorboard(input_path, f"img{i:03d}", epoch)
                elif self.dataset_type == "meshroom_radial_k3":
                    self._render_views_for_tensorboard_meshroom_radial_k3(epoch)
                else:
                    raise NotImplementedError(f"Unknown dataset type: {self.dataset_type}!")

                vis_end = time.time()
                print(f"Done with visualizations after {vis_end - vis_start} seconds.")

            if self.checkpoint_every is not None and epoch % self.checkpoint_every == 0:
                print("Saving checkpoint...")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                    # Save random number generator states for reproducibility
                    "pytorch_random_state": torch.random.get_rng_state(),
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                }, self.checkpoint_path)
                print("Done.")

            if epoch > 0 and (epoch+1) == 200:
                # Create a persistent checkpoint at the 200th epoch
                print(f"Persisting checkpoint at {epoch}...")
                # Checkpoint current state
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                    # Save random number generator states for reproducibility
                    "pytorch_random_state": torch.random.get_rng_state(),
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                }, os.path.join(self.out_dir, f"checkpoint_{epoch}.pt"))
                # Checkpoint best model so far
                torch.save(self.best_model.state_dict(), 
                           os.path.join(self.out_dir, f"best_model_checkpoint_{epoch}.pt"))
                print("Done.")

        # Test step
        self.test()
        print("Done.")

        torch.save(self.model.state_dict(), self.model_last_epoch_path)
