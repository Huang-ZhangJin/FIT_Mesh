import os

import torch
import trimesh
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree

import utils
from utils import set_random_seed
from sdfnet import SDFNetwork
from data_utils import sampleGT

set_random_seed(49)

class Fitter:
    def __init__(
        self, 
        data : torch.Tensor, 
        expname : str = "test",
    ):
        self.expname = expname
        
        self.data = data
        self.input_min = torch.min(self.data[..., :3], dim=0)[0].squeeze().cpu().numpy()
        self.input_max = torch.max(self.data[..., :3], dim=0)[0].squeeze().cpu().numpy()

        self.sdf_net = SDFNetwork(
            dim_output=1,
            dim_hidden=512,
            n_layers=8,
            skip_in=(4,),
            level=10,
            geometric_init=True,
            inside_outside=False,
            bias=max(0.5, max(self.input_max - self.input_min)) 
        ).cuda()

        ptree = cKDTree(self.data)
        sigma_set = []
        for p in np.array_split(self.data, 100, axis=0):
            d = ptree.query(p, 50 + 1)
            sigma_set.append(d[0][:, -1])
        sigmas = np.concatenate(sigma_set)
        self.local_sigma = torch.from_numpy(sigmas).float().cuda()
        self.global_sigma = 1.8
        self.sampler = utils.NormalPerPoint(self.global_sigma, self.local_sigma)

        self.data = data.float().cuda()

        learning_rate_schedule = [{
                                "Type" : "Step",
			                    "Initial" : 0.005,
			                    "Interval" : 2000,
			                    "Factor" : 0.5
			                    }]
        self.lr_schedules = self.get_learning_rate_schedules(learning_rate_schedule)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.sdf_net.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0),
                    "weight_decay": 0,
                },
            ])

        self.ek_bound = 1.5
        self.startepoch = 0
        self.points_batch = 16384
        self.nepoch = 100000
        self.global_step = 0

    def run(self):
        for epoch in range(self.startepoch, self.nepoch+1):
            if epoch % 1000 == 0:
                self.export_mesh(
                    filename=os.path.join("outmesh", self.expname, f"{epoch}.ply"),
                    resolution = 128
                )

            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            cur_data = self.data[indices]

            mnfld_pnts = cur_data[:, :3]
            normals = cur_data[:, 3:]
            mnfld_sigma = self.local_sigma[indices]

            self.sdf_net.train()
            self.adjust_learning_rate(epoch)

            nonmndlf_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()

            # forward
            mnfld_pred, mnfld_grad = self.sdf_net(mnfld_pnts)
            nonmnfld_pred, nonmndlf_grad = self.sdf_net(nonmndlf_pnts)

            mnfld_loss = (mnfld_pred.abs()).mean()
            if epoch < 2000:
                grad_loss = 0.1 * ((nonmndlf_grad.norm(2, dim=-1) - 1) ** 2).mean()
            else:
                eikonal_points = torch.empty_like(mnfld_pnts[:2000, ...].detach()).uniform_(-self.ek_bound, self.ek_bound)
                _, normal_eik = self.sdf_net(eikonal_points)
                normal_eik = torch.cat([normal_eik, nonmndlf_grad], dim=-2)
                grad_loss = 0.05 * ((normal_eik.norm(2, dim=-1) - 1) ** 2).mean()
            normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
            loss = mnfld_loss + normals_loss + grad_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0 :
                print(f"epoch {epoch}/{self.nepoch} | loss {loss} | sdf {mnfld_loss} | norm {normals_loss} | ek {grad_loss}")


    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.sdf_net.state_dict()},
            os.path.join(self.checkpoints_path, self.expname, "model_"+str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.sdf_net.state_dict()},
            os.path.join(self.checkpoints_path, self.expname, "model_latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.expname,  "optim_"+str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.expname, "optim_latest.pth"))

    
    def get_learning_rate_schedules(self, schedule_specs):
        schedules = []
        for schedule_specs in schedule_specs:
            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )
            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )
        return schedules

    
    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)


    @torch.no_grad()
    def export_mesh(
        self,
        filename: str = "ckpt/sdf_mc.ply",
        resolution: int = 64,
        batch_size: int = 64**3,
        thresh: float=0,
        device: str="cuda:0"
    ) -> None:
        bound = max(self.input_max - self.input_min)
        centers_shape = (resolution, resolution, resolution)
        half_grid_size = bound / resolution + 0.1

        X = torch.linspace(
            self.input_min[0] - half_grid_size, self.input_max[0] + half_grid_size, resolution
        ).to(device)
        Y = torch.linspace(
            self.input_min[1] - half_grid_size, self.input_max[1] + half_grid_size, resolution
        ).to(device)
        Z = torch.linspace(
            self.input_min[2] - half_grid_size, self.input_max[2] + half_grid_size, resolution
        ).to(device)
        X, Y, Z = torch.meshgrid(X, Y, Z, indexing="ij")
        mc_grid = torch.stack((X, Y, Z), dim=-1).view(-1, 3)  # [N, 3]

        import cumcubes

        out_dir = os.path.dirname(filename)
        os.makedirs(out_dir, exist_ok=True)
        sdf = torch.zeros_like(mc_grid[..., 0])
        for i in range(0, len(mc_grid), batch_size):
            sdf[i : i + batch_size] = self.sdf_net.sdf(mc_grid[i : i + batch_size])[..., 0]
        sdf = sdf.reshape(centers_shape).contiguous()
        vertices, faces = cumcubes.marching_cubes(
            sdf, thresh, ([-bound] * 3, [bound] * 3), verbose=False
        )
        cumcubes.save_mesh(vertices, faces, filename=filename)

        torch.cuda.empty_cache()



if __name__=='__main__':
    inmesh = "data/test1_rot.obj"

    meshpts_file = f"{inmesh}".replace(".obj", ".xyz")
    if os.path.exists(meshpts_file):
        data = np.loadtxt(meshpts_file)
    else:
        mesh = trimesh.load(inmesh)
        datapts, datanormal = sampleGT(mesh, samplepointsnum=1000000)
        data = np.concatenate((datapts, datanormal), axis=1)
        np.savetxt(meshpts_file, data, fmt='%.6f')
    data = torch.tensor(torch.from_numpy(data)).float()


    fitter = Fitter(data, expname="test1")
    fitter.run()
    