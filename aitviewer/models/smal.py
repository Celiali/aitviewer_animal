import CONFIG
import sys
sys.path.append('/home/cil/Documents/project/PPhorseMoshOpensource/aitviewer_animal/')
import numpy as np
import smplx
import torch
import trimesh
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

class SMAL(smplx.SMPL):
    NUM_JOINTS = 34
    NUM_BODY_JOINTS = 34
    SHAPE_SPACE_DIM = 41

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertex_joint_selector.extra_joints_idxs = torch.empty(0, dtype=torch.int32)

class HSMAL(smplx.SMPL):
    NUM_JOINTS = 35
    NUM_BODY_JOINTS = 35
    SHAPE_SPACE_DIM = 1369

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertex_joint_selector.extra_joints_idxs = torch.empty(0, dtype=torch.int32)


class SMALLayer(SMPLLayer):
    def __init__(self, model_path, model_cls, model_type="hsmal", num_betas=41, device=C.device, dtype=C.f_precision):
        super(SMPLLayer, self).__init__()

        self.num_betas = num_betas

        self.bm = model_cls(model_path=model_path, num_betas=num_betas)
        self.bm.to(device=device, dtype=dtype)

        self.model_type = model_type
        self._parents = None
        self._children = None
        self._closest_joints = None
        self._vertex_faces = None
        self._faces = None


def load_template():
    '''
    examples/load_template.py
    '''
    # Create a neutral hSMAL T Pose.
    device = "cuda:0"
    #device = "cpu"
    smal_layer = SMALLayer(
        model_path="/home/cil/Documents/project/code_from_others/silvia/smalr_online/smpl_models/smal_00781_4_all.pkl",
        model_type = "smal",
        model_cls=SMAL,
        device=device,
        num_betas=9,
    )
    smal_template = SMPLSequence.t_pose(smal_layer, name='SMAL', z_up=True)

    hsmal_layer = SMALLayer(
        model_path="/home/cil/Documents/project/code_from_others/aitviewer/data/smplx_models/hsmal/my_smpl_0000_horse_new_skeleton_horse.pkl",
        model_type = "hsmal",
        model_cls=HSMAL,
        device=device,
        num_betas=10,
    )
    hsmal_template = SMPLSequence.t_pose(hsmal_layer, name='HSMAL', z_up=True)

    # Display in viewer.
    v = Viewer()
    v.scene.add(smal_template,hsmal_template)
    v.run()

if __name__ == '__main__':
    load_template()