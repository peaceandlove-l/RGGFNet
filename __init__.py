# from velruse.app.baseconvert import base_n_decoder

def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'nyuv2rl', 'sunrgbd', 'sunrgbd224', 'zju']

    if cfg['dataset'] == 'nyuv2':
        from sam.lsy.mymodelnew.toolbox.datasets.nyuv2_new import NYUv2
        return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    if cfg['dataset'] == 'nyuv2rl':
        from sam.lsy.mymodelnew.toolbox.datasets.nyuv2_RL import NYUv2
        return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd':
        from sam.lsy.mymodelnew.toolbox.datasets.sunrgbd import SUNRGBD
        return SUNRGBD(cfg, mode='train'), SUNRGBD(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd224':
        from sam.lsy.mymodelnew.toolbox.datasets.sunrgbd224 import SUNRGBD224
        return SUNRGBD224(cfg, mode='train'), SUNRGBD224(cfg, mode='test')
    if cfg['dataset'] == 'zju':
        from sam.lsy.mymodelnew.toolbox.datasets.ZJU_RGBP import ZJU
        return ZJU(cfg, mode='train'), ZJU(cfg, mode='test')

# def get_teacher_model(cfg):
#     if cfg['model_name'] == 'ucnet':
#         from mymodelnew.FindTheBestDec.model.UCNet_b4_stu_indoor_RDE import Fusion
#         return Fusion(num_class=cfg['n_classes'])

def get_model(cfg):
    if cfg['model_name'] == 'sam':
        from sam.INdoorSAM import EnDecoderModel
        return EnDecoderModel()
    if cfg['model_name'] == 'DINO':
        from sam.lsy.mymodelnew.toolbox.Mymodels.DINO.DinoV3main import Dino
        return Dino(num_classes=41)
    if cfg['model_name'] == 'DINOtest':
        from sam.lsy.mymodelnew.toolbox.Mymodels.DINO.LoRADINO import LoRADinoWithText
        return LoRADinoWithText(num_classes=41)
    if cfg['model_name'] == 'DoraDino':
        from sam.lsy.mymodelnew.toolbox.Mymodels.DINO.paper3 import Dino
        return Dino(num_classes=38)
    if cfg['model_name'] == 'paper4':
        from sam.lsy.mymodelnew.toolbox.Mymodels.DINO.paper4_ngfc_3dgs_seg_agent import paper4
        return paper4(
            num_classes=cfg['n_classes'],
            freeze_backbone=cfg.get('freeze_backbone', True),
            field_dim=cfg.get('field_dim', 256),
            semantic_dim=cfg.get('semantic_dim', 128),
            decoder_dim=cfg.get('decoder_dim', 192),
            feature_layers=tuple(cfg.get('feature_layers', [0, 3, 7, 11])),
            num_gaussians_per_token=cfg.get('max_gaussians_per_token', 2),
            gaussian_kernel_size=cfg.get('max_render_kernel', 5),
            rags_steps=cfg.get('rags_steps', 2),
            policy_hidden_dim=cfg.get('policy_hidden_dim', 128),
            use_rgra_agent=cfg.get('use_rgra_agent', False),
            rgra_hidden_dim=cfg.get('rgra_hidden_dim', 32),
            rgra_refine_gain=cfg.get('rgra_refine_gain', 1.25),
        )
    if cfg['model_name'] == 'paper4rl':
        from sam.lsy.mymodelnew.toolbox.Mymodels.DINO.paper4_ngfc_3dgs_seg_1 import paper4
        return paper4(
            num_classes=cfg['n_classes'],
            freeze_backbone=cfg.get('freeze_backbone', True),
            field_dim=cfg.get('field_dim', 256),
            semantic_dim=cfg.get('semantic_dim', 128),
            decoder_dim=cfg.get('decoder_dim', 192),
            feature_layers=tuple(cfg.get('feature_layers', [0, 3, 7, 11])),
            num_gaussians_per_token=cfg.get('max_gaussians_per_token', 2),
            gaussian_kernel_size=cfg.get('max_render_kernel', 5),
            rags_steps=cfg.get('rags_steps', 2),
            policy_hidden_dim=cfg.get('policy_hidden_dim', 128),
        )
    if cfg['model_name'] == 'SAM3':
        from SAM3.INdoorSAM3 import SAM3Semantic
        return SAM3Semantic(num_classes=41)
    if cfg['model_name'] == 'inter':
        from sam.lsy.mymodelnew.toolbox.Mymodels.DINO.paper2 import st
        return st()
    if cfg['model_name'] == 'teacher':
        from sam.lsy.mymodelnew.toolbox.Mymodels.DINO.teDino import LoRADinoWithText
        return LoRADinoWithText(num_classes=41)
    if cfg['model_name'] == 'CMX':
        from sam.lsy.mymodelnew.toolbox.models.CMX.models.builder import EncoderDecoder
        return EncoderDecoder()
    if cfg['model_name'] == 'hitnet':
        from sam.lsy.mymodelnew.toolbox.models.TextureDiffusion.twig.model.cod import Hitnet
        return Hitnet()
    if cfg['model_name'] == 'dformer':
        from sam.lsy.mymodelnew.toolbox.Mymodels.DFormerv2.testmodel import test
        return test()
    if cfg['model_name'] == 'cirnet':
        from sam.lsy.mymodelnew.toolbox.models.CIRNet.model.CIRNet_Res50 import CIRNet
        return CIRNet()
    print('加载OK!!!')
# def get_model(cfg):
#     if cfg['model_name'] == 'new':
#         from mymodelnew.new_modl.new_model import NewMod
#         return NewMod(num_class=cfg['n_classes'])
#     if cfg['model_name'] == 'ucnet':
#         from mymodelnew.FindTheBestDec.model.hh import Fusion
#         return Fusion(num_class=cfg['n_classes'])
#     if cfg['model_name'] == 'acnet':
#         from mymodelnew.toolbox.models.ACNet2.acnet import ACNet
#         return ACNet(pretrained=True)
    # if cfg['model_name'] == 'cdinet':
    #     from toolbox.models.CDINet.model_vgg import CDINet
    #     return CDINet()
    # if cfg['model_name'] == 'ESAnet':
    #     from toolbox.models.ESANet.models.model import ESANet
    #     return ESANet()
    # if cfg['model_name'] == 'SGACnet':
    #     from toolbox.models_transbackbone.SGACNet.SGACNet.SGACNet2.src.models.model_SGACnet import SGACNet
    #     return SGACNet()
    # if cfg['model_name'] == 'catnet':
    #     from .models_transbackbone.CATNet.CatNet import CatNet
    #     return CatNet(41)
    # # if cfg['model_name'] == 'catnet':
    # #     from .models.CATNet.CatNet import CatNet
    # #     return CatNet(41)
    # if cfg['model_name'] == 'cmanet':
    #     from .models.CMANet.CMAnet import CMAnet
    #     return CMAnet(41)
    # if cfg['model_name'] == 'hainet':
    #     from .models_transbackbone.HAINet.HAI_models import HAIMNet_VGG
    #     return HAIMNet_VGG()
    # if cfg['model_name'] == 'cirnet':
    #     from .models.CIRNet.CIRNet_Res50 import CIRNet_R50
    #     return CIRNet_R50()
    # if cfg['model_name'] == 'icnet':
    #     from .models.ICNet.icnet import ICNet
    #     return ICNet(nclass=41)
    # if cfg['model_name'] == 'bbsnet':
    #     from .models.BBSNet.BBSNet_model import BBSNet
    #     return BBSNet()
    # if cfg['model_name'] == 'sagatenet':
    #     from .models.SAGATE1.network import DeepLab
    #     return DeepLab(41)
    # if cfg['model_name'] == 'rednet':
    #     from .models.rednet import RedNet
    #     return RedNet(num_classes=cfg['n_classes'], pretrained=True)
    # if cfg['model_name'] == 'testnet':
    #     from .models.encoderdecoder import EnDecoderModel
    #     return EnDecoderModel(n_classes=cfg['n_classes'],rgb_backbone='resnet34', pretrained=False)
    # if cfg['model_name'] == 'bbsnet':
    #     from .models.BBSnetmodel.BBSnet import BBSNet
    #     return BBSNet(n_class=cfg['n_classes'])



def get_inter_model(cfg):
    if cfg['model_name'] == 'inter':
        from sam.lsy.mymodelnew.toolbox.Mymodels.DINO.teDino import LoRADinoWithText
        return LoRADinoWithText(num_classes=41)

