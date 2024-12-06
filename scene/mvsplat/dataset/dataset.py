from dataclasses import dataclass

# from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    # image_shape: list[int]
    # background_color: list[float]
    # cameras_are_circular: bool
    # overfit_to_scene: str | None
    # view_sampler: ViewSamplerCfg

    def __init__(self,image_shape,background_color,cameras_are_circular,overfit_to_scene,view_sampler):
        self.image_shape = image_shape
        self.background_color = background_color
        self.cameras_are_circular = cameras_are_circular
        self.overfit_to_scene = overfit_to_scene
        self.view_sampler = view_sampler

