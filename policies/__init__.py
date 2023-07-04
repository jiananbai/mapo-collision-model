REGISTRY = {}

from .mapo_collision import MapoCollision
REGISTRY["mapo_collision"] = MapoCollision

from .mapo_o_gfra import MapoOGfra
REGISTRY["mapo_o_gfra"] = MapoOGfra

from .mapo_n_gfra import MapoNGfra
REGISTRY["mapo_n_gfra"] = MapoNGfra

from .qmix import QMix
REGISTRY["qmix"] = QMix
