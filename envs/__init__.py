REGISTRY = {}

from .collision import CollisionModel
REGISTRY["collision"] = CollisionModel

from .o_gfra import OGfra
REGISTRY["o_gfra"] = OGfra

from .n_gfra import NGfra
REGISTRY["n_gfra"] = NGfra

