# package info
__author__ = "Joao Moura"
__copyright__ = "Copyright 2021, Joao Moura"

__license__ = "Apache v2.0"
__version__ = "0.0.1"

# ----------------------------------------------------------------
# import from utils folder
# ----------------------------------------------------------------

from sliding_pack.utils import intfunc as integral

from sliding_pack.utils import plots

from sliding_pack.utils import gen_nom_traj as traj

# ----------------------------------------------------------------
# import from sliding_pack folder
# ----------------------------------------------------------------

from sliding_pack.sliding_pack import dynamic_model as dyn

from sliding_pack.sliding_pack import dynamic_model_mc as dyn_mc

from sliding_pack.sliding_pack import dynamic_model_test as dyn_test

from sliding_pack.sliding_pack import classes4opt as opt

from sliding_pack.sliding_pack import pusher_slider_opt as to

from sliding_pack.sliding_pack import double_slider_opt as db_to

from sliding_pack.sliding_pack import augmentation as aug

# ----------------------------------------------------------------
# import from simulation folder
# ----------------------------------------------------------------

from sliding_pack.simulation import double_sliders as db_sim



def load_config(filename):
    import os
    import yaml
    import pathlib
    path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'config',
        filename
    )
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
