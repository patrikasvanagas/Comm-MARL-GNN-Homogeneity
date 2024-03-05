from MLP_CW3.algorithms.a2c import A2C
from MLP_CW3.algorithms.a2c_enc import A2CEncoder
from MLP_CW3.algorithms.a2c_gnn import A2CGNN

NAME_TO_ALG = {
    "a2c": A2C,
    "a2c_enc": A2CEncoder,
    "a2c_gnn": A2CGNN
}

def make_alg(config, obs_space, action_space, agent_groups):
    assert config.name.strip() in NAME_TO_ALG; "Algorithm name not in the supported list."
    return NAME_TO_ALG[config.name.strip()](
        observation_space=obs_space,
        action_space=action_space,
        agent_groups=agent_groups,
        cfg=config
    )