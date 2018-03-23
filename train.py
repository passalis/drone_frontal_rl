import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import numpy as np
from keras.optimizers import RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl_utils.rl_model import define_model
from rl_utils.CameraEnviroment import CameraControlEnv

def train_model(seed=1):
    np.random.seed(seed)

    env = CameraControlEnv()
    env.seed(seed)

    model = define_model()

    memory = SequentialMemory(limit=500, window_length=1)

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=0.1, value_test=0.05,
                                  nb_steps=900000)

    dqn = DQNAgent(model=model, nb_actions=5, policy=policy, memory=memory, processor=None,
                   nb_steps_warmup=500, gamma=0.95, delta_clip=1, target_model_update=500, batch_size=32)

    dqn.compile(RMSprop(lr=.00005), metrics=['mae'])

    log_filename = 'results/camera_control_log.json'
    model_checkpoint_filename = 'results/rl_cnn_weights_{step}.model'
    callbacks = [ModelIntervalCheckpoint(model_checkpoint_filename, interval=10000)]
    callbacks += [FileLogger(log_filename, interval=1)]

    dqn.fit(env, nb_steps=800000, nb_max_episode_steps=50, verbose=2, visualize=False, log_interval=1,
            callbacks=callbacks)

    # After training is done, save the final weights.
    model_filename = 'models/rl_cnn.model'
    dqn.save_weights(model_filename, overwrite=True)


if __name__ == '__main__':
    train_model()
