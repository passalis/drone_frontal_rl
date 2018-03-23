import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import numpy as np
from keras.optimizers import RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl_utils.rl_model import define_model
from rl_utils.CameraEnviroment import CameraControlEnv


def evaluate_model(seed=12345, model_path=None, interactive=False):
    np.random.seed(seed)
    model = define_model()
    memory = SequentialMemory(limit=500, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=.05, nb_steps=50000)
    dqn = DQNAgent(model=model, nb_actions=5, policy=policy, memory=memory, processor=None,
                   nb_steps_warmup=500, gamma=0.95, delta_clip=1.0, target_model_update=500, batch_size=32)
    dqn.compile(RMSprop(lr=.00005), metrics=['mae'])
    if model_path is not None:
        dqn.load_weights(model_path)

    # Train Evaluation
    env = CameraControlEnv(dataset_pickle_path='data/dataset.pickle', testing=False, interactive=interactive)
    env.seed(seed)
    res = dqn.test(env, nb_episodes=500, nb_max_episode_steps=20, verbose=0, visualize=True)
    train_mean_reward = np.mean(res.history['episode_reward'])
    before_train_tilt_error = np.mean(np.abs(env.init_tilt_error))
    before_train_pan_error = np.mean(np.abs(env.init_pan_error))
    after_train_tilt_error = np.mean(np.abs(env.final_tilt_error))
    after_train_pan_error = np.mean(np.abs(env.final_pan_error))
    print("Training evaluation: ")
    print("Mean reward: ", train_mean_reward)
    print("Tilt: ", before_train_tilt_error, " -> ", after_train_tilt_error)
    print("Pan: ", before_train_pan_error, " -> ", after_train_pan_error)

    # Test Evaluation
    env = CameraControlEnv(dataset_pickle_path='data/dataset.pickle', testing=True, interactive=interactive)
    env.seed(seed)
    res = dqn.test(env, nb_episodes=500, nb_max_episode_steps=20, verbose=0, visualize=True)
    train_mean_reward = np.mean(res.history['episode_reward'])
    before_train_tilt_error = np.mean(np.abs(env.init_tilt_error))
    before_train_pan_error = np.mean(np.abs(env.init_pan_error))
    after_train_tilt_error = np.mean(np.abs(env.final_tilt_error))
    after_train_pan_error = np.mean(np.abs(env.final_pan_error))
    print("Testing evaluation: ")
    print("Mean reward: ", train_mean_reward)
    print("Tilt: ", before_train_tilt_error, " -> ", after_train_tilt_error)
    print("Pan: ", before_train_pan_error, " -> ", after_train_pan_error)


if __name__ == '__main__':
    # Evaluate the trained agent
    evaluate_model(model_path='models/final.model')

    # You can also run the agent in interactive mode (select the 'View' window and press any button to continue')
    evaluate_model(model_path='models/final.model', interactive=True)
