# Deep Reinforcement Learning for Frontal View Person Shooting

![alt text](https://github.com/passalis/drone_frontal_rl/blob/master/control.png "Camera control example")


This repository contains an implementation of the Deep RL method proposed in *Deep Reinforcement Learning for Frontal View
Person Shooting using Drones*. The following are supplied:
1. An [OpenAI gym](https://gym.openai.com/)-compliant environment that can be directly used with [keras-rl](https://github.com/keras-rl/keras-rl). This environment uses the [HPID dataset](http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html)  to simulate the camera control commands (up, down, left, right, stay). 
2. Code to train and evaluate an RL-agent that is trained to control the camera to perform frontal shooting.
3. A [pre-trained model](https://github.com/passalis/drone_frontal_rl/blob/master/models/final.model).

To run the code:



0. Install the required dependencies (Python 3.6 was used for training/testing the models):
```
pip3 install tensorflow-gpu keras keras-rl gym
```
Also install the python bindings for [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html) (or replace the calls to openCV library if you do not want to use OpenCV).
1. Download the [HPID dataset](http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html) to 'data/datasets'.
2. Run the [preprocess_dataset.py](https://github.com/passalis/drone_frontal_rl/blob/master/preprocess_dataset.py) script to create the dataset pickle.
3. (Optionally) train the model by running the [train.py](https://github.com/passalis/drone_frontal_rl/blob/master/train.py) script.
4. Evaluate the model by running the [evaluate.py](https://github.com/passalis/drone_frontal_rl/blob/master/evaluate.py) script.

Note that the evaluation function also supports interactive evaluation. This allows for more easily examining the behavior of the agent.

If you use this code in your work please cite the following paper:

<pre>
@inproceedings{frontal-rl,
  title       = "Deep Reinforcement Learning for Frontal View Person Shooting using Drones",
	author      = "Passalis, Nikolaos and Tefas, Anastasios",
	booktitle   = "Proceedings of the IEEE Conference on Evolving and Adaptive Intelligent Systems (to appear)",
	year        = "2018"
}
</pre>


Also, check my [website](http://users.auth.gr/passalis) for more projects and stuff!


This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 731667 (MULTIDRONE).

