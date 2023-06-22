Explainable Reinforcement Learning: A Hierarchical Planning Approach
======================================

This is based on the official implementation of the [Director][project] algorithm in TensorFlow 2.

[project]: https://danijar.com/director/

![Director Internal Goals](https://github.com/danijar/director/raw/main/media/header.gif)

Running the Agent
-----------------

Follow the instructions below.

Install dependencies:

```sh
pip install -r requirements.txt
```

Train agent:

```sh
python embodied/agents/director/train.py \
  --logdir ~/logdir/$(date +%Y%m%d-%H%M%S) \
  --configs dmc_vision \
  --task dmc_swingup
```

See `agents/director/configs.yaml` for available flags and
`embodied/envs/__init__.py` for available envs.

Some of the Results of Our Work
-----------------------

Our changes enable Director to perform well on DM Control Suite - SwingUp task. 

Left: Director, Right: Our agent

<img src="https://github.com/jdubkim/explainable-hierarchical-planning/blob/main/data/pendulum/director.gif" width="200" height="100"/> <img src="https://github.com/jdubkim/explainable-hierarchical-planning/blob/main/data/pendulum/regularised.gif" width="200" height="100"/>

The below is a visualisation of internal goals. 

![pendulum-swingup](https://github.com/jdubkim/explainable-hierarchical-planning/blob/main/data/pendulum/pendulum_viz.png)

How does Director work?
-----------------------

Director is a practical and robust algorithm for hierarchical reinforcement
learning. To solve long horizon tasks end-to-end from sparse rewards, Director
learns to break down tasks into internal subgoals. Its manager policy selects
subgoals that trade off exploratory and extrinsic value, its worker policy
learns to achieve the goals through low-level actions. Both policies are
trained from imagined trajectories predicted by a learned world model. To
support the manager in choosing realistic goals, a goal autoencoder compresses
and quantizes previously encountered representations. The manager chooses its
goals in this compact space. All components are trained concurrently.

![Director Method Diagram](https://github.com/danijar/director/raw/main/media/method.png)

For more information:

- [Google AI Blog](https://ai.googleblog.com/2022/07/deep-hierarchical-planning-from-pixels.html)
- [Project website](https://danijar.com/project/director/)
- [Research paper](https://arxiv.org/pdf/2206.04114.pdf)


