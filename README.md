# Bridging the Sim to Real Gap with Domain Randomization

This repository contains the code for the Reinforcement Learning project in the MLDL course at PoliTO.


Student: Pranav Tripathi 

## REINFORCE
*   `reinforce/agent_reinforce.py` Agent for REINFORCE without a baseline and REINFORCE with baseline = 20
*   `reinforce/task2_3_train.py` to train the agent
*   `reinforce/task2_3_test.py` to test the agent

## Actor-Critic
*   `actor_critic/agent_ac.py` Agent for Actor-Critic
*   `actor_critic/task2_3_train.py` to train the agent
*   `actor_critic/task2_3_test.py` to test the agent

## SAC (Soft Actor-Critic)
*   `sac/task4.py` to train and test SAC in the source env with default parameters
*   `sac/task5_tuning.py` to tune SAC hyperparameters using **Optuna** (optimizes for Target environment performance)
*   `sac/task5_train.py` to train and test SAC in Source -> Source, Source -> Target, Target -> Target with the hyperparameters found through tuning
*   `sac/task_test.py` to test a saved SAC model

## Uniform Domain Randomization (UDR)
*   `udr/task6_train.py` to train and test UDR in Source -> Target (and Source -> Source)
*   `udr/task6_test.py` to test a saved UDR model

## SimOpt (Simulation Optimization)
*   `simopt/simopt.py` functions to define SimOpt
*   `simopt/train_simopt.py` to train and test SimOpt in Source -> Target
*   `simopt/test_simopt.py` to test a saved SimOpt model
