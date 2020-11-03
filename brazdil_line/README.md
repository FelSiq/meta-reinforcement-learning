# Meta-learning + Reinforcement Learning
This repository hold code related to experiments with meta-learning and reinforcement learning. Note that the "meta-learning" definition followed here follows [[1]](#1) and [[2]](#2), which means it is meta-knowledge based and do not use any Recurrent Neural Networks.

# Sub-directories descriptions:
1. algs: keep the reinforcement learning algorithms to create base-learners.
2. metadata: store all metadata produced in the experiments.
3. test_envs: store code related to the Gridworld environment.

# Modules descriptions:
1. analytics: plot graphs related to the gathered meta-data such as class distributions.
2. build_metamodel: use the gathered meta-data to create meta-models using [XGBoost algorithm](https://xgboost.readthedocs.io/en/latest/).
3. display_env: plot some Gridworld environment samples while changing some hyper-parameters related to the sample generation. Useful to study the effects of certain hyper-parameters while holding the others fixed.
4. feat_imp_plot: plot the feature importance of the collected meta-characteristics. Also uses a model from XGBoost algorithm, and only has analysis for a specific scenario.
5. get_metadata: collect N samples of metadata using user custom arguments.
6. metafeatures: holds the implementation of meta-feature extraction functions.
7. noise_injection: analysis of the effects in the meta-model predictive power while injecting more and more random gaussian noise in the meta-features.
8. test_RL_ALGORITHM: there is a test script for every base model algorithm. Use those to run the corresponding algorithm while checking the progress in real time if you wish.
9. utils: keeps useful functions for all other modules.

# References
<a id="1">[1]</a> 
VILALTA, R.; GIRAUD-CARRIER, C.; BRAZDIL, P.; SOARES, C. Using meta-learningto support data mining.International Journal of Computer Science & Applications,v. 1, 01 2004.

<a id="2">[2]</a> 
[2] BRAZDIL, P.; GIRAUD-CARRIER, C.; SOARES, C.; VILALTA, R.Metalearning - Appli-cations to Data Mining.[S.l.: s.n.], 2009. ISBN 978-3-540-73262-4.
