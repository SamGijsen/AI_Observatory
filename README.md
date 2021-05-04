# AI_Observatory


## Surprise Minimization
Minimize _-ln p(o)_

As preferences are modelled via (higher) prior probabilities, maximizing preferences corresponds to surprise minimization for both perception and action.
(It may also be formulated to encode preferences for states!)

Planning as inference: Bring preferences and policy values within the domain of Bayesian belief updating.

Average surprisal (i.e., self-information) in IT is known as entropy. Minimizing surpisal thus also minimizes the entropy of sensory outcomes (which can be thought of as homeostatis - i.e., minimizing variability within internally sensed bodily states, thus keeping them within homeostatic ranges).
Surpisal can also be viewed as the negative log marginal likelihood or model evidence. Here, minimizng surpisal is maximizing the evidence for one's model of the world.

_VFE_ serves as an approximation to the marginal likelihood _p(o)_, functioning as an upper bound.

## Exploration-exploitation
Exploring and exploiting are here just two aspects of expected free energy, and which behavior is favored in a given situation depends on current levels of uncetainty and the level of expected reward. This behaviour, i.e. a posterior over policies, is informed by both _VFE_ and _EFE_. These quantities respectively furnish retrospective and prospective action policy evaluations.

_G(pi) = KL[ q(o|pi) || p(o) ] + E_q(s|pi)[ H[ p(o|s) ]]_

The
