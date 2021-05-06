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

The expected free energy as a function of the policies (_pi_) is here written in terms of 
1) Risk/Expected complexity/beliefs about probability for reward. That is, the lower the KL, the higher the changes of attaining rewarding outcomes under that policy.
2) Expected value of the entropy of the likelihood function, with _H(p(o|s)) = SUM[ p(p|s) ln[p(o|s)] ]_. A higher-entropy likelihood means there are less precise predictions about outcomes given beliefs about the possible states of the world. This term is commonly refrered to as the measure of ambiguity (or expected inaccuracy). Policies that minimize ambiguity will try to occupy states that are expected to generate the most precise (i.e. most informative) observations. 
Putting the risk and ambiguity terms together means that minimizing EFE will drive selection of policies that maximize both reward and information gain.

Note also that EFE entails stronger (more precise) preferences for one outcome over others will have the effect of down-weighting the value of information, leading to reduced information seeking.

### Expected free energy precision
If there are no habits, lower $`\gamma_`$ values lead to more randomness in policy selection. In case of strong habits, lower values increase how much habits influence policy selection, because the influence of G is reduced relative to E. (See: _pi = sigma(ln E - F - \Gamma G))_
In case of deep policies, this parameter is updated after each observation based on whether _F(pi)_ and _G(pi)_ increase or decrease, leading to reduced or increased expected precision values, respectively. In shallow parameters the prior is used for each timestep (with a depth of 1).


```math
a^2+b^2=c^2
```
