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

_\Gamma_ (scalar) modulates the degree to which G controls policy selection. It encodes a prior belief about the confidence with which policies can be inferred (i.e., how reliable beliefs about the best policy are expected to be. It may be tought of as an expected free energy (G) precision. If no habits are present, lower _\Gamma_ values lead to more randomness in policy selection, while higher values increase the influence of habits on policy selection. At higher values, the agent is more confident that 

## Implementation
1) Variational message passing (mean-field approximation)
2) Belief propagation (Bethe)
3) Marginal message passing (compromise between efficiency of 1) and accuracy of 2))

Information  gain vs Log prior preferences / reward / utility
EFE: natural units for both intrinsic and extrinsic value
Knob on expected complexity vs expected accuracy or ambiguity

## Prediction errors
#### State prediction errors
These track how _F(pi)_ changes over time as beliefs about states _s(pi,tau)_ are updated (i.e., reductions in F correspond to reductions in state prediction error.

(0.5 * B_backwards * B_forwards) + A*o_t - s

#### Outcome prediction errors
These track how _G(pi)_ changes over time as beliefs about policies are updated (i.e., reductions in G correspond to reductions in outcome prediction error). That is to say, when this type of PE is minimized, policies are identified that minimize the expected difference between predicted and preferred outcomes (and minimize ambiguity, which maximizes information gain).


## Questions

#### Why does a-novelty seem added instead of subtracted from G?
Using spm_softmax(vector), the larger (more positive) elements will yield larger probabilities. For G, we are computing state information gain, 'a' information gain, and prior preferences. Under this formulation, each term contributes positively to policy selection, and thus the more these terms positively contribute to G(k), the more likely policy k is to be selected. For prior preferences, this means the less negative the better, and for 'a'-information gain, the more positive the better.

## Study ideas
It is easy to end up with trials in which information gain and prior preferences are equal; do subjects actually maintain an equal selection probability over policies?

![afbeelding](https://user-images.githubusercontent.com/44772298/119142458-cfb2a300-ba46-11eb-8c6d-d94d370fdf7c.png)
