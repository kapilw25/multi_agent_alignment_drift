
```
Topic1: 
Latent Collab in Multi Agent Systems
-- If Multi - architechture (Llama, Phi, etc) LLMs are working in same environment , their alignment space will be different
-- Our goal: to resolve the possibility of alignment [safe vs Unsafe] drift

-- for the given 5 agents, let's say, best AQI is best (top score) for Llama, then Llama becomes the baseline
then and make the remaining LLMs align equally as Llama using steering
--
Look for literature
1) Game theoretic equilibrium
2)  [Multi Agent] Alignment benchmark
3) Purdue paper - Anthropic dataset >> take 7 dimension >> check if there is any conflict of direction
```

## *Do NOT modify above raw notes*

```
1) Project 1: Latent Alignment for Multi Agent Collaboration

- In multi agent systems, esp. when models from different families collaborate, agents may exhibit different alignment behaviors. Even if individual agents appear aligned in isolation, collaboration can induce alignment drift, leading to conflicts or contradictions.

- Inspired by latent collaboration in multi agent systems paper, this project investigates how to bring multiple agents into the same alignment zone before collaboration, in order to avoid alignment leakage and increase consensus.

1.1) Problem:
- Agents come from different model families with different alignment characteristics.
- Output level agreement is insufficient to ensure shared alignment.
- When agents work together, misalignment may surface across tasks or interactions.
- Central question: How can we align multiple agents in latent space so they operate within a shared alignment region during collaboration?

1.2) Our Approach (Latent space + steering + AQI):

1.2.1) Selecting a Baseline Model: 
- Compute AQI for all models. 
- The model with the highest AQI is chosen as the baseline alignment reference.

1.2.2) Latent Alignment via Steering: 
- Other models are steered in latent space toward the baseline model’s alignment region and see if they really align without the model breaking.
- Steering is performed at the hidden state level.
- Different steering formulations are explored (as part of the project)
- After steering, we obtain: Baseline model & Multiple agents steered into the baseline alignment space

1.2.3) Multi Agent Interaction:
- The now steered agents and the baseline agent are placed into a game theoretic setup.
- Agents are given equal participation (eg: symmetric or Stackelberg like interactions).

1.3) Evaluation: We will have to look for multi agents alignment benchmark, if none, we make one of our own.

1.4) Goal: The goal is to ensure that cooperating agents do not exhibit alignment related conflicts, and instead converge toward consistent, non contradictory behavior during multi-agent interaction and eventually increase the overall alignment of the system. 
```
