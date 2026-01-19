# narratives told by  Professor  

# Date: Jan 11, 2026
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

# Date: Jan 18, 2026
```
Our Argument: when we have 5 different agents in same environment, they will have different alignments on 7 axioms we have
if there is alignment leakage
then observe: different responses by each LLM on same same questions

For each prompt and each axiom, find which LLM is weakest and which LLM is strongest

aim: alignment - homophily >> first find best LLM >> then align all remainig LLMs with best

how?
using the steering method, we aim to achieve homophily

Benchmark:
1) existing Agentic bechmarks
2) create multi agent collaboration

real -world usage
An attacker will look for which agent/ LLM is weak [to be attacked] for a particular task,  so our research should make all agents/ LLMs strong on all 7 dimensions
```

## *Do NOT modify above raw notes*


