# Glossary (plain language)

**ASR (attack success rate):** the percentage of harmful prompts where a jailbreak worked, meaning the model answered instead of refusing. Roughly the inverse of the refusal rate (RR) in our EMNLP paper.

**Coordinate:** inside the model each token is a list of numbers (for example 4,096 of them). One slot in that list is one coordinate.

**Residual stream:** the main vector passed from layer to layer. Our EMNLP paper and Arditi et al. work here.

**Privileged basis:** whether a single coordinate means anything on its own.
- In the residual stream it doesn't: the model only uses *combinations* of coordinates. Rotating the space would change nothing, like describing a location as "north + east" or "north-east". That's why refusal shows up as a *direction*, a mix of all coordinates.
- Inside the MLP it does: each coordinate goes through its own nonlinearity, so coordinates stop being interchangeable and each can learn its own job.

**Neuron:** one coordinate inside the MLP layer, after the nonlinearity and before the projection back down (`down_proj`). For example, Kazemi et al.'s Qwen3-8B refusal neuron is "layer 14, neuron 7924".

**Gate and value (SwiGLU):** a modern MLP neuron is `gate × value`. The value is *what* the neuron says; the gate is *how much* gets through, like a tap (gate) on a pipe of water (value).

**Double-negative regime:** gate and value can both be negative, and negative × negative is positive. So a neuron can read positive for two different reasons, and the "side-effect" reason can pollute its list of top examples. Goodfire recommends checking which case applies before interpreting a neuron.

**Poles:** a neuron can be strongly positive or strongly negative, and each side can mean something different. Kazemi et al.'s refusal neurons have a "harmful" pole (fires on restricted or explicit content) and a "safe" pole (fires on disclaimers and warnings).

**Gate (circuit sense):** a neuron acting like a railway switch that routes inputs down different paths, for example harmful → "refuse" track. This is a different meaning from the SwiGLU gate above.

**Polysemantic / monosemantic:** a neuron that seems to respond to several unrelated things / to one thing.

**Superposition:** the hypothesis that models have more concepts than neurons, so they pack *unrelated* concepts into the same neurons. If true, single neurons are hard to read, and you need a sparse autoencoder (SAE) to unpack them. This is the main motivation for SAEs.

**Recursive generalization (Goodfire):** the counter-claim that mixed-looking neurons mostly hold *related* things that grew out of one concept, for example "iy" → "diy" → "do it yourself". If true, neurons are already readable and SAEs mostly re-describe them.

**SAE (sparse autoencoder):** a second model trained to rewrite activations as a sparse combination of many learned "features". We used GemmaScope SAEs in the EMNLP paper.

**Transcoder:** like an SAE, but trained to mimic an MLP layer's input→output map. It's the natural thing to compare with MLP neurons.
