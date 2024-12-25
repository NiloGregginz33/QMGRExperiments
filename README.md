I’ve been working on a series of quantum circuit experiments to explore information preservation in Hawking radiation, focusing on whether injected properties like charge and spin are encoded in the emitted radiation. My goal is to simulate a black hole's behavior in an idealized setting and analyze how quantum information evolves in the process. I’d appreciate feedback on the methodology, the results, and how closely this aligns with established theories. This is not for school, I just need to know so I can move forward. All the code is linked on my github if you want to see the results for yourself, or so someone can tell me if I did something wrong in what/how I found something or the methodology. I encourage people to test the code themselves to replicate what Ive found. If Im not using the right word for something, please correct me but be patient, I really do want to learn but what I found I think is quite weird. Sorry if I do not ask the question in a way or phrase this post in a way that seems off-topic, irrelevant, informal, or asks you to review code. I want to learn and thats what im here for. 

If you want to run the code on your own machine, use ** pip install -r requirements.txt ** . this is because we are using python for this so please install it. it helps to go through them in order to see how things keep building over time . 

I do want to test for spin, maybe how the black hole stores the mass. I think if charge can be preserved then maybe something like color can too since the 3 forces unify though I dont want to speculate too much. This builds on the holographic principle. I also tested what negative energy does in ex5 but it seems to indicate negative energy influences radiation states as well. More work needs to be done. I believe that mass electric charge and angular momentum are conserved because a black hole should have no other properties than those. That is why those properties (electric charge, spin) are conserved, whereas the color charge assiciated with QCD is none of those. The clear discrimination towards the green color charge implies violations of SU3 symmetry. Changing the rotational basis for the green color charge, only leads to the conclusion (it seems) that the rotational basis or the angles still see a discrepenxy when measured against red and blue. This suggests that even the amount of information that can leave as Hawking Radiation is capped. Interestingly, Leonard Susskind says that the Bekenstein Bound is the cap for this entropy encoding. Since the holographic principle only talks about rhe 2D surface or boundary for the black hole. I saw a youtube video once of some guy demonstrating bells inequality through polarizing lenses what if we had rose tinted glasses as they say. The correlations also violate bells inequality if thats necessary to mention. The last thing I want to say is that though this may seem like an artifact of the simulation but the information is more or less the same when you fully go quantum, as Qex1.py demonstrates.

Simulating a Black Hole In my experiments, I model the black hole and radiation system as follows:

Black Hole as a Qubit:

A single qubit represents the black hole’s quantum state, capturing the fundamental idea that black holes have a finite set of quantum properties (mass, charge, spin) encoded in their event horizon, in addition you can view this system as the Hawking radiation particle pairs split, they are entangled because of conservation laws. 

Radiation as Qubits:

Additional qubits represent the emitted Hawking radiation, which entangle with the black hole qubit as the system evolves. 

Evaporation Process:

Radiation qubits are sequentially entangled with the black hole qubit to mimic the emission of Hawking radiation over time. 

Charge Injection:

I simulate charge injections into the black hole using Pauli gates (X for positive charge, Z for negative charge) to analyze whether these properties are encoded in the radiation.

Limitations of the Simulation While the quantum circuit captures certain aspects of black hole physics, there are critical limitations:

No Spatial Geometry:

The simulation does not account for the curved spacetime geometry near a black hole, which plays a significant role in Hawking radiation. 
Information Encoding Simplified:

The black hole qubit directly interacts with the radiation qubits, but the real-world mechanism for encoding information in Hawking radiation remains an open question. 

No Event Horizon Constraints:

In reality, the event horizon prevents signals from escaping a black hole. This simulation treats the black hole as a quantum system without such spatial constraints.

Idealized Dynamics:

The system assumes perfect unitary evolution and ignores potential complications from decoherence or quantum gravity effects. Despite these simplifications, the model aims to test whether key principles—such as information preservation and entanglement—can be explored in a controlled, quantum mechanical context.

Experimental Goals The primary questions driving this work are:

Information Encoding: Does the radiation qubits’ state reflect the black hole’s quantum properties, such as injected charge or spin? 

Static vs. Dynamic Behavior: Does the black hole encode information globally in a time-independent manner (static encoding), or does the radiation evolve dynamically based on the timing of interactions? 

Insights into the Information Paradox: Can the results offer new insights into how black holes preserve information, consistent with unitarity? 

Methodology Here’s an overview of how the experiments are set up:

System Initialization:

The black hole qubit is initialized in a superposition state, entangled with radiation qubits via controlled gates (CX). 

Charge Injections:

I alternate between injecting positive (X gate) and negative (Z gate) charges into the black hole qubit, testing how these affect the radiation’s quantum state. 

Evaporation Simulation:

Radiation qubits are sequentially entangled with the black hole qubit to mimic the gradual release of Hawking radiation. 

Measurements:

Radiation qubits are measured to analyze how information is encoded. Entanglement entropy and statevector evolution are tracked to study the distribution of information. 

Results 

Experiment A: Static Behavior Measurement: The radiation collapsed into a single dominant state (0000) with 100% probability. 

Implication: This suggests limited entanglement or static encoding, where the radiation doesn’t reflect dynamic changes in the black hole’s state. 

Experiment B: Dynamic Behavior Measurement: Outcomes showed a diverse distribution (0000, 0101, 1111, 1010) with nearly equal probabilities. 

Implication: This suggests time-dependent encoding, where radiation qubits retain memory of past charge injections. 

Charge Preservation 

Across all experiments, the emitted radiation consistently encoded injected charge states, supporting the hypothesis that information is preserved. 

Questions for the Community

How well does this simulation capture key aspects of black hole physics?

Are the idealizations (e.g., no spatial geometry or event horizon) reasonable for exploring information preservation? Does the methodology align with theoretical models?

Are there improvements I could make to better simulate the entanglement and evaporation process? Static vs. Dynamic Encoding:

Do the results suggest new insights into whether information release is time-dependent or global? 

Implications for the Information Paradox:

Can these findings contribute to understanding how black holes preserve information, or do they merely confirm existing assumptions? 

Code https://github.com/NiloGregginz33/QMGRExperiments

Closing Notes I’ve done my best to set up this simulation based on theoretical insights, but I’m not a physicist by training.

I’d love feedback on:

Whether the methodology and results align with established physics, or if any of these experiments have been done before this on this topic. Any suggestions for refining the model or testing additional scenarios. 

Thank you for reading, and I appreciate any insights you can offer!
