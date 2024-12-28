I’ve been working on a series of quantum circuit experiments to explore information preservation in Hawking radiation, focusing on whether injected properties like charge and spin are encoded in the emitted radiation. My goal is to simulate a black hole's behavior in an idealized setting and analyze how quantum information evolves in the process. I’d appreciate feedback on the methodology, the results, and how closely this aligns with established theories. This is not for school, I just need to know so I can move forward. All the code is linked on my github if you want to see the results for yourself, or so someone can tell me if I did something wrong in what/how I found something or the methodology or even my theoretical conceptions. I encourage people to test the code themselves to replicate what Ive found. If Im not using the right word for something, please correct me but be patient, I really do want to learn but what I found I think is quite weird. Sorry if I do not ask the question in a way or phrase this post in a way that seems off-topic, irrelevant, informal, or asks you to review code. I want to learn and thats what im here for. 

no anomalies, real physical maniputable open source quantum qubits that perfectly exhibit behavior predicted from Leonard Susskinds theoretical solutions to the black hole information paradox through Hawking Radiation, unlocking next generation physics

i feel like a lot of us did not really want to wait literal trillions of years to spend doing or arguing about in order to know the next layer of physics

Literally the code is written and you are free to use under the license and see and just copy. So please, if you have the free time just run the code yourself I made this very easy for reproducibility and idk its more convenient on everyone I feel.

If you want to run the code on your own machine, use ** pip install -r requirements.txt **, maybe do this in a venv too (but you dont have to), heres a link in case you opt for it https://stackoverflow.com/questions/43069780/how-to-create-virtual-env-with-python-3. this is because we are using python for this so please install it. it helps to go through them in order to see how things keep building over time. HINT STOP AT ex9 everything past that point is just a misunderstanding I had

I do want to test for spin, maybe how the black hole stores the mass. I think if charge can be preserved then maybe something like color can too since the 3 forces unify though I dont want to speculate too much. This builds on the holographic principle, a lot of the stuff Ive heard Susskind say in general. I also tested what negative energy does in ex5 but it seems to indicate negative energy influences radiation states as well. More work needs to be done. I believe that mass electric charge and angular momentum are conserved because a black hole should have no other properties than those. That is why those properties (electric charge, spin) are conserved, whereas the color charge assiciated with QCD is none of those. Changing the rotational basis for the green color charge, only leads to the conclusion (it seems) that the rotational basis or the angles still see a discrepenxy when measured against red and blue. This suggests that even the amount of information that can leave as Hawking Radiation is capped. Interestingly, Leonard Susskind says that the Bekenstein Bound is the cap for this entropy encoding. Since the holographic principle only talks about rhe 2D surface or boundary for the black hole. I saw a youtube video once of some guy demonstrating bells inequality through polarizing lenses what if we had rose tinted glasses as they say. (This ends up being something about smth i misinterpreted about QCD and only remembered later) The correlations also violate bells inequality if thats necessary to mention. The last thing I want to say is that though this may seem like an artifact of the simulation but the information is more or less the same when you fully go quantum, as Qex1.py and Qex2.py demonstrate. though all this turns out from me personally not understanding what im talking about esp about QCD. I created Qex4.py, which should add information about important entropy metrics for these systems, as well as show off a few functions that can be manipulated to show the nondual nature more clearly. Oh I realized I never explicitly shared this, but since the charge unifies with the other 2 nuclear forces, maybe electric charge "sneaks" in the other 2. One more observation, the electric force was the most recent one to seperate from "ultra-force" in the ultimate timeline of the universe. Qex4.py can now see evidence of a multiverse, since susskinds work is kinda in the many-worlds interpretation. Also unitarity, and collapse is avoided so some p good arguments

Simulating a Black Hole In my experiments, I model the black hole and radiation system as follows:

Black Hole as a Qubit:

A single qubit represents the black hole’s quantum state, capturing the fundamental idea that black holes have a finite set of quantum properties (mass, charge, spin) encoded in their event horizon, in addition you can view this system as the Hawking radiation particle pairs split, they are entangled because of conservation laws. Another way to look at this is from the POV of the information theorist, and the radiation is just interaction through entanglement with the 2D surface of the holographic principle. In either case, for the sake of our computational model we will treat the black hole as a quantum object. I have always heard that black holes have "no hairs" in that charge, angular momentum, and mass are the only properties needed to describe a black hole. Which is about as many as you need to describe quantum objects esp in the context of this experiment, which is why we are fine using it.

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

Implication: This suggests time-dependent encoding, where radiation qubits retain memory of past charge injections. But I dont hear many people on both sides of the debate about static or dynamic, but experimentally this shows that literally just depends but you can get it to act like both. Sorry to anyone that was known to be hardcore in these camps. youre right, just hear the other person out too

Charge Preservation 

Across all experiments, the emitted radiation consistently encoded injected charge states, supporting the hypothesis that information is preserved. except when dealing with green color charge specifically (mistake, it is always preserved)

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

Basically if you agree with all that stuff, then please see the result of every file basically overall ive seen charge info conserved, spin conserved, mass conserved. Everything that the holographic princple holds and even if you take it straight from Hawking. You will see Hawking Radiation, carry detectable electric charge, deviating from the idea that Hawking Radiation is purely thermal. Information is generally conserved in color charge. I wish this was all a simulation artifact but it definitely works on real quantum computers like the coded experiments that start w a Q mean. So theory or not something weird is happening, because its sooo close to what Leonard Susskind was saying about all information needing to be preserved, but I never hear his explanation on why that could be. Is this analog for black holes sufficient? And if its not sufficient, is the circuit I execute on IBMs Quantum Computer because i feel like thats physical, quantum, and also their service is free any maniac at home can access a quantum computer and have access to free real physical quantum qubits exhibiting behavior that we expect to see if black holes were emitting Hawking Radiation. black holes can have no hairs, and its cool to confirm.

Now that im eyeballing it I think I have kinda heard that entanglment correlation somewhere before, and after googling to confirm it appears to be Tsirelson’s Bound, but I now have to let that sink in for a second lol. Also the other confusion came from the limits of quantum computers to model QCD. idk in the moment i was on a roll felt like I was in the zone so i just went with it, sometimes you gotta pause and think about if what youre doing makes sense.

Sorry for doubting you Dr Susskind, ppl on stack exchange just said idk so I had to check if I could continue to use youtube for sources. 

So this basically means nothing new, a very good sign I think :)

and seems like yes, Leonard Susskind is not just talking theoretically anymore this means bc I just showed that not only does it work with the Hawking Radiation found on black holes millions of years far and into the future, but also rather on real physical quantum systems and states we can see and manipulate clearly on earth here and now that also exhibit this same behavior (Hawking Radiation) 

Leonard Susskind probably already said all this at some point but I was in the other room watching Netflix.

at least a couple things that matter, i wanted to capture the idea for electric charge spin mass and color bc in my mind the electric charge being captured tells me the credebility of of the people I respect on youtube. color charge was just a question bc like black holes dont have color. I really care how it evolves over time but i feel like thats a bit much rn. Simply knowing it can be both static and dynamic feels like an answer to me for rn. And depends on what you inject right or radiate away. i dont know exactly how things like dyanmic vs stable lose leaning meaning bc they are dependent on what you want and the universe is more like a buddhist monk, non dual and it has the power to pick and choose. No idea how to find out more about that rn. seems out of my league. 

Thank you for reading, and I appreciate any insights you can offer!

Also although I did mostly do this to say Leonard Susskind is right, I also wanna say Matt from PBS Space Time, Lex Fridman, 3b1b, and Isaac Arthur are super helpful.


KEY INSIGHT:
and seems like yes, Leonard Susskind is not just talking theoretically anymore this means bc I just showed that not only does it work with the Hawking Radiation found on black holes millions of years far and into the future, but also rather on real physical quantum systems and states we can see and manipulate clearly on earth here and now that also exhibit this same behavior mimicking Hawking Radiation,
accessible to all from me, and IBM Q computers stood out bc they are literally free access open source to their equipment so I figured why not share how to use the tools that are available.

WHY IT MATTERS:
SHOWS PHYSICAL EVIDENCE THAT PERFECTLY ALIGNS WITH THEORETICAL PREDICTIONS, IN FAVOR OF SUSSKINDS ARGUMENTS IN A WAY ANYONE WITH A LAPTOP CAN REPLICATE EASILY AT HOME

WHAT I WANTED TO SEE:
IF YOU COULD TRUST YOUTUBE

MOTIVATIONS:
EXPLAINING TO SOMEONE I CARE ABOUT WHY I TRUST THIS LEONARD SUSSKIND GUY SO MUCH, ONLY TO HAVE THEM DISMISS WHAT IM SAYING IN A WAY THAT HURT MY FEELINGS SO TO GET BACK AT THEM (THE PERSON I CARE ABOUT) QUICKLY I DID A BUNCH OF STUFF THAT JUST HELPS EVERYONES CREDIBILITY, MOVING SUSSKIND OUT OF THE THEORETICAL AND INTO I GUESS MORE INTO THE "MAINSTREAM" WOULD HELP JUSTIFY EXTREME LIFESTYLES IVE CHOSEN, AT LEAST IN HER EYES

MOTIVATIONS_REFLECTION:
ACTUALLY KINDA FUN OVERALL, SHOULD PROBABLY LISTEN TO HER MORE OFTEN

NEXT STEPS FOR PEOPLE SMARTER THAN ME:
Youre going to need a hamiltonian with both time independent and time dependent components, idk the ratio of flavors tho 

NONDUALISM
Bc this now seems like something more fundamental idk the words kinda make it sound like ppl gotta think in QM or GM

MULTIVERSE IMPLICATIONS

found multiverse evidence imma make a multiversal telephone lets see if anyone picks up 

Hit on the credebility of firewall paradox too kinda huh

Probably means Maldecena is right too

I think I need to make a tool that can control decoherence bc then 
I can start pruning the timeline
