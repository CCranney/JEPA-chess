# JEPA-chess

## Autonomous Machine Intelligence - World Models

In 2022, Yann LeCun published a paper outlining "[A path towards autonomous machine intelligence.](https://openreview.net/pdf?id=BZ5a1r-kVsf)" Here LeCun outlined some of the pitfalls of modern AI development that serve as obstacles to developing true, common-sense intelligence. He further suggests various methods for overcoming those obstacles in future development. Much of the paper is theoretical, indicating areas of intelligence that have little in the way of current functional machine learning models. However, LeCun proposes and outlines a specific model architecture that could address one of the largest deficiencies in common-sense intelligence, the world model. The architecture is the Joint Embedding Predictive Architure, or JEPA. Its purpose is to create an abstract representation of the world that can serve as a reference for safe experimentation space outside the real world, much like how one can simulate and refine a sentence in their thoughts before actually speaking it out loud.

Current development of JEPA-based architectures is skanty on GitHub. Several recent papers, [I-JEPA](https://arxiv.org/abs/2301.08243) and [V-JEPA](https://openreview.net/pdf?id=WFYbBOEOtv), have outlined specific use cases for JEPA in encoding images and videos, respectively. [An earlier paper](https://arxiv.org/abs/2211.10831) published about the same time as LeCun's original introduction outlined several experimentation efforts for introducing the impact of actions on the abstract state of an image. These are promising starts, but more development is required.

## Chess

This repository is an experimental effort to design a JEPA architecture that captures a world model of chess. To be clear, the initial development efforts will _NOT_ be focused on making a model that _plays_ chess. LeCun makes clear that the world model itself is not a form of reinforcement learning. It does not produce actions, but rather serves as a reference for considering actions. JEPA is designed to learn in an non-contrastive and non-generative format - in short, it learns through observation.

But why chess? To be frank, because technically the concepts of autonomous machine intelligence outlined by LeCun can all be addressed through such a game. One can perceive the game, predict future states in reaction to specific moves, and make moves based on those future states (perceiver, world model, actions). This could theoretically scale up to a form of abstract thought. One could also develop a sort of strategist model that considers the impact specific moves will have on some immediate or future goals (costs). Past moves, future predictions, and plans can be stored and referenced as the game progresses (short-term memory). If such a model could be put together, one could theoretically begin development on a configurator model, least understood of them all, that orchestrates the others toward achieving optimal performance.

These are lofty ambitions, a dream of future development. Chess was chosen because incremental developments could add up to such a goal over time. At present, however, we are starting at ground zero. Not only has little development been made on applying world models in such a fashion, the end goal is not directly lucrative. We already have models that handily beat professional chess players. This is a dream project, a passion project, for those who read LeCun's paper and felt there was something to be chased here.

## Project Structure

This project will be designed with human readability in mind. The intent is to attract and engage with collaborators interested in experimenting with the JEPA architecture as a world model for simulating action states, and as such must be readable to interested parties. 

### Abstract Classes

To enable this requirement, a second repository will be developed in tandem with this one, the [Jepa-Abstract-Architecture repository](https://github.com/CCranney/JEPA-Abstract-Architecture). The idea is to require a blueprint of key classes of the architectures or strategies employed. Hopefully, this second repository can serve as a backbone for future JEPA development outside of the present chess use case. Development collaborations and ideas for both repositories are welcome. A basic re-implementation of the [JEPA_SSL_NeurIPS_2022 repository](https://github.com/vladisai/JEPA_SSL_NeurIPS_2022) was done using these abstract classes, both as a test case in preparation for this project and to help remember specific implementation details as reference. It is called [JEPA-dot-tracking](https://github.com/CCranney/JEPA-dot-tracking).

### Discussion Board

The current plan is to use [the discussion board](https://github.com/CCranney/JEPA-chess/discussions) to outline ideas or strategies for implementation. This can be adjusted in the future.



## Conclusion

All discussions are welcome. Let's work some magic :)
