# Next Steps

The immediate priority for the next phase of this project is developing the **automatic model selector**. Establishing this capability is central to determining whether this project can be classified as a true "framework" rather than a set of standalone models.

### Implementation Plan
1.  **Build the Complexity Check:** Develop a script that calculates dataset variance mathematically. A primary approach will involve evaluating how much variance is explained by the first few principal components on the raw feature matrix (without utilizing any ground-truth labels).
2.  **Dataset Testing:** Deploy this variance-based selector against a third, unseen Loghub dataset (e.g., Thunderbird).
3.  **Validation:** Allow the selector to route the data to either the PCA baseline or the Deep Learning Autoencoder. Only after the routing decision is made will we calculate the true F1 scores to verify if the selector made the correct architectural choice.
