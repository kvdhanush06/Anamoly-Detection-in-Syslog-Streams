# Weekly Progress Sheets

## Week 0 (July 28 - August 3)
**Tasks Completed:**
* Brainstormed potential capstone projects, identified log anomaly detection as a critical problem, and finalized the topic. 
* Outlined the initial tasks, objectives, and overall architecture for the project.

**Tasks Planned for next week:**
* Study base research papers and analyze existing log anomaly detection methods.
* Prepare initial structured datasets for baseline testing.

---

## Week 1 (August 4 - August 10)
**Tasks Completed:**
* Studied the base research paper and analyzed existing log anomaly detection methods.
* Downloaded and prepared the structured HDFS dataset for initial testing.

**Tasks Planned for next week:**
* Write the code for the lightweight baseline models (PCA and RPCA).
* Run the PCA model on the HDFS dataset.

---

## Week 2 (August 11 - August 17)
**Tasks Completed:**
* Wrote the code for PCA and RPCA baseline models. 
* Successfully ran the PCA model on the HDFS dataset and matched the paper's high accuracy.

**Tasks Planned for next week:**
* Test the RPCA model and evaluate its performance limitations.
* Acquire the BGL supercomputer dataset, which is much more complex and unstructured.

---

## Week 3 (August 18 - August 24)
**Tasks Completed:**
* Tested the RPCA model and noted its performance limitations compared to PCA.
* Acquired the BGL supercomputer dataset and set up the initial pipeline for it.

**Tasks Planned for next week:**
* Test the baseline models on this new BGL data to evaluate their robustness on complex data.
* Analyze why models might fail on unstructured logs.

---

## Week 4 (August 25 - August 31)
**Tasks Completed:**
* Ran the baseline models on the BGL dataset and proved that they fail on complex, unstructured data.
* Discovered that extracting logs purely by time mixes up unrelated events and ruins accuracy.

**Tasks Planned for next week:**
* Create a better way to extract logs by grouping them by their physical hardware node.
* Research NLP (Natural Language Processing) techniques like Word2Vec to help the AI understand the meaning of the logs.

---

## Week 5 (September 1 - September 7)
**Tasks Completed:**
* Built the node-based log grouping logic to correctly organize the messy BGL logs.
* Added the Word2Vec algorithm to convert the text logs into numerical data the AI can understand.

**Tasks Planned for next week:**
* Replace the basic linear math models with advanced Deep Learning models.
* Design and build a Deep Autoencoder Neural Network.

---

## Week 6 (September 8 - September 14)
**Tasks Completed:**
* Successfully built and trained the Deep Learning Autoencoder on the complex BGL data.
* Proved that the Autoencoder significantly outperforms the older baseline models.

**Tasks Planned for next week:**
* Find a way to automate the anomaly threshold instead of relying on human guessing (the top 5% rule).
* Research statistical methods to make the threshold adapt on its own.

---

## Week 7 (September 15 - September 21)
**Tasks Completed:**
* Added the dynamic POT (Peaks-Over-Threshold) mathematical algorithm to our pipeline.
* Successfully made the system autonomous, allowing it to flag anomalies automatically without human input.

**Tasks Planned for next week:**
* Audit the code to ensure there is no unfair data leakage between the training and testing phases.
* Write a strict testing loop to guarantee the results are mathematically stable.

---

## Week 8 (September 22 - September 28)
**Tasks Completed:**
* Implemented a strict 10-fold cross-validation testing loop to ensure zero data leakage.
* Finalized all the code, permanently saved the best models, and locked in a highly robust final accuracy score.

**Tasks Planned for next week:**
* Prepare presentation slides and comprehensive report for the Capstone Review.

---

## Week 9 (September 29 - October 5)
**Tasks Completed:**
* Prepared presentation slides for the Capstone Review.
* Practiced answering anticipated reviewer questions and delivered the final presentation.

**Tasks Planned for next week:**
* Review project feedback and plan future improvements (if any).
