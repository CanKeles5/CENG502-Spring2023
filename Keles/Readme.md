# Patch Slimming for Efficient Vision Transformers

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).
The paper was published at CVPR 2022. Authors aim to reduce the computational costs of doing inference with vision transformers. This is accomplished by pruning a number of the patches in each layer of the network.

We aim to reproduce the DeiT-Small models results for this project.

## 1.1. Paper summary

The paper aims to reduce the computational costs of vision tranformers by reducing the number of patches to be processed in each layer. Authors state that patches in a layer are highly similar and have cosine similarity scores around 0.8 in the last layers of ViTs. For every layer, the impact of each patch on the last feature representation is estimated. Models are pruned starting from the last layer.

For the ViT-Ti models, authors have reduced the number of FLOPs by %45 with only %0.2 top-1 accuracy decrease on the ImageNet dataset.

# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.
To reduce the number of patches in the network, authors calculate a significance score for all the patches. This significance score calculates ... .
We calculate a vector m'l' for each layer that contains information whether a patch is preserved or pruned.

Starting from the last layer of the model, we calculate significance scores for each of the patches in a layer. We select the top r patches with the highest significance scores and preserved them, while discarding the rest. We keep track of the patches to be preserved using a matrix m, with shape [num_layers, num_patches]. Each element along the first dimension represents the patches to be preserved in the corresponding layer. The values are boolean.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.
- How do we prune the model? Setting the mask values to 0 doesnt seem to improve the number of FLOPs.
- Hyper parameters for fine tuning indivudial layers were not provided. We used the hyperparameters that were used to train the models originally becouse of time and resource constraints.
- Search granuality is not provided?

# 3. Experiments and results
## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.
- Run main.py with the command "..."
## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.
- Accucacy: How much can we maintain?
- FLOPs: Did we improve the efficiency of the model?

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
Muhammed Can Keleş can.keles@metu.edu.tr
