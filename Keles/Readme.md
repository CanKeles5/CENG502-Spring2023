# Patch Slimming for Efficient Vision Transformers

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper was published at CVPR 2022. Authors aim to reduce the computational costs of doing inference with vision transformers. This is accomplished by pruning a number of the patches in each layer of the network.

We aim to reproduce the DeiT--Small models results for this project. Authors state that they have reduced the number of FLOPs by %43.6 while only a %0.4 drop in accuracy on the ImageNet dataset. While choosing the smaller DeiT-Tiny model would allow us to do more experiments, we were unable to find a pre trained model.

## 1.1. Paper summary

The paper aims to reduce the computational costs of vision tranformers by reducing the number of patches to be processed in each layer. Authors state that patches in a layer are highly similar and have cosine similarity scores around 0.8 in the last layers of ViTs. For every layer, the impact of each patch on the last feature representation is estimated. Models are pruned starting from the last layer.

For the ViT-Ti models, authors have reduced the number of FLOPs by %45 with only %0.2 top-1 accuracy decrease on the ImageNet dataset.

# 2. The method and my interpretation

## 2.1. The original method


<div align="center">

![equation](https://latex.codecogs.com/svg.latex?\text{MSA}(Z_l)%20=%20\text{Concat}_h%20\left(%20\sum_{h=1}^H%20P_{hl}%20V_{hl}%20\right)%20\mathbf{W}_o^l%20=%20\sum_{h=1}^H%20P_{hl}%20Z_{l-1}%20\mathbf{W}_{hv}%20\mathbf{W}_o^l)

</div>


@TODO: Explain the original method.
To reduce the number of patches in the network, authors calculate a significance score for all the patches. This significance score calculates ... .
We calculate a vector m'l' for each layer that contains information whether a patch is preserved or pruned.

Authors show that patches within a layer are mostly redundant as we go deeper in the model. In the last layers, the cosine similratiy between some pathces reaches 0.8. This implies that some of the patches are redundant and can be eliminated without much performance decrease. In this paper, authors propose a method to reduce the number of patches that are fed into the attention layers. For each layer a binary vector ml is used for representing if a patch is preserved or discarded.

PUT PRUNED AND NON-PRUNED MSA & MLP FORMULATIONS

In CNN's pruning channels is common. Pruning channels in ViT's dont work well mainly becouse in ViT's each of the patches correspond to one another in different layers. Authors propose a method where we prune the ViT's in a top down manner. We start from the last layer and selectively eliminate a number of pathces in each layer, while preserving the patches in the previous layer for each layer.

Starting from the last layer of the model, we calculate significance scores for each of the patches in a layer. We select the top r patches with the highest significance scores and preserved them, while discarding the rest. We keep track of the patches to be preserved using a matrix m, with shape [num_layers, num_patches]. Each element along the first dimension represents the patches to be preserved in the corresponding layer.

Impact Estimation:

...

# 2.1.2. Pruning

<p align="center">
  <img src="https://github.com/CanKeles5/CENG502-Spring2023/assets/52157220/bb97de7a-14ff-4ae3-b37b-1dd49aa7346f" alt="Sublime's custom image"/>
</p>


## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.
The paper was easy to understand and covered the necessary details most of the time. Below we list a number of details that we werent able to find in the paper:

- In the paper authors state that they only calcuate attention scores for the patches that have 1 in the corresponding position in the mask, and layer the attention scores are padded to the original input shape before feeding them into the MLP layers. In the paper it is not stated if the positions of the patches are preserved or the zero padding is done by adding zeros after the calculated attention scores. We thought that preserving the position of the attention scores whould be a better choise and we have implemeted the padding in this way.

- Hyper parameters for fine tuning indivudial layers were not provided. We used the hyperparameters that were used to train the models originally becouse of time and resource constraints. These hyperparameters might be crucial becouse we are only training a single layer.

- When calculating the attention scores with the mask, authors formulate a new attention calculating where they use a matrix consisting of the vecotr m's values in its diagonal. If we implement the method directly, the number of FLOPs will not decrease for the pruned model as the input shape hasnt changed. We instead use the vector m[l] as a boolean vector and with indexing we extract the patches we are interested in calculating. After we calculate the attention scores, we pad the output before feeding it to the MLP as described previously. For pruning the MLP, ...

# 3. Experiments and results
## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.
- Run main.py with the command "..."
## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.
There are two metrics we are interested in this project, one is number of FLOPs and the other is how much accuracy can we maintain from the original model.

We have reduced the number of FLOPs by %a.aa. We think that we can further reduce the number of FLOPs by optimizing our implementation.

We have not managed to preserve the accuracy of the model fully. When pruning the models, we need to fine tune the layers for a few epochs on the ImageNet dataset. While authors state that fine tuning a single layer is very computationally cheap, in our PyTorch implementation training a model with only a single layer unfreezed still takes a lot of time. We fine tuned our models on the ImageNet-mini dataset which is a ...

- Accucacy: How much can we maintain?
- FLOPs: Did we improve the efficiency of the model?

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.
! Give reference to the original ppr

# Contact
Muhammed Can Keleş can.keles@metu.edu.tr
