# EPGD
### Efficient Project Gradient Descent for Ensemble Adversarial Attack
*Fisrt Place Solution for IJCAI19 Targeted Adversarial Attack competition; Sumbitted to IJCAI19 AIBS WORKSHOP* 

https://arxiv.org/abs/1906.03333;

Fanyou Wu, Rado Gazo, Eva Haviarova and Bedrich Benes @ Purdue University

Recent advances show that deep neural networks are not robust to deliberately crafted adversarial examples which many are generated by adding human imperceptible perturbation to clear input. Consider $l_2$ norms attacks, Project Gradient Descent (PGD) and the Carlini and Wagner (C\&W) attacks are the two main methods, where PGD control max perturbation for adversarial examples while C\&W approach treats perturbation as a regularization term optimized it with loss function together. If we carefully set parameters for any individual input, both methods become similar. In general, PGD attacks perform faster but obtains larger perturbation to find adversarial examples than the C\&W when fixing the parameters for all inputs. In this report, we propose an efficient modified PGD method for attacking ensemble models by automatically changing ensemble weights and step size per iteration per input. This method generates smaller perturbation adversarial examples than PGD method while remains efficient as compared to  C\&W method. Our method won the first place in IJCAI19 Targeted Adversarial Attack competition. 


### Usage
* [Download](https://purdue0-my.sharepoint.com/:u:/g/personal/wu1297_purdue_edu/ESqaQdzuv-dEsGY4hlJY6S0BH7Dmosd0UxK2JfnkmH9s8g?e=bQwtEf) model parameters
* Use Docker to set environment. `docker build`
* Use `run.sh ./dev_data ./output` to run codes.

