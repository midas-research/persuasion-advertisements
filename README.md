# Persuasion in Advertisements

Website live at the link: https://midas-research.github.io/persuasion-advertisements/

![Persuasion strategies in advertisements. Marketers use both text and vision modalities to create ads containing different messaging strategies. Different persuasion strategies are constituted by using various rhetorical devices such as slogans, symbolism, colors, emotions, allusion.](img/flowchart.png "Persuasion Strategies present in Advertisements")

Modeling what makes an advertisement persuasive, i.e., eliciting the desired response from consumer, is critical to the study of propaganda, social psychology, and marketing. Despite its importance, computational modeling of persuasion in computer vision is still in its infancy, primarily due to the lack of benchmark datasets that can provide persuasion-strategy labels associated with ads. Motivated by persuasion literature in social psychology and marketing, we introduce an extensive vocabulary of persuasion strategies and build the first ad image corpus annotated with persuasion strategies. We then formulate the task of persuasion strategy prediction with multi-modal learning, where we design a multi-task attention fusion model that can leverage other ad-understanding tasks to predict persuasion strategies. Further, we conduct a real-world case study on 1600 advertising campaigns of 30 Fortune-500 companies where we use our model’s predictions to analyze which strategies work with different demographics (age and gender). The dataset also provides image segmentation masks, which labels persuasion strategies in the corresponding ad images on the test split. We publicly release our code and dataset

![Different persuasion strategies are used for market- ing the same product (footwear in this example).](img/description.jpg "Examples of Persuasion Strategies present in Advertisements")

The annotations can be accessed by visiting the website above or visting the links below:

[Images](https://drive.google.com/drive/folders/1UJ-lQHg0IW_9n4zvp5PJanmPaqGmsw0u?usp=sharing)

[Train Image Annotations](https://github.com/midas-research/persuasion-advertisements/blob/Persuasion-Prediction-Model/Persuasion-Modelling-Code/data/annotations_file_train_set.json)

[Test Image Annotations](https://github.com/midas-research/persuasion-advertisements/blob/Persuasion-Prediction-Model/Persuasion-Modelling-Code/data/annotations_test_set.json)

[Segmentation Masks](https://github.com/midas-research/persuasion-advertisements/blob/Persuasion-Prediction-Model/Persuasion-Modelling-Code/segmentation-masks/AnnotationImageSegmentation_Batch_5_.xml)

[Explore the Dataset](https://midas-research.github.io/persuasion-advertisements/CategoryWisePage.html)

[Model](https://github.com/midas-research/persuasion-advertisements/tree/Persuasion-Prediction-Model)


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

If you use our dataset, please cite the following paper:

[1] Yaman Kumar Singla, Rajat Jha, Arunim Gupta, Milan Aggarwal, Aditya Garg, Ayush Bhardwaj, Tushar, Balaji Krishnamurthy, Rajiv Ratn Shah, and Changyou Chen. "Persuasion Strategies in Advertisements: Dataset, Modeling, and Baselines" (2022). 
