# Data drift detection using Autoencoders
*Disclaimer for dark mode users: some of the graphics' title in README cannot be read properly in Github Dark Mode*

## Definition
Data drift in this projects means that a new class which has not been seen is introduced during model run.
It can be an anomaly (changed object of the same class on which training was performed) or completely new object of different class

## Approach
![arch drawio](https://user-images.githubusercontent.com/31950564/164248321-b3d34024-0d5f-48ec-984a-9e39c3f52720.png)
Low-dimensional representations (embeddings) of Autoencoders (AE) are used to cluster them to detect new classes through Sihlouete coefficient,
reconstruction error is also used to detect "anomaly" inside one class. Autoencoder is trained without labels on one (good) class.
When the new class is detected a new AE for average representation of this class can be introduced.  

**Following research**

*Researched problem can be treated as small one*

It's unclear wether this approach will work on bigger problems (e.g. higher res images) in order to scale tiling can help

*Tiling big images to create attention maps of a kind to specify drifts*

Tiling images and training AE for each segment of a camera can be done do speficy drifts, moreover this approach can be used in
feredative way to provide better accuracy

*Camera "problems" should be treated as "style"*

So of course, we can say that there is data drift (anomaly if we're talking about one image) when camera is moved, or lightning condition are different,
but question here can be put differently. We actually can say what's wrong can be with camera: lightning conditions, focus, movement, scratches.
And all of these can me modelled (e.g. algoritmically put on images simulating such condition) that gives us two advatages:
1) We can generate data and model drift for such occasions
2) More interesting, given, that we know such things can happen,
can we train an VAEGAN to disentagle style (e.g. camera problems) and features (e.g. objects) and maybe even interpolate car from
information given (e.g. yes, the image is lighted badly, but we can restore bad parts of it and we know with some certainty that this is specified object
and we can say it's not anomaly one) that way we will be more sensitive to react on anomalies.

**Remaining questions:**

- What is maximum capacity for one encoder to distinguish classes?

## Data:
Dataset used for this project was mainly [MVTEC data](https://www.mvtec.com/company/research/datasets/mvtec-ad) you can find its Dataloader at `src` folder.
Dataset needs to be downloaded separately from link provided, no registration / additional fee needed.
## Autoencoders:
Autoencoders are stored in `model` folder. They're written [pytoch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

Supported architectures:
- Variational Autoencoder
- Vanila Autoencoder
- VAEGAN (has not been tested) 

## Results
This is compilation of results from `notebooks` folder, check `.ipynb` files for more details.
### Autoencoders:
- 2 Autoencoders models were trained: "big" and "small". Small AE model embedding size is 8. Big AE model embedding size is 32.
- Variational Autoencoder with embedding size 32 was trained as well and showed similar results to Big AE model.
Big AE can generalize better, this can be seen from PCA's of the same MVTEC data:

PCA on embeddings for Big AE

![Large AE](https://user-images.githubusercontent.com/31950564/164252475-913d590c-d95e-4c0b-9119-e7c259cba73a.png)

PCA on embegging for Small AE

![small AE](https://user-images.githubusercontent.com/31950564/164253757-741c4567-0f64-458e-9d29-fc45b2ef5452.png)

Original bottle images (AE was trained only on good bottles)

![bottles_orig](https://user-images.githubusercontent.com/31950564/164262906-e11cf5c2-31c5-4ea3-baac-3b3e2fc758aa.png)

Bottle reconstructions

![bottles_rec](https://user-images.githubusercontent.com/31950564/164262964-97ba7798-4dfe-487b-9460-fe0f05150174.png)

Original transistor images

![transistors_orig](https://user-images.githubusercontent.com/31950564/164263109-1851ba43-8b34-40b7-9daf-119a2a479067.png)

Reconstructed transistor images

![trainsistors_rec](https://user-images.githubusercontent.com/31950564/164263169-1b7e3329-1611-4e96-a7c4-0a68585721f1.png)

### Clusters:

- Even with 32 dimensions clustering works not the best. DBSCAN which should've solved curse of dimensionality, worked not so good as supposed to.
- Combination with 2-component PCA (explaining around 80% of variance) and K-means clustering using Sihlouhete coefficient to determine cluster
cardinality worked fine ![Kmeans Sihlouehte](https://user-images.githubusercontent.com/31950564/164256395-4858693d-d6a6-4b31-a5c0-c621ce9a1aa3.png)
- One can use GMM instead of K-means clustering to quantify uncertainty in class cardinality.
![GMM_sihloehte](https://user-images.githubusercontent.com/31950564/164256245-e5639431-8e9d-4b1c-99b6-ed88d4298f80.png)  

Dynamic results on MVTEC data:

![output](https://user-images.githubusercontent.com/31950564/164250176-78f82b2d-e6b7-46b4-8f1b-c4ac19666eeb.gif)


## Setup
Before clonning repository download [MVTEC data](https://www.mvtec.com/company/research/datasets/mvtec-ad)

```bash
git clone https://github.com/uncleDecart/data_drift
cd data_drift
pip install -r requirements.txt
jupyter-notebook
```
