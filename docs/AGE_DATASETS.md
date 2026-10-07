# Public datasets for checking the age and minor screening

What each dataset can and cannot tell us. Licences are as stated by the source at the time of writing (October 2026); check
them again before using a dataset for anything beyond internal evaluation. Images are never stored in the repository.

| Dataset | Size and content | Ages | Licence | Tests | Used here |
|---|---|---|---|---|---|
| [UTKFace](https://huggingface.co/datasets/py97/UTKFace-Cropped) | 23,708 pre-cropped aligned faces, one per image | exact, 0-116 | non-commercial research | the age model on clean faces | yes: 2,250-face stratified sample, `python -m services.age_eval run` |
| [LAGENDA](https://huggingface.co/datasets/uaebn/lagenda) (Open Images) | 10,000 photographs, face and person boxes with ages, group scenes, many small faces | annotated, 0-95 | CC BY 2.0 | the whole chain in real photographs: detection, pairing, ageing | yes: 750 photographs (600 with a labelled minor), `python -m services.age_eval wild`. MiVOLO v2 was trained on LAGENDA, so its age errors here are optimistic; the detection results are not affected |
| [FairFace](https://huggingface.co/datasets/HuggingFaceM4/FairFace) | 108,501 faces from Flickr, balanced by race and gender | 10-year groups (0-2, 3-9, 10-19, 20-29 ...) | CC BY 4.0 | the age model on faces it was not trained on | yes: validation split, `python -m services.age_eval fairface` |
| [Adience](https://exposing.ai/adience/) | 26,580 Flickr photographs, unconstrained | 8 groups (0-2, 4-6, 8-13, 15-20 ...) | Creative Commons (per photograph) | in-the-wild faces including children | not downloaded |
| [Children in the wild (ICCWD)](https://arxiv.org/abs/2506.10117) | 10,000 image-caption pairs labelled child present or absent | present / absent | CC BY 4.0 | image-level child detection, including fictional and partly visible children | not downloaded: the paper does not say where the images are hosted |
| [YLFW](https://arxiv.org/abs/2301.05776) | children's faces in the wild, for recognition | identity, not age | see paper | hard children's faces | not downloaded |
| Kaggle: [Faces: Age Detection from Images](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset) | about 20,000 faces in the wild with age, gender, ethnicity | exact | see the Kaggle page | more faces, wider age range | not downloaded: needs a Kaggle login |
| Kaggle: [Age Estimation Dataset: Minors (10-30)](https://www.kaggle.com/datasets/axondata/face-recognition-age-estimation-dataset) | 10,000+ consented photographs of people aged 10-30, a folder per year | exact, 10-30 | see the Kaggle page (a vendor sample) | the 14-25 boundary that matters most | not downloaded: needs a Kaggle login |
| Kaggle: [Human Age Recognition](https://www.kaggle.com/c/human-age-recognition) | about 13,000 images in 8 age groups | groups | competition rules | coarse groups only | not downloaded |

## Kaggle downloads

Kaggle only serves files to a signed-in account, and the credentials are the owner's to supply. To let the tools here fetch Kaggle
datasets: create an API token at kaggle.com (Settings, API, Create New Token), save the downloaded file as
`%USERPROFILE%\.kaggle\kaggle.json`, and run `uv pip install kaggle`. Do not paste the token into a chat or commit it.

## What the datasets cannot show

- Ages in LAGENDA and FairFace are annotator judgements, not documents; the true ages of some people differ.
- None of these is a sample of the media this system will see. Recall measured here is an estimate, not a guarantee.
- Pasted-face tests (a child's face pasted into a scene at a chosen size) measure only face detection at that size.
