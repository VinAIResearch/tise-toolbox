##### Table of contents
1. [Getting Started](#Getting-Started)
2. [Evaluation Toolbox](#Evaluation-Toolbox)
3. [Text-to-Image Models](#Text-to-Image-Models) 
4. [Benchmark Results](#Benchmark-Results)
5. [Acknowledgments](#Acknowledgments)
6. [Contacts](#Contacts)

# TISE: Bag of Metrics for Text-to-Image Synthesis Evaluation
[Tan M. Dinh](https://di-mi-ta.github.io/),
[Rang Nguyen](https://rangnguyen.github.io/),
[Binh-Son Hua](https://sonhua.github.io/)<br>
VinAI Research, Vietnam
> **Abstract**: 
In this paper, we conduct a study on the state-of-the-art methods for text-to-image synthesis and propose a framework to evaluate these methods. We consider syntheses where an image contains a single or multiple objects. Our study outlines several issues in the current evaluation pipeline: (i) for image quality assessment, a commonly used metric, e.g., Inception Score (IS), is often either miscalibrated for the single-object case or misused for the multi-object case; (ii) for text relevance and object accuracy assessment, there is an overfitting phenomenon in the existing R-precision (RP) and SOA metrics, respectively; (iii) for multi-object case, many vital factors for evaluation, e.g., object fidelity, positional alignment, counting alignment, are largely dismissed; (iv) the ranking of the methods based on current metrics is highly inconsistent with real images. To overcome these issues, we propose a combined set of existing and new metrics to systematically evaluate the methods. For existing metrics, we offer an improved version of IS named IS* by using temperature scaling to calibrate the confidence of the classifier used by IS; we also propose a solution to mitigate the overfitting issues of RP and SOA. For new metrics, we develop counting alignment, positional alignment, object-centric IS, and object-centric FID metrics for evaluating the multi-object case. We show that benchmark with our bag of metrics results in a highly consistent ranking among existing methods, being well-aligned to human evaluation. As a by-product, we create AttnGAN++, a simple but strong baseline for the benchmark by stabilizing the training of AttnGAN using spectral normalization. We also release our toolbox, so-called TISE, for advocating fair and consistent evaluation of text-to-image synthesis models.

Details of our evaluation framework and benchmark results can be found in [our paper](https://arxiv.org/abs/2112.01398):
```bibtex
@inproceedings{dinh2021tise,
    title={TISE: Bag of Metrics for Text-to-Image Synthesis Evaluation},
    author={Tan M. Dinh and Rang Nguyen and Binh-Son Hua},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2022}
}
```
**Please CITE** our paper when TISE is used to help produce published results or is incorporated into other software.

## Getting Started

### Installation

- *Clone this repository*
``` 
git clone https://github.com/VinAIResearch/tise-toolbox.git
cd tise-toolbox
```

- *Setup the environment*
```
conda create -p ./envs python=3.7.3
conda activate ./envs
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

- *Install other dependencies*
    1. We use [CountSeg](https://github.com/GuoleiSun/CountSeg) for the object counter. Please follow the [official repository](https://github.com/GuoleiSun/CountSeg) to install CountSeg.
    2. We use [Detectron2](https://github.com/facebookresearch/detectron2) for the object detector. Please follow this [link](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to install Detectron2.

### Pre-trained models 
Run the below command to download the necessary pre-trained models:
```
python download_scripts/download_pretrained_models.py
```

### Data

#### CUB
Run the below command to download and prepare CUB data:
```
python download_scripts/download_cub_data.py
```

#### MS-COCO
Run the below command to download and prepare MS-COCO (version 2014) data:
```
python download_scripts/download_ms_coco_metadata.py
sh download_scripts/download_ms_coco_images.sh
```

#### Evaluation data 
Run the below command to download the necessary evaluation data:
```
python download_scripts/download_evaluation_data.py
```

## Evaluation Toolbox

### Generating images from test captions
The test captions for each set of metrics can be found in the `captions` folder of each aspect evaluation criteria's subfolder. Please use your text-to-image model to create images from the test captions in these files. We'll go over the structure of evaluation data and how to use it in the following sections.

#### Image Realism, Text Relevance and Counting Alignment
The test data has the format as below:
```
[
    ...
    {
        "caption_id": "", 
        "caption": "",    // raw format 
        ...               // other fields, which are not required for image generation
    },
    ...
]
```
Please use your text-to-image model to generate the image for each `item` in test data. For each `item`, the input caption is `item['caption']` and the generated image is saved with the name as `item['caption_id'].png`.

The sample pseudo code for generating images for these aspect metrics is:

```python
import pickle

with open (f'captions/{XXXX}.pkl', 'rb') as f:
    test_data = pickle.load(f)

GENERATED_IMAGE_DIR = f'images/{YOUR_METHOD}'

for item in test_data:
    caption_id = str(item['caption_id'])
    caption = item['caption']
    generated_image = your_text_to_image_model(caption)
    generated_image.save(f'{GENERATED_IMAGE_DIR}/{caption_id}.png')
```

Please replace `XXXX` with the name of the appropriate test caption file. These test caption files can be found in the `captions` folder of each aspect evaluation criteria 's sub-folder. 

#### Semantic Object Accuracy
We follow the structure of the original version of [SOA](https://github.com/tohinz/semantic-object-accuracy-for-generative-text-to-image-synthesis) for the SOA test caption data. There are `80` pickle files
containing the test captions for each MS-COCO object class. We generate  `3` images for each caption in each file. 

The sample pseudo code for generating images for SOA is:

```python 
import pickle 

with open(label_XX_XX.pkl, "rb") as f:
    label_XX_XX_test_data = pickle.load(f)

GENERATED_IMAGE_DIR = f'images/{YOUR_METHOD}'

for item in label_XX_XX_test_data:
    caption_id = str(item['caption_id'])
    caption = item['caption']
    for idx in range(3):
        generated_image = your_text_to_image_model(caption)
        generated_image.save(f'{GENERATED_IMAGE_DIR}/{label_XX_XX}/{caption_id}_{idx}.png')
```

#### Positional Alignment
The test data has the format as below:
```
{
    "behind" : [
        {
            "caption" : "",     // raw caption
            "caption_id": "",
            ...                 // other fields, which are not required for image generation
        }
        ...
    ],
    "bottom": [ ... ],
    "under" : [ ... ],
    ...
}
```

The sample pseudo code for generating images for PA is:
```python 
import os 
import pickle 

with open("captions/PA_input_captions.pkl", "rb") as f:
    test_data = pickle.load(f)

GENERATED_IMAGE_DIR = f'images/{YOUR_METHOD}'

for positional_word in test_data:
    for item in test_data['positional_word']:
        caption_id = str(item['caption_id'])
        caption = item['caption']
        generated_image = your_text_to_image_model(caption)
        if not os.path.exists(f'{GENERATED_IMAGE_DIR}/{positional_word}'):
            os.makedirs(f'{GENERATED_IMAGE_DIR}/{positional_word}') 
        generated_image.save(f'{GENERATED_IMAGE_DIR}/{positional_word}/{caption_id}.png')
```

For more reference, please see [gen_evaluation_images_coco.sh](text_to_images_models/AttnGAN++/gen_evaluation_images_coco.sh) and [gen_evaluation_images_cub.sh](text_to_images_models/AttnGAN++/gen_evaluation_images_cub.sh) about how to generate evaluation images of our AttnGAN++ model.

### Single-object Text-To-Image Synthesis (CUB)

#### 1. Image Realism

Move to `image_realism` metric folder:

```
cd image_realism
```

- *Improved Inception Score (IS*)*

Please update the argument `METHOD` with the name of your method and run the command below to compute IS* metric.

```
METHOD=attngan++
GENERATED_IMAGE_DIR=images/cub/"$METHOD"
SAVED_RESULT_PATH=results/IS/cub/"$METHOD".txt
GPU_ID=0

python IS/bird/inception_score_star_bird.py \
--gpu "$GPU_ID" \
--image_folder "$GENERATED_IMAGE_DIR" \
--saved_file "$SAVED_RESULT_PATH"
```

- *Fréchet Inception Distance (FID)*

Please update the argument `METHOD` with the name of your method and run the command below to compute FID metric.

```
METHOD=attngan++
GENERATED_IMAGE_DIR=images/cub/"$METHOD"
SAVED_RESULT_PATH=results/FID/cub/"$METHOD".txt
GPU_ID=0

python FID/fid_score.py \
--gpu "$GPU_ID" \
--batch-size 50 \
--path1 "FID/data/bird_val.npz" \
--path2 "$GENERATED_IMAGE_DIR" \
--saved_file "$SAVED_RESULT_PATH"
```

#### 2. Text Relevance

Move to `text_relevance` metric folder:

```
cd text_relevance
``` 

Please update the argument `METHOD` with the name of your method and run the command below to compute RP metric.

```
METHOD=attngan++
GENERATED_IMAGE_DIR=images/cub/"$METHOD"
SAVED_RESULT_PATH=results/cub/"$METHOD".txt
GPU_ID=0

CUDA_VISIBLE_DEVICES="$GPU_ID" \
python RP_cub.py \
--image_dir "$GENERATED_IMAGE_DIR" \
--saved_file_path "$SAVED_RESULT_PATH"
```

### Multi-object Text-To-Image Synthesis (MS-COCO)

#### 1. Image Realism 

Move to `image_realism` metric folder:

```
cd image_realism
```

- *Improved Inception Score (IS*)*

Please update the argument `METHOD` with the name of your method and run the command below to compute IS* metric.

```
METHOD=attngan++
GENERATED_IMAGE_DIR=images/coco/"$METHOD"
SAVED_RESULT_PATH=results/IS/coco/"$METHOD".txt
GPU_ID=0

python IS/coco/inception_score_star_coco.py \
--gpu "$GPU_ID" \
--image_folder "$GENERATED_IMAGE_DIR" \
--saved_file "$SAVED_RESULT_PATH"
```

- *Fréchet Inception Distance (FID)*

Please update the argument `METHOD` with the name of your method and run the command below to compute FID metric.

```
METHOD=attngan++
GENERATED_IMAGE_DIR=images/coco/"$METHOD"
SAVED_RESULT_PATH=results/FID/coco/"$METHOD".txt
GPU_ID=0

python FID/fid_score.py \
--gpu "$GPU_ID" \
--batch-size 50 \
--path1 "FID/data/coco_val.npz" \
--path2 "$GENERATED_IMAGE_DIR" \
--saved_file "$SAVED_RESULT_PATH" 
```

#### 2. Object Fidelity

Move to `object_fidelity` metric folder:

```
cd object_fidelity
```

- *Crop objects.* 

We leverage the generated images from `Image Realism` evaluation for accessing `Object Fidelity`. Hence, you need to evaluate `Image Realism` first or following the Image Realism's instruction to generate the test images. Then, please run the command below to crop objects.

```
METHOD=attngan++
GENERATED_IMAGE_DIR=../image_realism/images/coco/"$METHOD"
SAVED_CROPPED_OBJECTS_DIR=cropped_objects/"$METHOD"
GPU_ID=0

CUDA_VISIBLE_DEVICES="$GPU_ID" \
python crop_object.py \
--source_image_dir "$GENERATED_IMAGE_DIR" \
--saved_cropped_object_dir "$SAVED_CROPPED_OBJECTS_DIR"
```

- *O-IS* 

Please update the argument `METHOD` with the name of your method and run the command below to compute O-IS metric.

```
METHOD=attngan++
CROPPED_OBJECTS_DIR=cropped_objects/"$METHOD"
SAVED_RESULT_PATH=results/O-IS/"$METHOD".txt
GPU_ID=0

python O-IS/object_centric_inception_score.py \
--gpu_id "$GPU_ID" \
--image_dir "$CROPPED_OBJECTS_DIR" \
--saved_file "$SAVED_RESULT_PATH" 
```

- *O-FID* 

Please update the argument `METHOD` with the name of your method and run the command below to compute O-FID metric.

```
METHOD=attngan++
CROPPED_OBJECTS_DIR=cropped_objects/"$METHOD"
SAVED_RESULT_PATH=results/O-FID/"$METHOD".txt
GPU_ID=0

python O-FID/fid_score.py \
--gpu "$GPU_ID" \
--batch-size 50 \
--path1 "O-FID/data/cropped_object_coco.npz" \
--path2 "$CROPPED_OBJECTS_DIR" \
--saved_file "$SAVED_RESULT_PATH" 
```

#### 3. Text Relevance

Move to `text_relevance` metric folder:

```
cd text_relevance
``` 

Please update the argument `METHOD` with the name of your method and run the command below to compute RP metric.

```
METHOD=attngan++
GENERATED_IMAGE_DIR=images/coco/"$METHOD"
SAVED_RESULT_PATH=results/coco/"$METHOD".txt
GPU_ID=0

python RP_coco.py \
--image_dir="$GENERATED_IMAGE_DIR" \
--saved_file_path="$SAVED_RESULT_PATH" 
```

####  4. Positional Alignment

Move to `positional_alignment` metric folder:

```
cd positional_alignment
``` 

Please update the argument `METHOD` with the name of your method and run the command below to compute PA metric.

```
METHOD=attngan++
GENERATED_IMAGE_DIR=images/"$METHOD"
SAVED_RESULT_PATH=results/"$METHOD".txt
GPU_ID=0

CUDA_VISIBLE_DEVICES="$GPU_ID" \
python PA.py \
--image_dir="$GENERATED_IMAGE_DIR" \
--saved_file_path="$SAVED_RESULT_PATH" 
```

####  5. Counting Alignment

Move to `counting_alignment` metric folder:

```
cd counting_alignment
``` 

Please update the argument `METHOD` with the name of your method and run the command below to compute CA metric.

```
METHOD=attngan++
GENERATED_IMAGE_DIR=images/"$METHOD"
SAVED_RESULT_PATH=results/"$METHOD".txt
GPU_ID=0

python CA.py \
--gpu_id="$GPU_ID" \
--image_dir="$GENERATED_IMAGE_DIR" \
--result_file="$SAVED_RESULT_PATH" 
```

####  6. Semantic Object Accuracy

Move to `semantic_object_accuracy` metric folder:

```
cd semantic_object_accuracy
``` 

Please update the argument `METHOD` with the name of your method and run the command below to compute SOA metric.

```
METHOD=attngan++
GENERATED_IMAGE_DIR=images/"$METHOD"
DETECTED_RESULTS_DIR=detected_results/"$METHOD"
SAVED_RESULT_PATH=results/"$METHOD".txt
GPU_ID=0

CUDA_VISIBLE_DEVICES="$GPU_ID" \
python SOA.py \
--images="$GENERATED_IMAGE_DIR" \
--detected_results="$DETECTED_RESULTS_DIR" \
--saved_file="$SAVED_RESULT_PATH"
```

####  7. Ranking Score

- Please add the score for each aspect metric of your method to the file named as `<YOUR_METHOD>.json` in the `methods` folder. The format is: 
```
{ "FID": "", "IS*": "", "O-IS": "", "O-FID": "", "CA": "", "PA": "", "SOA-I": "", "SOA-C": "", "RP": ""}
```

- Run the below command to compute the ranking score of your method compared to other ones. The results can be found in `results` folder or on terminal screen.

```
python ranking_score.py
```

## Text-to-Image Models 
Below is a list of related text-to-image generation models we use in our benchmark with their codes.
- <a name="GAN-INT-CLS"></a> **[GAN-INT-CLS]** Generative Adversarial Text to Image Synthesis [[paper](https://arxiv.org/abs/1605.05396)] [[code](https://github.com/reedscot/icml2016)]
- <a name="StackGAN"></a> **[StackGAN]** Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks [[paper](https://arxiv.org/abs/1710.10916)] [[code](https://github.com/hanzhanggit/StackGAN)]
- <a name="StackGAN++"></a> **[StackGAN++]** Realistic Image Synthesis with Stacked Generative Adversarial Networks [[paper](https://arxiv.org/abs/1710.10916)] [[code](https://github.com/hanzhanggit/StackGAN-v2)]
- <a name="AttnGAN"></a> **[AttnGAN]** Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks [[paper](https://arxiv.org/abs/1711.10485)] [[code](https://github.com/taoxugit/AttnGAN)]
- <a name="DM-GAN"></a> **[DM-GAN]** Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis [[paper](https://arxiv.org/abs/1904.01310)] [[code](https://github.com/MinfengZhu/DM-GAN)]
- <a name="CP-GAN"></a> **[CPGAN]** Full-Spectrum Content-Parsing Generative Adversarial Networks for Text-to-Image Synthesis [[paper](https://arxiv.org/abs/1912.08562)] [[code](https://github.com/dongdongdong666/CPGAN)]
- <a name="DF-GAN"></a> **[DF-GAN]** Deep Fusion Generative Adversarial Networks for Text-to-Image Synthesis [[paper](https://arxiv.org/abs/2008.05865)] [[code](https://github.com/tobran/DF-GAN)]
- <a name="AttnGAN + CL"></a> **[AttnGAN + CL]** Improving Text-to-Image Synthesis Using Contrastive Learning [[paper](https://arxiv.org/abs/2107.02423)] [[code](https://github.com/huiyegit/T2I_CL/tree/main/AttnGAN%2BCL)]
- <a name="DM-GAN + CL"></a> **[DM-GAN + CL]** Improving Text-to-Image Synthesis Using Contrastive Learning [[paper](https://arxiv.org/abs/2107.02423)] [[code](https://github.com/huiyegit/T2I_CL/tree/main/DM-GAN%2BCL)]
- <a name="DALL·E Mini"></a> **[DALL·E Mini]** Generate images from a text prompt [[code](https://github.com/borisdayma/dalle-mini)]
- <a name="AttnGAN++"></a> **[AttnGAN++]** Revisiting the Attentional Generative Adversarial Network for Text-To-Image Synthesis [[our paper](https://arxiv.org/abs/2112.01398)] [[our code](https://github.com/VinAIResearch/tise-toolbox/tree/master/text_to_images_models/AttnGAN%2B%2B)]

## Benchmark Results 

### Single-object Text-To-Image Synthesis (CUB)

<details><summary>CLICK TO VIEW</summary>

| Method       |   IS*   |    FID  |    RP   |   
|:-------------|:------: |:-------:|:------: |
| GAN-INT-CLS  |  7.51   |  194.41 |  3.83   |  
| StackGAN++   |  12.69  |  27.40  |  13.57  |  
| AttnGAN      |  13.63  |  24.27  |  65.30  |  
| AttnGAN + CL |  14.42  |  17.96  |  60.82  |  
| DM-GAN       |  15.00  |  15.52  |  76.25  |  
| DM-GAN + CL  |  15.08  |  14.57  |  69.80  |  
| DF-GAN       |  14.70  |  16.46  |  42.95  |   
| AttnGAN++    |  15.13  |  15.01  |  77.31  | 
</details>


### Multi-object Text-To-Image Synthesis (MS-COCO)

<details><summary>CLICK TO VIEW</summary>

| Method       |   IS* |    FID |    RP |   SOA-C |   SOA-I |   O-IS |   O-FID |   CA |     PA |   RS |
|:-------------|------:|-------:|------:|--------:|--------:|-------:|--------:|-----:|-------:|-----:|
| GAN-CLS      |  8.1  | 192.09 | 10    |    5.31 |    5.71 |   2.46 |   51.13 | 2.51 |  32.79 |  7   |
| StackGAN     | 15.5  |  53.44 |  9.1  |    9.24 |    9.9  |   3.36 |   29.09 | 2.41 |  34.33 | 11.5 |
| AttnGAN      | 33.79 |  36.9  | 50.56 |   47.13 |   49.78 |   5.04 |   20.92 | 1.82 |  40.08 | 29   |
| DM-GAN       | 45.63 |  28.96 | 66.98 |   55.77 |   58.11 |   5.22 |   17.48 | 1.71 |  42.83 | 41   |
| CPGAN        | 59.64 |  50.68 | 69.08 |   81.86 |   83.83 |   6.38 |   20.07 | 2.07 |  43.28 | 43   |
| DF-GAN       | 30.45 |  21.05 | 42.44 |   37.85 |   40.19 |   5.12 |   14.39 | 1.96 |  40.39 | 31.5 |
| AttnGAN + CL | 36.85 |  26.93 | 57.52 |   47.45 |   49.33 |   4.92 |   19.92 | 1.72 |  43.92 | 37   |
| DM-GAN + CL  | 46.61 |  22.6  | 70.36 |   58.68 |   61.05 |   5.09 |   15.5  | 1.66 |  49.06 | 51.5 |
| DALLE-Mini (zero-shot)   | 19.82 |  62.9  | 48.72 |   26.64 |   27.9  |   4.1  |   23.83 | 2.31 |  47.39 | 23.5 |
| AttnGAN++    | 54.63 |  26.58 | 72.48 |   67.83 |   69.97 |   6.01 |   15.43 | 1.57 |  47.75 | 56   |
| Real-Images  | 51.25 |   2.62 | 83.54 |   90.02 |   91.19 |   8.63 |    0    | 1.05 | 100    | 65   |
</details>


## Acknowledgments
Our code borrowed some parts of the official repositories of text-to-image models, which are used in our benchmark.
Thank you so much to the authors for their efforts to release source code and pre-trained weights.

## Contacts
If you have any questions, please drop an email to _tan.m.dinh.vn@gmail.com_ or open an issue in this repository.


