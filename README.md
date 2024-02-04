# Vesuvius Grand Prize Submission
We just want to read the scrolls!!

![location](https://github.com/SQMah/Vesuvius-Grand-Prize-Submission/blob/main/location.png?raw=true)

**Location of segments within the greater scroll.** The ids correspond to scroll segments with their locations here: [http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/](http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/)

## Methodology
### Minimum System Specs
- 1 TB drive
- 1 Nvidia GPU with high VRAM (I personally tried with 40GB)

### Producing Results
Set up a Linux-based system with CUDA 12.1.

1. Change directory into the folder
2. Run `$ conda env create -f environment.yml`.
3. Run `$ conda activate Vesuvius-Challenge`.
4. Run `$ python data_downloader.py`.
5. Run `$ python data_setup.py`.
6. Place downloaded `[model.ckpt](https://drive.google.com/file/d/1rh0xGOPhznqPT6QqcK6tbnq86eAM9XiI/view?usp=drive_link)` into `./models`.
7. Run `$ accelerate launch inference_unetr_pp.py`.

Results will be saved in the `results/` folder.

### Training
After following the above steps to set up and activate the Conda environment:

1. Run `$ python data_downloader.py`.
2. Run `$ python data_setup.py`.
3. Run `$ python training_unetr_pp.py`.

Trained models will be saved in the `training/` folder.

## Hallucination Mitigation
Hallucinations were mitigated in 4 ways:
1. Labeled data was created using only 64x64 pixel windowed models. The 256x256 pixel windowed model was used to generate cleaner/more legible results only in the last two iterations.
2. Including more negative ink labels. The patch extraction technique described below in technical details extracts more negative labels than positive ones, reducing bias towards positive labels. Hence, the model is less likely to hallucinate positive ink labels. The risk of mis-interpreting via hallucination of negative ink labels is far lower.
3. Strong data augmentation. Distorting augmentations such as optical, grid, and elastic deformations were used during training, greatly reducing the possibility that the models memorize the shape of greek letters instead of learning true ink signals.
4. Results were generated over a 32 pixel stride instead of the 64 pixel stride used during training. Therefore, the results will be generated on unseen parts of characters even if that part of the scroll was used during training.

## Technical Details
I use a custom adaptation of the state of the art [UNETR++ model](https://arxiv.org/abs/2212.04497), a transformer based UNET derivative used in medical imaging as a 3d feature extractor, max pooling over the depth layers, then a final feature extractor based on [Segformer B-5](https://arxiv.org/abs/2105.15203).

We exclusively ran detections on PHerc Paris 3 (scroll 1), with an ink detection of **256x256** pixels, which corresponds to a ~2.02496mm ink detection window, with a stride of 32 pixels to ensure sufficient training data. Since this is larger than the recommended 64x64 pixel detection window, the ways in which I mitigated hallucinations is discussed below.

## Patch Extraction Technique
I propose a patch extraction technique that works well for larger window sizes <=512 pixels as well as allowing the model to have sufficient examples for both positive and negative ink labels. This technique is especially important to learn characters where negative ink labels (negative space) are crucial, for example in distinguishing characters ο, ϲ, and θ, which have very similar ink structures especially when the data is noisy.

![patch example](https://github.com/SQMah/Vesuvius-Grand-Prize-Submission/blob/main/patch.png?raw=true)

The patch extractor works by first identifying all the areas in the manually annotated ink label ground truth data that contain ink, and then only passing the ink area and the surrounding non-ink area that is critical to understanding what character it is. This includes the non-ink labels inside the character itself, which is crucial in distinguishing the aforementioned ο, ϲ, and θ.


Example patch extraction from manually annotated ink labels on PHerc Paris 3 segment 20231012184423. The green boxes denote the area of the scroll that the model will be trained. Note that the stride is 64, hence there are many overlapping boxes.

You can run and visualize the patch extract algorithm yourself using `window_visualizer.py`.


