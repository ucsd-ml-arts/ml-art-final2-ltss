# Final Project

Lucas Tindall, ltindall@ucsd.edu
Samuel Sunarjo, ssunarjo@eng.ucsd.edu 

## Abstract Proposal

In this project we build a system for real time translation of human faces to cartoon stylized faces. We trained CycleGANs on different cartoon styles to perform the transformations. The end product is a real time application which displays a camera feed where real human faces are replaced by the selected cartoon style. 


## Project Report

Upload your project report (4 pages) as a pdf with your repository, following this template: [google docs](https://docs.google.com/document/d/133H59WZBmH6MlAgFSskFLMQITeIC5d9b2iuzsOfa4E8/edit?usp=sharing).

## Model/Data

Briefly describe the files that are included with your repository:

- CycleGAN folder
- CartoonGAN folder 

- checkpoint model files 

The datasets we used for training can be found in the following repositories
- [CaVI Dataset](https://github.com/lsaiml/CaVINet): includes the real human faces used in all models and the hand drawn caricatures
- [Anime Face Dataset](https://github.com/Mckinsey666/Anime-Face-Dataset): includes anime faces
- [Simpsons Faces Dataset](https://www.kaggle.com/kostastokis/simpsons-faces): includes faces from The Simpsons 
- [Cartoon Set Dataset](https://google.github.io/cartoonset/): includes cartoon avatar faces 

## Code

- camera_gen.py 
- generate.py 

## Results

Documentation of your results in an appropriate format, both links to files and a brief description of their contents:
- What you include here will very much depend on the format of your final project
  - image files (`.jpg`, `.png` or whatever else is appropriate)
  - 3d models
  - movie files (uploaded to youtube or vimeo due to github file size limits)
  - audio files
  - ... some other form

## Technical Notes

Any implementation details or notes we need to repeat your work. 
- Does this code require other pip packages, software, etc?
- Does it run on some other (non-datahub) platform? (CoLab, etc.)

## Reference

References to any papers, techniques, repositories you used:
- Papers
- Repositories
- Blog posts

