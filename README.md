# Knowledge Distillation-Based Approach for Medical Image Analysis
> This is the official implementation of "Knowledge Distillation-Based Approach for Medical Image Analysis"

## Installation
1.Requirments:

You can use the command below to install all requirments easily:

```sh
pip install -r requirements.txt
```

2.Pre-train Teacher-model:

```sh
python train by gpu.py
```

3.Distill the knowledge to the student model:
> Make sure you set the teacher model in right path.
```sh
python knowledge distillation.py
```
## Evaluation

1.SHAP:
```sh
python shap.py
```


2.GradCAM:
> Move the images to choose_folder_path = "./your_path".
```sh
python grad_cam.py
```



