##The code for paper "Learning to Predict Concept Ordering for Common Sense Generation"
***
###Dataset
The dataset we use are stored in the **commongen** folder, you could unzip the data in the `submission` folder. In the folder, the "commongen.SPLIT.src_alpha.txt" and "commongen.SPLIT.tgt.txt" are the original files provided by CommonGen dataset.
###Model
The model we use in our experiments are stored in the **methods** folder. If you want to run the code, you need to create the virtual environment for each model:
```
    conda create -n MODEL python=3.8
    conda activate MODEL
    pip install -r requirements.txt
```
,where `MODEL` is the name you want to do the corrsponding experiment.
Then you could run the bash files in each sub-folder. We also write the key hyper-parameters in the bash files. We could provide the trained models if needed. 
###Generated sentences
*  We provide all the generated sentences from our experiments in the **generation** folder. It contains 2 sub-folders for development dataset and test dataset. You could verify the results in the paper by using the generated sentence in each sub-folder. For example, if you want to verify the results in Table 2, you could go to `./generation/testset/table2` and evaluate the results with the metrics.

* We use the evaluation metrics (BLEU/ROUGE/METEOR/CIDEr/SPICE/Coverage) provided by [CommonGen project](https://github.com/INK-USC/CommonGen) to evaluate the generation quality. For the Kendall's tau for the concept ordering, you could run the code under `methods/Probabilistic/util.py`
