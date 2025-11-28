PyTorch Demo
============================================

Overview
--------
This short tutorial shows how to transfer demo files to Gauss, activate the
shared PyTorch environment, train a IMDB sentiment model on all four P100 GPUs, and then run
predictions using the trained model.


1. Send Your Demo Files to the Server
-------------------------------------
Replace "YOURNAME" with your Gauss username.

    scp -r demo_files/ YOURNAME@gauss.stat.virginia.edu:/home/YOURNAME/


2. SSH Into the Server
----------------------
    ssh YOURNAME@gauss.stat.virginia.edu


3. Activate the PyTorch Anaconda Environment
--------------------------------------------
This assumes you already followed the startup tutorial and installed miniconda!

    conda activate /opt/conda_envs/torch21_runtime/


4. Train the Model
------------------
This uses all 4 NVIDIA P100 GPUs and takes a couple minutes.

The trained model will be saved to: `./imdb-sentiment-model.pt`

Run training:
    cd ~/gauss_demos/pytorch/sentiment-demo
    python train.py


5. Generate Predictions With the Trained Model
----------------------------------------------
    python evaluate.py


Credits
-------
Huge thanks to [@benjarison](https://github.com/benjarison) for creating this demo.

Note:
-----
This environment will give you:

NVIDIA drivers: 470.182.03   
CUDA Version: 11.4
Transformers: 4.36.2
PyTorch: 2.1.0

If you need anything newer, you're out of luck. POWER8 only supports CUDA <= 11.4 (driver <= 470–510)
