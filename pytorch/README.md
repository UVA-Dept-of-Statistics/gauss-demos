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

If you need anything newer, you're out of luck. POWER8 only supports CUDA <= 11.4 (driver <= 470–510). Try the next section to get that working.

Alternative Method
------------------

The original demonstration was run on a [Lambda AI server](https://lambda.ai/). This has nothing to do with the Gauss server, and costs a bit of money. The upside, though, is that you get the most current software and hardware.

The way to run the code is almost the same as the above steps: spin up a server, `scp` all your files on to it, ssh in there, etc. However, before running the python code, you need to install some libraries:

    sudo apt update && python3 -m venv venv
    source venv/bin/activate
    pip3 install transformers[torch] sentencepiece protobuf==3.20 scipy
    python3 train.py 

After these finish installing, remove this portion of code from `train.py` (this patch was only needed for our department's old POWER8 server):

    import os
    
    # ------- POWER8 SAFETY SWITCHES -------
    # These MUST be set before importing torch / transformers
    os.environ["XNNPACK_GLOBAL_DISABLE"] = "1"
    os.environ["ATEN_CPU_CAPABILITY"] = "default"
    os.environ["PYTORCH_ENABLE_MKLDNN"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"   # avoid weird thread spawning too
    # --------------------------------------


