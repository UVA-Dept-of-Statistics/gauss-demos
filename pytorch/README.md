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

Alternative Method 1
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

Alternative Method 2
---------------------

Here's how to run the same code on UVa's Rivanna cluster:

1. ssh in to the login node (turn on your vpn first)
------------------

    ssh YOURNAME@login.hpc.virginia.edu

2. get all the code/data from github
-------------------------------------

    git clone https://github.com/UVA-Dept-of-Statistics/gauss-demos.git

3. create an anaconda environment on the login node
--------------------------------------------------

Note that this only needs to be done once. Good news because it takes a bit of time.

    module purge
    module load miniforge/24.3.0-py3.11
    source $(conda info --base)/etc/profile.d/conda.sh
    conda info --base
    # must point to /sfs/weka/.../miniforge/24.3.0-py3.11

    conda remove -n pytorch_rivanna --all -y
    conda create -n pytorch_rivanna python=3.11 -y
    conda activate pytorch_rivanna
    echo $CONDA_PREFIX
    # should say something like /home/<netid>/.conda/envs/pytorch_rivanna

    mamba install -y pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install "transformers[torch]" sentencepiece protobuf==3.20 scipy

If that miniforge version is bad, you can get the most recent version of miniforge with `module spider miniforge`

You might also want to verify that that worked with `python -c "import torch; print(torch.__version__)"` Otherwise your job will fail super fast.

4. tell rivanna to run `python train.py` on a *non* login node
------------------------------------------------------------

You are not allowed to run long running jobs on the poor little login baby node. Run this slurm script to get it all done. 

Edit the particulars if you wish. At the very least, change your email address.

   sbatch run_train_on_rivanna.sh 

5. occasionally check on job status
-----------------------------------

Run this as many times as you want.

    squeue -u YOURNAME

You'll also get start and stop emails, too, as long as you kept that directive in the slurm script.
