# Cosmological parameters with SNIa

With conda, in your console:

git clone https://github.com/vicbonj/SN.git

cd SN/

conda create -n env_for_SN python=3.6

source activate env_for_SN

pip3 install -r requirements.txt

ipython3

%run mcmc_lcdm.py
