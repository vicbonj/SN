# Cosmological parameters with SNIa

With conda, in your console in iOS:

git clone https://github.com/vicbonj/SN.git

cd SN/

conda create -n env_for_SN python=3.6

source activate env_for_SN

gcc -dynamiclib -o testlib_total.dylib -lm -fPIC testlib_total.c
gcc -dynamiclib -o testlib.dylib -lm -fPIC testlib.c

pip3 install -r requirements.txt

ipython3

%run mcmc_lcdm.py
