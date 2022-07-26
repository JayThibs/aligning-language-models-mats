cuda:
	sudo /opt/deeplearning/install-driver.sh

conda:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh
	echo "Close the window for Conda to work!"

conda-env:
	conda create --name llm-env python=3.7
	echo "RUN: conda activate llm-env"

pip:
	pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
	pip install -r requirements.txt

testgpt:
	python run_gpt.py ${ARGS}
