install:
	pip3 install -r setup/requirements.txt
	pip3 install -e .
	pip3 install -e src/thirdparty/nigh/
	bash setup/setup.sh
