clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_c_file.sh {} \;
	rm -rf coverage
	rm -rf dist
	rm -rf build
	rm -rf doc/_build
	rm -rf doc/auto_examples
	rm -rf doc/generated
	rm -rf doc/modules
	rm -rf examples/.ipynb_checkpoints

install: clean
	#pip uninstall exposing --yes
	python setup.py install --record files.txt
	cat files.txt | xargs rm -rf
	rm files.txt
	python setup.py install --force

upload:
	python setup.py sdist upload -r pypi
	# pip install --upgrade stream-learn
