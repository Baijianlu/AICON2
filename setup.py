# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:09:30 2019

@author: Tao.Fan
This script used to setup the scripts
"""
try:
    from setuptools import setup, Extension
    use_setuptools = True
    print("setuptools is used.")
except ImportError:
    from distutils.core import setup, Extension
    use_setuptools = False
    print("distutils is used.")
    
#extension_aicon = Extension('aicon._extern', sources = [])
#ext_modules_aicon = [extension_aicon]
packages_aicon = ['aicon']
scripts_aicon = ['Scripts/AICON']

if __name__ == '__main__':

    version_nums = [None, None, None]
    with open("aicon/version.py") as f:
        for line in f:
            if "__version__" in line:
                for i, num in enumerate(line.split()[2].strip('\"').split('.')):
                    version_nums[i] = int(num)
                break


    if None in version_nums:
        print("Failed to get version number in setup.py.")
        raise

    version_number = ".".join(["%d" % n for n in version_nums])
    
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    if use_setuptools:
        setup(name='AICON',
              version=version_number,
              description='This is the AICON module.',
              author='Tao Fan',
              author_email='Tao.Fan@skoltech.ru',
              url='https://github.com/Baijianlu/AICON2.git',
              long_description=long_description,
              long_description_content_type="text/markdown",
              packages=packages_aicon,
              install_requires=['numpy>=1.17.2', 'scipy>=1.3.1', 'pymatgen>=2020.4.2', 'atomate>=0.9.4', 'pymongo>=3.10.1'],
              provides=['aicon'],
              scripts=scripts_aicon,
              classifiers=[
                  "Programming Language :: Python :: 3",
                  "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                  "Operating System :: OS Independent",
              ],
              python_requires=">=3.6")
    else:
        setup(name='AICON',
              version=version_number,
              description='This is the AICON module.',
              author='Tao Fan',
              author_email='Tao.Fan@skoltech.ru',
              url='https://github.com/Baijianlu/AICON2.git',
              long_description=long_description,
              long_description_content_type="text/markdown",
              packages=packages_aicon,
              requires=['numpy', 'scipy', 'pymatgen', 'atomate', 'pymongo'],
              provides=['aicon'],
              scripts=scripts_aicon)
