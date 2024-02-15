from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('zonopyrobots/properties.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(name='zonopy-robots',
      version=main_ns['__version__'],
      install_requires=['zonopy','urchin'], # scipy for loadmat in jrs trig
      packages=find_packages(),
      package_dir={"": "."},
      package_data = {'zonopyrobots': ['robots/assets/**/*', 'joint_reachable_set/jrs_trig/**/*']}
)
