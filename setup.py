from setuptools import setup, find_packages

setup(name='zonopy-robots',
      version='0.0.1',
      install_requires=['zonopy','urchin','scipy'], # scipy for loadmat in jrs trig
      packages=find_packages(),
      package_dir={"": "."},
      package_data = {'zonopyrobots': ['robots/assets/**/*', 'joint_reachable_set/jrs_trig/**/*']}
)
