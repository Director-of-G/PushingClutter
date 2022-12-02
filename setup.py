from setuptools import setup, find_packages

# Install sliding_pack:
# 1. $ cd /path/to/push_slider
# 2. $ pip3 install .

print(find_packages())

setup(
    name='rrt_pack',
    version='1.0',
    url='https://github.com/motion-planning/rrt-algorithms.git',
    author='SZanlongo and Tahsincan Köse',
    license='Apache v2.0',
    packages=find_packages(),
    package_data={'': ['*.yaml']},
    include_package_data=True,
    install_requires=[
        'numpy',
        'rtree',
        'plotly',
    ],
    zip_safe=False
)
