from setuptools import setup, find_packages

setup(
    name='PFVPR',
    version='0.8.0',
    install_requires=[
        'numpy>=1.18.1',
        'scipy>=1.4.1',
        'tqdm',
        'matplotlib',
        'tabulate',
        'ffmpeg'
    ],
    scripts=[
        'src/data/interpolate_raw_data.py',
        'src/data/create_reference_maps.py',
        'src/data/create_query_traverses.py',
        'src/models/ParticleFilter.py',
        'src/models/TopologicalFilter.py',
        'src/models/SeqMatching.py',
        'src/models/SingleImageMatching.py',
        'src/figures/tables.py',
        'src/figures/results.py'
    ]
)
