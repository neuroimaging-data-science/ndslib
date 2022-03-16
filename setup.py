from setuptools import setup
import os.path as op

opts = dict(package_data={'ndslib': [op.join('templates', '*')]})

if __name__ == '__main__':
    setup(**opts)
