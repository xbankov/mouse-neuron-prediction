from setuptools import setup, find_packages

# Read the dependencies from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
print(find_packages(where='src'))
setup(
    name='neuroscience',  # Replace with your project name
    version='0.1.0',  # Specify your project version
    description='Classify stimulus from the mouse neural activity using 3D convolutional network',
    author='Mikuláš Bankovič',  # Replace with your name
    author_email="mikulas.bankovic27@gmail.com",  # Replace with your email
    url='https://github.com/xbankov/neuroscience',  # Replace with your project's GitHub URL
    packages=find_packages(where='src'),  # Replace with your package/module names
    package_dir={'': 'src'},
    install_requires=install_requires,  # Use the dependencies from requirements.txt
)
