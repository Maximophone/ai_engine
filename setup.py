from setuptools import setup, find_packages

setup(
    name='ai_core',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A core library for interacting with various AI models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ai_core', # Replace with your repo URL
    packages=find_packages(include=['ai_core', 'ai_core.*']),
    install_requires=[
        'openai>=1.0.0',
        'anthropic>=0.20.0',
        'google-generativeai>=0.4.0',
        'Pillow>=9.0.0', # For image_utils
        'requests>=2.25.0' # Potentially needed by wrappers or future additions
    ],
    extras_require={
        'test': [
            'python-dotenv>=0.15.0',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose your license
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
) 