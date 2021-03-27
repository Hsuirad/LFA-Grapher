import setuptools

full_desc = ""

with open("DESC.md", 'r', encoding='utf-8') as fh:
    full_desc = fh.read()

setuptools.setup(
    name="chromatography-grapher-HSUIRAD",
    version="2.1.6",
    author="Dariush Aligholizadeh and Alan Mao",
    author_email="daligho1@umbc.edu",
    description="A simple graphing tool for graphing pictures of chromatography from grayscale values",
    long_description=full_desc,
    long_description_content_type = 'text/markdown',
    url='https://github/com/Hsuirad/chromatography-grapher',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires = ['numpy', 'statsmodels', 'pillow', 'matplotlib', 'opencv-python', 'scipy'],
    packages=setuptools.find_packages(),
    python_requires=">=3.6"
)
