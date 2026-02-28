from setuptools import setup, find_packages

setup(
    name="loreal_insight_agent",
    version="2.0.0",
    description="L'Oréal Data Insight Agent – Multi-tool AI data analysis system",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "gradio>=5.29.0",
        "langchain>=1.2.0",
        "langchain-classic>=1.0.0",
        "langchain-community>=0.4.0",
        "langchain-core>=1.2.0",
        "matplotlib>=3.10.0",
        "numpy>=1.26.0",
        "openai>=1.79.0",
        "pandas>=2.2.0",
        "python-dotenv>=1.1.0",
        "seaborn>=0.13.0",
        "SQLAlchemy>=2.0.0",
        "typer>=0.15.0",
        "pydantic>=2.11.0",
        "tabulate>=0.9.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
