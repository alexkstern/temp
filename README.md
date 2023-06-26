# About

This is a web app where you can add your own documents and search through them semantically or chat with them using chat gpt or gpt 4.


# Installation

```sh
pip install requirements.txt
```


# Usage

First add your openai api key to the environment variables

```sh
export OPENAI_API_KEY={OPENAI_API_KEY}
```


Add pdfs or .txt in an approprietly named folder in data/ (e.g. obesity)

Then run with the specified folder name, e.g.

```sh
python ingest.py obesity
```


Then run the app
```sh
python app.py
```

It will print out a public link that you can use for a couple hours and share the app.
