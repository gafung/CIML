# Environment
- Windows 10
- Python 3.5.3 :: Continuum Analytics, Inc.


# How to run live demo
1. Download Python from [Anaconda](https://anaconda.org/)
2. Open Anaconda Prompt
3. Create new environment & activate
```commandline
conda create python=3.5 -n insider_env
activate insider_env

```
4. Install dependencies & start local web server
```commandline
cd "Live Demo"
pip install -r requirements.txt
python ciml.py
```
5. Open web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).
![Screenshot][screenshot]

[screenshot]: https://github.com/gafung/CIML/blob/master/README.png