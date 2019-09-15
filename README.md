# Simple Music Genre Classification

1. Run the `get_data.sh` file to get the data in an organised folder
```bash
chmod 777 get_data.sh
./get_data.sh
``` 

2. Run this command to make the environment ready
```bash
 pip install -r requirements.txt
```

3. Run `data-extract.py` to extract the features for DNN & CNN
```bash
python3 data-extract.py
```
4. Run `CNN.py` or `DNN.py` to classify using CNN or DNN respectively


## Happy Coding