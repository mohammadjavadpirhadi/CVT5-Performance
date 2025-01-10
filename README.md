# CVT5-Performance
Official Implementation of Performance Comparison of CVT5: Using Compressed Video Encoder and UMT5 for Dense Video Captioning (EvalMG25 @ COLING 2025)

## How to use

*Make sure you have followed the steps in [this repo](https://github.com/mohammadjavadpirhadi/CVT5) to install the requirements!*

Then, run the following commands to measure the performance of the compressed video encoder and the RGB video encoder, respectively:

```bash
python profile_usage.py --config [CONFIG_FILE_NAME]
```

```bash
python rgb_profile_usage.py --config [CONFIG_FILE_NAME]
```

To measure the performance of the model you have to create a ```.json``` file nder the ```configs``` directory (```rgb_configs``` for RGB videos). You can find the configs we have used for different settings under these directories.

*Note: Please update the directories in the config files.*
