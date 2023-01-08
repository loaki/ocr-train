#%%
import numpy as np
import os

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

#%%

train_data = object_detector.DataLoader.from_pascal_voc(
    'train',
    'train',
    ['v/d', 'w/l', 'player', 'class']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'validate',
    'validate',
    ['v/d', 'w/l', 'player', 'class']
)

#%%

# Model architecture	Size(MB)*	Latency(ms)**	Average Precision***
# EfficientDet-Lite0	4.4         146	            25.69%
# EfficientDet-Lite1	5.8	        259	            30.55%
# EfficientDet-Lite2	7.2	        396	            33.97%
# EfficientDet-Lite3	11.4	    716	            37.70%
# EfficientDet-Lite4	19.9	    1886	        41.96%

spec = model_spec.get('efficientdet_lite4')

#%%

model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)

#%%

model.evaluate(val_data)

#%%

model.export(export_dir='.', tflite_filename='object_detector.tflite')

#%%

model.evaluate_tflite('object_detector.tflite', val_data)

#%%