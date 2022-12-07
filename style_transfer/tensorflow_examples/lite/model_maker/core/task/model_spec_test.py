# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_examples.lite.model_maker.core.task import model_spec as ms


class ModelSpecTest(tf.test.TestCase):

  def test_get(self):
    spec = ms.get('mobilenet_v2')
    self.assertIsInstance(spec, ms.ImageModelSpec)

    spec = ms.get('average_word_vec')
    self.assertIsInstance(spec, ms.TextModelSpec)

    spec = ms.get(ms.mobilenet_v2_spec)
    self.assertEqual(spec, ms.mobilenet_v2_spec)

    with self.assertRaises(KeyError):
      ms.get('not_exist_model_spec')


class AverageWordVecModelSpecTest(tf.test.TestCase):

  def test_tokenize(self):
    model_spec = ms.AverageWordVecModelSpec()
    text = model_spec._tokenize('It\'s really good.')
    self.assertEqual(text, ['it\'s', 'really', 'good'])

    model_spec = ms.AverageWordVecModelSpec(lowercase=False)
    text = model_spec._tokenize('That is so cool!!!')
    self.assertEqual(text, ['That', 'is', 'so', 'cool'])


if __name__ == '__main__':
  tf.test.main()
