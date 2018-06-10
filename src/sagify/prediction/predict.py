from __future__ import absolute_import

import os
# Do not remove the following line
import sys;sys.path.append("..")  # NOQA


_MODEL_PATH = os.path.join('/opt/ml/', 'model')  # Path where all your model(s) live in


class ModelService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            import keras
            cls.model = keras.models.load_model(os.path.join(_MODEL_PATH, 'model.h5'))
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        clf = cls.get_model()
        return clf.predict(input)


def predict(json_input):
    """
    Prediction given the request input
    :param json_input: [dict], request input
    Example:
    {
        "addition": "112+143"
    }

    :return: [dict], prediction
    """

    def _format_addition(input_str):
        def _format(input_str_num, part_one):
            required_spaces_num = 3 - len(input_str_num)
            spaces = ''
            for _ in range(required_spaces_num):
                spaces += ' '

            return spaces + input_str_num if part_one else input_str_num + spaces

        two_parts = input_str.split('+')
        formatted_part_one = _format(two_parts[0], True)
        formatted_part_two = _format(two_parts[1], False)

        return '{}+{}'.format(formatted_part_one, formatted_part_two)

    addition_str = _format_addition(json_input['addition'])

    from src.encoding_utils import decode_prediction, encode_query
    input_model = encode_query(addition_str)

    prediction = ModelService.predict(input_model)

    result = {
        'result': decode_prediction(prediction)
    }

    return result
