from application.ai_models.ai_models import CNN_Model_Name, LNN_Model_Name


class AiModelNameConverter:
    __yolov_string = 'yolov5'

    __lnn_binary_string = 'binary_classification'
    __lnn_multiple_string = 'multiple_classification'

    __cnn_binary_string = 'binary_crossentropy'
    __cnn_multiple_string = 'categorical_crossentropy'

    def convert(ai_model_string):
        if ai_model_string == AiModelNameConverter.__yolov_string:
            return CNN_Model_Name.YOLOV5
        
        if ai_model_string == AiModelNameConverter.__lnn_binary_string:
            return LNN_Model_Name.Binary
        
        if ai_model_string == AiModelNameConverter.__lnn_multiple_string:
            return LNN_Model_Name.Multiple
        
        if ai_model_string == AiModelNameConverter.__cnn_binary_string:
            return CNN_Model_Name.Binary
        
        if ai_model_string == AiModelNameConverter.__cnn_multiple_string:
            return CNN_Model_Name.Multiple

        return None