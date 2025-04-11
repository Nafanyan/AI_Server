from application.ai_models.ai_models import CNN_Model_Name, LNN_Model_Name


class AiModelNameConverter:
    __yolov_string = 'yolov5'
    __binary_string = 'binary_classification'
    __multiple_string = 'multiple_classification'

    def convert(ai_model_string):
        if ai_model_string == AiModelNameConverter.__yolov_string:
            return CNN_Model_Name.YOLOV5
        
        if ai_model_string == AiModelNameConverter.__binary_string:
            return LNN_Model_Name.Binary
        
        if ai_model_string == AiModelNameConverter.__multiple_string:
            return LNN_Model_Name.Multiple

        return None