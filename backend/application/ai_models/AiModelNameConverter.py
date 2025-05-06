from application.ai_models.ai_models import Model_Classification_Type, Model_Classification_Type


class AiModelNameConverter:
    __binary_string = 'binary_classification'
    __multiple_string = 'multiple_classification'

    def convert(ai_model_string):
        
        if ai_model_string == AiModelNameConverter.__binary_string:
            return Model_Classification_Type.Binary
        
        if ai_model_string == AiModelNameConverter.__multiple_string:
            return Model_Classification_Type.Multiple
        
        return None